import os
import re
import sys
import gc
import torch
import codecs
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.data import Data
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric

# Add that directory to sys.path if it's not already there
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from devign.src.utils.objects.cpg.function import Function
from devign.src.utils.functions.parse import clean_gadget, regex_split_operators

AVAILABLE_DATASETS = ["train", "valid", "test"]
NODES_DIM = 205
WORD2VEC_ARGS = {
        "vector_size" : 100, 
        "alpha" : 0.01, 
        "window" : 5, 
        "min_count" : 3, 
        "sample" : 1e-5,
        "workers" : 4, 
        "sg" : 1, 
        "hs" : 0, 
        "negative" : 5
    }
EDGE_TYPE = "Ast"

W2V_KV = None

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset",
    nargs="*",
    help="Select dataset(s). If not provided, all datasets are used.",
    choices=AVAILABLE_DATASETS,
    default=['train']
)
parser.add_argument(
    "--workers",
    type=int,
    default=max(1, (os.cpu_count() or 2) - 1),
    help="Number of worker processes for parallel CPU-bound stages.",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=128,
    help="Rows per chunk to limit peak memory usage.",
)
args = parser.parse_args()


def tokenizer_with_mapping(code, flag=False) -> Dict[int, List[str]]:
    # Dictionary to hold the line-to-token mapping
    line_to_tokens_map = {}
    gadget: List[str] = []
    tokenized: List[str] = []
    # remove all string literals
    no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '', code)
    # remove all character literals
    no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
    code = no_char_lit_line

    if flag:
        try:
            code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]
        except UnicodeDecodeError:
            pattern = re.compile(r'(\\x)([0-9A-Fa-f]{0,1})(?![0-9A-Fa-f])')
            no_char_lit_line = pattern.sub(lambda m: m.group(1) + m.group(2).ljust(2, '0'), no_char_lit_line)
            try:
                code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]
            except UnicodeDecodeError:
                # Keep raw content if malformed escape sequences still exist.
                code = no_char_lit_line

    for line_num, line in enumerate(code.splitlines()):
        if line == '':
            continue
        stripped = line.strip()
        gadget.append(stripped)

        # Process this line using the clean_gadget function
        clean = clean_gadget(gadget)

        # Process each cleaned line, tokenize and map them to their respective line
        for cg in clean:
            if cg == '':
                continue
            # Remove code comments
            pat = re.compile(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)')
            cg = re.sub(pat, '', cg)

            # Remove newlines & tabs
            cg = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', '', cg)
            
            # Tokenize this cleaned line
            splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
            cg_tokens = re.split(splitter, cg)

            # Remove None type and extra spaces
            cg_tokens = list(filter(None, cg_tokens))
            cg_tokens = list(filter(str.strip, cg_tokens))
            # List of tokens
            tokenized.extend(cg_tokens)

            # Map the tokens back to the original line number
            line_to_tokens_map[line_num] = cg_tokens

    return tokenized, line_to_tokens_map

def tokenize_code(code: Any) -> List[str]:
    tokenized, _ = tokenizer_with_mapping(str(code), True)
    return tokenized

def order_nodes(nodes, max_nodes):
    # sorts nodes by line and column

    nodes_by_column = sorted(nodes, key=lambda n: int(nodes[n].get_column_number()))
    nodes_by_line = sorted(nodes_by_column, key=lambda n: int(nodes[n].get_line_number()))

    if len(nodes) > max_nodes:
        # print(f"CPG cut - original nodes: {len(nodes)} to max: {max_nodes}")
        nodes_by_line = nodes_by_line[:max_nodes]

    for i, n in enumerate(nodes_by_line):
        nodes[n].order = i
    
    # Create a nodes by line map
    nodes_by_line_map = {}
    for n in nodes_by_line:
        line = nodes[n].get_line_number()
        code = nodes[n].get_code()
        try:
            nodes_by_line_map[line].append(code)
        except KeyError:
            nodes_by_line_map[line] = [code]

    nodes_by_line_dict = {key: nodes[key] for key in nodes_by_line}
    
    return OrderedDict(nodes_by_line_dict), nodes_by_line_map

def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and
            node.label not in ["Comment", "Unknown"]}

def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    if not cpg or "functions" not in cpg or not cpg["functions"]:
        return None, None
    for function in cpg["functions"]:
        if function is None:
            continue
        func = Function(function)
        # Only nodes with code and line number are selected
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)
        # Order nodes and get code line map
        ordered_nodes, nodes_by_line_map = order_nodes(nodes, max_nodes)

    return ordered_nodes, nodes_by_line_map

def _serialize_nodes_and_edges(ordered_nodes: OrderedDict):
    node_records = []
    node_ids = set(ordered_nodes.keys())
    edge_index = [[], []]

    for node_id, node in ordered_nodes.items():
        node_records.append(
            {
                "node_id": node_id,
                "order": node.order,
                "type": node.type,
                "code": node.get_code(),
                "label": node.label,
            }
        )

    for node_id, node in ordered_nodes.items():
        src_order = node.order
        for edge in node.edges.values():
            if edge.type != EDGE_TYPE:
                continue

            if edge.node_in in node_ids and edge.node_in != node_id:
                edge_index[0].append(ordered_nodes[edge.node_in].order)
                edge_index[1].append(src_order)

            if edge.node_out in node_ids and edge.node_out != node_id:
                edge_index[0].append(src_order)
                edge_index[1].append(ordered_nodes[edge.node_out].order)

    return node_records, edge_index


def process_cpg_to_nodes_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        ordered_nodes, _ = parse_to_nodes(row["cpg"], NODES_DIM)
        if not ordered_nodes:
            return None

        node_records, edge_index = _serialize_nodes_and_edges(ordered_nodes)
        return {
            "id": row["id"],
            "adv": row["adv"],
            "func": row["func"],
            "cpg": row["cpg"],
            "target": int(row["target"]),
            "cwe": row.get("cwe"),
            "node_records": node_records,
            "edge_index": edge_index,
        }
    except Exception as e:
        print(f"[WARN] Failed to parse CPG for id={row.get('id')}: {e}")
        return None


def _init_w2v_worker(keyed_vectors: Word2VecKeyedVectors):
    global W2V_KV
    W2V_KV = keyed_vectors


def _build_node_feature_tensor(node_records: List[Dict[str, Any]], nodes_dim: int):
    kv_size = W2V_KV.vector_size
    target = torch.zeros(nodes_dim, kv_size + 1).float()
    embeddings = []
    code_embedding_mapping = {}

    for rec in node_records:
        node_code = rec["code"]
        if "'\\''" in node_code:
            node_code = node_code.replace("'\''", "'\\").replace("''", "'")

        tokenized_code = tokenize_code(node_code)
        if not tokenized_code:
            continue

        vectors = []
        for token in tokenized_code:
            if token in W2V_KV.key_to_index:
                vectors.append(W2V_KV[token])
            else:
                vectors.append(np.zeros(kv_size))

        source_embedding = np.mean(np.array(vectors), axis=0)
        embedding = np.concatenate((np.array([rec["type"]]), source_embedding), axis=0)
        embeddings.append(embedding)
        code_embedding_mapping[rec["node_id"]] = (node_code, source_embedding)

    if embeddings:
        nodes_tensor = torch.from_numpy(np.array(embeddings)).float()
        target[:nodes_tensor.size(0), :] = nodes_tensor

    return target, code_embedding_mapping


def process_nodes_to_input_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if W2V_KV is None:
            raise RuntimeError("W2V worker is not initialized.")

        x, code_embedding_mapping = _build_node_feature_tensor(row["node_records"], NODES_DIM)
        edge_index = torch.tensor(row["edge_index"]).long()
        label = torch.tensor([row["target"]]).float()
        graph_input = Data(x=x, edge_index=edge_index, y=label)

        return {
            "id": row["id"],
            "adv": bool(row["adv"]),
            "func": row["func"],
            "cpg": row["cpg"],
            "target": int(row["target"]),
            "cwe": row.get("cwe"),
            "input": graph_input,
            "code_embedding_mapping": code_embedding_mapping,
        }
    except Exception as e:
        print(f"[WARN] Failed to build graph input for id={row.get('id')}: {e}")
        return None

def extract_cpg_dict(cpg_data):
    """
    Extract CPG dictionary from either Dict or List[Dict] format.
    
    Args:
        cpg_data: Either a dict or a list containing a dict
        
    Returns:
        dict: The CPG dictionary
    """
    if isinstance(cpg_data, list):
        if len(cpg_data) > 0:
            return cpg_data[0]
        else:
            return None
    return cpg_data

def flip_target(target):
    """
    Flip the target label: 0 -> 1, 1 -> 0
    
    Args:
        target: Original target value
        
    Returns:
        int: Flipped target value
    """
    return 1 if target == 0 else 0

def flatten_dataset(df):
    """
    Transform paired dataset (wide format) into stacked format (long format).
    Each original row becomes two rows:
    - Original row: adv=False, target flipped, using orig_func/orig_cpg
    - Adversarial row: adv=True, target unchanged, using func/cpg
    
    Args:
        df: Input DataFrame with orig_func, orig_cpg, func, cpg columns
        
    Returns:
        pd.DataFrame: Flattened dataset ready for processing
    """
    print("\n[FLATTENING] Transforming dataset from wide to long format...")
    original_count = len(df)
    
    original_rows = []
    adversarial_rows = []
    
    for row in df.itertuples(index=True):
        idx = row.Index
        # Extract CPG dicts from lists if needed
        orig_cpg_data = extract_cpg_dict(getattr(row, 'orig_cpg', None))
        cpg_data = extract_cpg_dict(getattr(row, 'cpg', None))
        
        # Skip if CPG data is invalid
        if orig_cpg_data is None or cpg_data is None:
            print(f"  ⚠ Skipping row {idx}: Invalid CPG data")
            continue
        
        # Ensure target is int
        target_val = int(getattr(row, 'target'))
        
        # Create original row (adv=False)
        original_row = {
            'id': str(idx),
            'adv': False,
            'func': getattr(row, 'orig_func', None),
            'cpg': orig_cpg_data,  # Clean dict
            'target': flip_target(target_val)  # Flipped
        }
        
        # Include optional columns if they exist
        if hasattr(row, 'cwe'):
            original_row['cwe'] = getattr(row, 'cwe')
        
        original_rows.append(original_row)
        
        # Create adversarial row (adv=True)
        adversarial_row = {
            'id': str(idx),
            'adv': True,
            'func': getattr(row, 'func', None),
            'cpg': cpg_data,  # Clean dict
            'target': target_val  # Unchanged
        }
        
        # Include optional columns if they exist
        if hasattr(row, 'cwe'):
            adversarial_row['cwe'] = getattr(row, 'cwe')
        
        adversarial_rows.append(adversarial_row)
    
    # Combine into single DataFrame
    flattened_df = pd.concat([
        pd.DataFrame(original_rows),
        pd.DataFrame(adversarial_rows)
    ], ignore_index=True)
    
    # Ensure correct data types
    flattened_df['target'] = flattened_df['target'].astype(int)
    flattened_df['id'] = flattened_df['id'].astype(str)
    flattened_df['adv'] = flattened_df['adv'].astype(bool)
    
    print(f"  ✓ Original rows: {original_count}")
    print(f"  ✓ Flattened rows: {len(flattened_df)} ({len(original_rows)} original + {len(adversarial_rows)} adversarial)")
    print(f"  ✓ Target distribution: {flattened_df['target'].value_counts().to_dict()}")
    
    return flattened_df


def collect_global_corpus_tokens(df: pd.DataFrame) -> List[List[str]]:
    print("\n[TOKENIZATION] Collecting tokens for entire dataset (one-time pass)...")
    corpus_tokens = []
    for func in df["func"].tolist():
        tokens = tokenize_code(func)
        if tokens:
            corpus_tokens.append(tokens)
    print(f"  ✓ Collected token sequences: {len(corpus_tokens)}")
    return corpus_tokens


def train_word2vec_once(corpus_tokens: List[List[str]]) -> Word2Vec:
    if not corpus_tokens:
        raise ValueError("Token corpus is empty. Cannot train Word2Vec.")

    print("\n[WORD2VEC] Training once on full corpus...")
    w2vmodel = Word2Vec(**WORD2VEC_ARGS)
    w2vmodel.build_vocab(corpus_iterable=corpus_tokens)
    w2vmodel.train(corpus_tokens, total_examples=w2vmodel.corpus_count, epochs=5)
    print("  ✓ Word2Vec training completed.")
    return w2vmodel


def iter_chunks(records: List[Dict[str, Any]], chunk_size: int):
    for i in range(0, len(records), chunk_size):
        yield i, records[i: i + chunk_size]


def process_dataset_parallel(dataset_df: pd.DataFrame, w2vmodel: Word2Vec, workers: int, chunk_size: int) -> pd.DataFrame:
    records = dataset_df.to_dict("records")
    output_rows = []

    with ProcessPoolExecutor(max_workers=workers) as cpg_executor, ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_w2v_worker,
        initargs=(w2vmodel.wv,),
    ) as input_executor, Progress(
        TextColumn("[bold magenta]Processing chunks ({task.completed}/{task.total})..."),
        BarColumn(),
        TextColumn("[bold cyan]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
    ) as progress:
        total_chunks = max(1, (len(records) + chunk_size - 1) // chunk_size)
        main_task = progress.add_task("[magenta]Chunk processing", total=total_chunks)

        for start, chunk in iter_chunks(records, chunk_size):
            cpg_rows = list(cpg_executor.map(process_cpg_to_nodes_row, chunk))
            cpg_rows = [row for row in cpg_rows if row is not None]

            if cpg_rows:
                input_rows = list(input_executor.map(process_nodes_to_input_row, cpg_rows))

                output_rows.extend([row for row in input_rows if row is not None])

            del cpg_rows
            if "input_rows" in locals():
                del input_rows
            gc.collect()

            progress.update(main_task, advance=1)
            print(f"\n  ✓ Processed rows {start} to {min(start + chunk_size, len(records))}")

    if not output_rows:
        return pd.DataFrame(columns=["id", "adv", "func", "cpg", "target", "input"])

    output_df = pd.DataFrame(output_rows)
    columns = ["id", "adv", "func", "cpg", "target", "input", "cwe", "code_embedding_mapping"]
    output_df = output_df[[col for col in columns if col in output_df.columns]]
    return output_df


def enforce_strictly_balanced_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only mirrored pairs by id:
    - exactly 2 rows per id
    - one adv=False and one adv=True
    - one target=0 and one target=1
    """
    if df.empty:
        print("[STRICT] Dropped 0 rows, keeping 0 rows for 0 complete pairs")
        return df

    working_df = df.copy()
    working_df["id"] = working_df["id"].astype(str)
    working_df["adv"] = working_df["adv"].astype(bool)
    working_df["target"] = working_df["target"].astype(int)

    valid_ids = []
    for group_id, group in working_df.groupby("id", sort=False):
        if len(group) != 2:
            continue

        adv_values = set(group["adv"].tolist())
        target_values = set(group["target"].tolist())

        if adv_values == {False, True} and target_values == {0, 1}:
            valid_ids.append(group_id)

    strict_df = working_df[working_df["id"].isin(valid_ids)].reset_index(drop=True)

    dropped_rows = len(working_df) - len(strict_df)
    complete_pairs = len(valid_ids)
    print(f"[STRICT] Dropped {dropped_rows} rows, keeping {len(strict_df)} rows for {complete_pairs} Gold Standard pairs")

    return strict_df

if __name__ == "__main__":

    for dataset in args.dataset:

        print(f"\nGenerating INPUT for {dataset.upper()} dataset")
        print("=" * 80)

        dataset_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_{dataset}.pkl"
        output_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl"

        if os.path.exists(output_path):
            print(f"⚠ Output file already exists and will be overwritten: {output_path}")
            dataset_df = pd.read_pickle(dataset_path)
        else:
            dataset_df = pd.read_pickle(dataset_path)

        print(f"\n✓ Loaded {len(dataset_df)} rows from: {dataset_path}")
        print(f"  Columns: {list(dataset_df.columns)}")
        
        # FLATTEN THE DATASET BEFORE PROCESSING
        dataset_df = flatten_dataset(dataset_df)
        
        total_examples = len(dataset_df)
        print(f"\n✓ Total examples to process after flattening: {total_examples}")

        corpus_tokens = collect_global_corpus_tokens(dataset_df)
        w2vmodel = train_word2vec_once(corpus_tokens)

        output_df = process_dataset_parallel(
            dataset_df=dataset_df,
            w2vmodel=w2vmodel,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )

        output_df = enforce_strictly_balanced_pairs(output_df)

        os.makedirs("tmp/cwe20cfa/w2v", exist_ok=True)
        w2v_path = "tmp/cwe20cfa/w2v/w2vmodel.wv"
        w2vmodel.save(w2v_path)
        output_df.to_pickle(output_path)

        print("\n[FINAL] Saved outputs once at end.")
        print(f"  ✓ Word2Vec model: {w2v_path}")
        print(f"  ✓ Final dataset: {output_path}")
        print(f"  ✓ Final rows: {len(output_df)}")
        if not output_df.empty:
            print(f"  ✓ Unique IDs: {output_df['id'].nunique()}")
            print(f"  ✓ Target distribution: {output_df['target'].value_counts().to_dict()}")
            print(f"  ✓ Adv distribution: {output_df['adv'].value_counts().to_dict()}")