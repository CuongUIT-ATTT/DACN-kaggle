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
    "vector_size": 100,
    "alpha": 0.01,
    "window": 5,
    "min_count": 3,
    "sample": 1e-5,
    "workers": 4,
    "sg": 1,
    "hs": 0,
    "negative": 5,
}
EDGE_TYPE = "Ast"

# Worker-shared keyed vectors (initialized once per process).
W2V_KV: Optional[Word2VecKeyedVectors] = None

# Precompiled regex patterns for tokenizer performance.
STRING_LITERAL_RE = re.compile(r'[\"]([^"\\\n]|\\.|\\\n)*[\"]')
CHAR_LITERAL_RE = re.compile(r"'.*?'")
MALFORMED_HEX_ESCAPE_RE = re.compile(r'(\\x)([0-9A-Fa-f]{0,1})(?![0-9A-Fa-f])')
COMMENT_RE = re.compile(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(//.*)')
ESCAPE_CLEAN_RE = re.compile(r'(\n)|(\\\\n)|(\\\\)|(\t)|(\r)')
SPLITTER_RE = re.compile(r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)')


# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    help="Select dataset(s). If not provided, all datasets are used.",
    choices=AVAILABLE_DATASETS,
    default=["train"],
)
parser.add_argument(
    "--mode",
    type=str,
    default="augmented",
    choices=["augmented", "original"],
    help="Processing mode: 'augmented' (flatten orig/adv pairs) or 'original' (single samples).",
)
parser.add_argument(
    "--workers",
    type=int,
    default=max(1, (os.cpu_count() or 2) - 1),
    help="Number of worker processes for parallel CPU-bound processing.",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=32,
    help="chunksize passed to ProcessPoolExecutor.map for lower IPC overhead.",
)
args = parser.parse_args()


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim
        self.target = torch.zeros(self.nodes_dim, self.kv_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes, code_embedding_mapping = self.embed_nodes(nodes)

        if embedded_nodes.size > 0:
            nodes_tensor = torch.from_numpy(embedded_nodes).float()
            self.target[:nodes_tensor.size(0), :] = nodes_tensor

        return self.target, code_embedding_mapping

    def embed_nodes(self, nodes):
        embeddings = []
        code_embedding_mapping = {}

        for n_id, node in nodes.items():
            node_code = node.get_code()
            if "'\\''" in node_code:
                node_code = node_code.replace("'\''", "'\\").replace("''", "'")

            tokenized_code, _ = tokenizer_with_mapping(node_code, True)
            if not tokenized_code:
                continue

            vectors = self.get_vectors(tokenized_code)
            source_embedding = np.mean(np.array(vectors), axis=0)
            embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            embeddings.append(embedding)
            code_embedding_mapping[n_id] = (node_code, source_embedding)

        if not embeddings:
            return np.array([]), code_embedding_mapping

        return np.array(embeddings), code_embedding_mapping

    def get_vectors(self, tokenized_code):
        vectors = []
        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                vectors.append(np.zeros(self.kv_size))
        return vectors


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)
        return torch.tensor(connections).long()

    def nodes_connectivity(self, nodes):
        coo = [[], []]

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for edge in node.edges.values():
                if edge.type != self.edge_type:
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)

        return coo


def ensure_directories_exist(paths):
    for _, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def tokenizer_with_mapping(code, flag=False) -> Dict[int, List[str]]:
    line_to_tokens_map = {}
    gadget: List[str] = []
    tokenized: List[str] = []

    no_str_lit_line = STRING_LITERAL_RE.sub("", code)
    no_char_lit_line = CHAR_LITERAL_RE.sub("", no_str_lit_line)
    code = no_char_lit_line

    if flag:
        try:
            code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]
        except UnicodeDecodeError:
            no_char_lit_line = MALFORMED_HEX_ESCAPE_RE.sub(
                lambda m: m.group(1) + m.group(2).ljust(2, "0"),
                no_char_lit_line,
            )
            try:
                code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]
            except UnicodeDecodeError:
                code = no_char_lit_line

    for line_num, line in enumerate(code.splitlines()):
        if not line:
            continue

        stripped = line.strip()
        gadget.append(stripped)
        clean = clean_gadget(gadget)

        for cg in clean:
            if not cg:
                continue

            cg = COMMENT_RE.sub("", cg)
            cg = ESCAPE_CLEAN_RE.sub("", cg)
            cg_tokens = SPLITTER_RE.split(cg)
            cg_tokens = list(filter(None, cg_tokens))
            cg_tokens = list(filter(str.strip, cg_tokens))

            tokenized.extend(cg_tokens)
            line_to_tokens_map[line_num] = cg_tokens

    return tokenized, line_to_tokens_map


def tokenize_code(code: Any) -> List[str]:
    tokenized, _ = tokenizer_with_mapping(str(code), True)
    return tokenized


def load_cwe20cfa_dataset(path: str):
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df = df[["func", "target", "cwe"]].dropna()
    return df


def get_cwe_dict(dfs: List[pd.DataFrame]) -> Dict[str, int]:
    cwe_dict = {}
    for df in dfs:
        for cwe_id, number in df.cwe.value_counts().items():
            try:
                cwe = cwe_id[0]
                cwe_dict[cwe] = cwe_dict.get(cwe, 0) + number
            except IndexError:
                pass
    return cwe_dict


def extract_cpg_dict(cpg_data):
    if isinstance(cpg_data, list):
        return cpg_data[0] if cpg_data else None
    return cpg_data


def flip_target(target):
    return 1 if target == 0 else 0


def flatten_dataset(df):
    print("\n[FLATTENING] Transforming dataset from wide to long format...")
    original_count = len(df)

    original_rows = []
    adversarial_rows = []

    for row in df.itertuples(index=True):
        idx = row.Index
        orig_cpg_data = extract_cpg_dict(getattr(row, "orig_cpg", None))
        cpg_data = extract_cpg_dict(getattr(row, "cpg", None))

        if orig_cpg_data is None or cpg_data is None:
            continue

        target_val = int(getattr(row, "target"))

        original_row = {
            "id": str(idx),
            "adv": False,
            "func": getattr(row, "orig_func", None),
            "cpg": orig_cpg_data,
            "target": flip_target(target_val),
        }
        if hasattr(row, "cwe"):
            original_row["cwe"] = getattr(row, "cwe")

        adversarial_row = {
            "id": str(idx),
            "adv": True,
            "func": getattr(row, "func", None),
            "cpg": cpg_data,
            "target": target_val,
        }
        if hasattr(row, "cwe"):
            adversarial_row["cwe"] = getattr(row, "cwe")

        original_rows.append(original_row)
        adversarial_rows.append(adversarial_row)

    flattened_df = pd.DataFrame(original_rows + adversarial_rows)
    if flattened_df.empty:
        return flattened_df

    flattened_df["target"] = flattened_df["target"].astype(int)
    flattened_df["id"] = flattened_df["id"].astype(str)
    flattened_df["adv"] = flattened_df["adv"].astype(bool)

    print(f"  ✓ Original rows: {original_count}")
    print(f"  ✓ Flattened rows: {len(flattened_df)}")
    print(f"  ✓ Target distribution: {flattened_df['target'].value_counts().to_dict()}")

    return flattened_df


def prepare_original_dataset(df):
    """
    For original mode: process func directly without flattening pairs.
    Each row remains as-is with func, cpg, target columns.
    """
    print("\n[PREPARE] Processing original dataset (single samples)...")
    original_count = len(df)

    rows = []
    for idx, row in df.iterrows():
        cpg_data = extract_cpg_dict(getattr(row, "cpg", None))
        if cpg_data is None:
            continue

        target_val = int(getattr(row, "target"))
        sample_row = {
            "id": str(idx),
            "adv": False,
            "func": getattr(row, "func", None),
            "cpg": cpg_data,
            "target": target_val,
        }
        if hasattr(row, "cwe"):
            sample_row["cwe"] = getattr(row, "cwe")
        rows.append(sample_row)

    prepared_df = pd.DataFrame(rows)
    if prepared_df.empty:
        return prepared_df

    prepared_df["target"] = prepared_df["target"].astype(int)
    prepared_df["id"] = prepared_df["id"].astype(str)
    prepared_df["adv"] = prepared_df["adv"].astype(bool)

    print(f"  ✓ Original rows: {original_count}")
    print(f"  ✓ Processed rows: {len(prepared_df)}")
    print(f"  ✓ Target distribution: {prepared_df['target'].value_counts().to_dict()}")

    return prepared_df


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


def order_nodes(nodes, max_nodes):
    nodes_by_column = sorted(nodes, key=lambda n: int(nodes[n].get_column_number()))
    nodes_by_line = sorted(nodes_by_column, key=lambda n: int(nodes[n].get_line_number()))

    if len(nodes) > max_nodes:
        nodes_by_line = nodes_by_line[:max_nodes]

    for i, n in enumerate(nodes_by_line):
        nodes[n].order = i

    nodes_by_line_map = {}
    for n in nodes_by_line:
        line = nodes[n].get_line_number()
        code = nodes[n].get_code()
        if line in nodes_by_line_map:
            nodes_by_line_map[line].append(code)
        else:
            nodes_by_line_map[line] = [code]

    nodes_by_line_dict = {key: nodes[key] for key in nodes_by_line}
    return OrderedDict(nodes_by_line_dict), nodes_by_line_map


def filter_nodes(nodes):
    return {
        n_id: node
        for n_id, node in nodes.items()
        if node.has_code() and node.has_line_number() and node.label not in ["Comment", "Unknown"]
    }


def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    if not cpg or "functions" not in cpg or not cpg["functions"]:
        return None, None

    ordered_nodes = None
    nodes_by_line_map = None

    for function in cpg["functions"]:
        if function is None:
            continue
        func = Function(function)
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)
        ordered_nodes, nodes_by_line_map = order_nodes(nodes, max_nodes)

    return ordered_nodes, nodes_by_line_map


def _init_worker(keyed_vectors: Word2VecKeyedVectors):
    global W2V_KV
    W2V_KV = keyed_vectors


def process_full_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Single worker pipeline to minimize IPC overhead:
    row -> parse CPG -> node embeddings -> edge index -> Data object.
    """
    try:
        if W2V_KV is None:
            raise RuntimeError("W2V worker is not initialized.")

        ordered_nodes, _ = parse_to_nodes(row.get("cpg"), NODES_DIM)
        if not ordered_nodes:
            return None

        nodes_embedding = NodesEmbedding(NODES_DIM, W2V_KV)
        graphs_embedding = GraphsEmbedding(EDGE_TYPE)

        x, code_embedding_mapping = nodes_embedding(ordered_nodes)
        edge_index = graphs_embedding(ordered_nodes)
        label = torch.tensor([int(row["target"])]).float()
        graph_input = Data(x=x, edge_index=edge_index, y=label)

        return {
            "id": str(row["id"]),
            "adv": bool(row["adv"]),
            "func": row["func"],
            "cpg": row["cpg"],
            "target": int(row["target"]),
            "cwe": row.get("cwe"),
            "input": graph_input,
            "code_embedding_mapping": code_embedding_mapping,
        }
    except Exception as e:
        print(f"[WARN] Failed row id={row.get('id')}: {e}")
        return None


def process_dataset_parallel(dataset_df: pd.DataFrame, w2vmodel: Word2Vec, workers: int, chunk_size: int) -> pd.DataFrame:
    records = dataset_df.to_dict("records")
    if not records:
        return pd.DataFrame(columns=["id", "adv", "func", "cpg", "target", "input"])

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(w2vmodel.wv,),
    ) as executor:
        with Progress(
            TextColumn("[bold magenta]Processing rows..."),
            BarColumn(),
            TextColumn("[bold cyan]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[magenta]Worker map", total=len(records))

            results = []
            for item in executor.map(process_full_row, records, chunksize=chunk_size):
                results.append(item)
                progress.update(task, advance=1)

    results_list = [item for item in results if item is not None]

    if not results_list:
        return pd.DataFrame(columns=["id", "adv", "func", "cpg", "target", "input"])

    output_df = pd.DataFrame(results_list)
    columns = ["id", "adv", "func", "cpg", "target", "input", "cwe", "code_embedding_mapping"]
    output_df = output_df[[col for col in columns if col in output_df.columns]]
    return output_df


def enforce_strictly_balanced_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only mirrored pairs by id:
    - exactly 2 rows per id
    - one adv=False and one adv=True
    - one target=0 and one target=1
    
    Only applies to augmented mode.
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


def enforce_original_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For original mode: just filter out rows missing cpg or func.
    """
    if df.empty:
        print("[SANITY] No rows provided")
        return df

    working_df = df.copy()
    valid_mask = (working_df["func"].notna()) & (working_df["cpg"].notna())
    sanity_df = working_df[valid_mask].reset_index(drop=True)

    dropped_rows = len(working_df) - len(sanity_df)
    print(f"[SANITY] Dropped {dropped_rows} rows missing func or cpg, keeping {len(sanity_df)} rows")
    return sanity_df


if __name__ == "__main__":
    for dataset in args.dataset:
        mode = args.mode
        print(f"\nGenerating INPUT for {dataset.upper()} dataset ({mode.upper()} mode)")
        print("=" * 80)

        dataset_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_{dataset}.pkl"
        output_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl"

        if os.path.exists(output_path):
            print(f"⚠ Output file already exists and will be overwritten: {output_path}")

        dataset_df = pd.read_pickle(dataset_path)

        print(f"\n✓ Loaded {len(dataset_df)} rows from: {dataset_path}")
        print(f"  Columns: {list(dataset_df.columns)}")

        # Process dataset based on mode
        if mode == "augmented":
            dataset_df = flatten_dataset(dataset_df)
        else:  # original
            dataset_df = prepare_original_dataset(dataset_df)

        total_examples = len(dataset_df)
        print(f"\n✓ Total examples to process: {total_examples}")

        corpus_tokens = collect_global_corpus_tokens(dataset_df)
        w2vmodel = train_word2vec_once(corpus_tokens)

        output_df = process_dataset_parallel(
            dataset_df=dataset_df,
            w2vmodel=w2vmodel,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )

        # Apply post-processing based on mode
        if mode == "augmented":
            output_df = enforce_strictly_balanced_pairs(output_df)
            output_desc = "Gold Standard pairs"
        else:  # original
            output_df = enforce_original_sanity(output_df)
            output_desc = "raw samples"

        os.makedirs("tmp/cwe20cfa/w2v", exist_ok=True)
        w2v_path = "tmp/cwe20cfa/w2v/w2vmodel.wv"
        w2vmodel.wv.save(w2v_path)
        output_df.to_pickle(output_path)

        gc.collect()

        print("\n[FINAL] Saved outputs:")
        print(f"  ✓ Word2Vec keyed vectors: {w2v_path}")
        print(f"  ✓ Final dataset: {output_path}")
        print(f"  ✓ Final rows: {len(output_df)} ({output_desc})")
        if not output_df.empty:
            print(f"  ✓ Unique IDs: {output_df['id'].nunique()}")
            print(f"  ✓ Target distribution: {output_df['target'].value_counts().to_dict()}")
            if "adv" in output_df.columns:
                print(f"  ✓ Adv distribution: {output_df['adv'].value_counts().to_dict()}")
