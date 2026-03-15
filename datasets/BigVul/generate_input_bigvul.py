import os
import re
import sys
import torch
import codecs
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from collections import OrderedDict
from torch_geometric.data import Data
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

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
EXAMPLES_PER_SAVE = 100

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset",
    nargs="*",
    help="Select dataset(s). If not provided, all datasets are used.",
    choices=AVAILABLE_DATASETS,
    default=['train']
)
args = parser.parse_args()


class NodesEmbedding:
    def __init__(self, nodes_dim: int, w2v_keyed_vectors: Word2VecKeyedVectors):
        self.w2v_keyed_vectors = w2v_keyed_vectors
        self.kv_size = w2v_keyed_vectors.vector_size
        self.nodes_dim = nodes_dim

        assert self.nodes_dim >= 0

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.nodes_dim, self.kv_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes, code_embedding_mapping = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        self.target[:nodes_tensor.size(0), :] = nodes_tensor

        return self.target, code_embedding_mapping

    def embed_nodes(self, nodes):
        embeddings = []

        code_embedding_mapping = {}

        for n_id, node in nodes.items():
            # Get node's code
            node_code = node.get_code()

            if "'\\''" in node_code:
                node_code = node_code.replace("'\''", "'\\").replace("''", "'")
            # Tokenize the code
            tokenized_code, line_to_tokens_map = tokenizer_with_mapping(node_code, True)
            if not tokenized_code:
                # print(f"Dropped node {node}: tokenized code is empty.")
                print(f"Empty TOKENIZED from node CODE {node_code}")
                continue
            # Get each token's learned embedding vector
            vectorized_code = np.array(self.get_vectors(tokenized_code, node))
            # The node's source embedding is the average of it's embedded tokens
            source_embedding = np.mean(vectorized_code, 0)
            # The node representation is the concatenation of label and source embeddings
            embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            embeddings.append(embedding)
            # Add node mapping
            code_embedding_mapping[n_id] = (node_code, source_embedding)

        return np.array(embeddings), code_embedding_mapping

    # fromTokenToVectors
    def get_vectors(self, tokenized_code, node):
        vectors = []

        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
                if node.label not in ["Identifier", "Literal", "MethodParameterIn", "MethodParameterOut"]:
                    print(f"No vector for TOKEN {token} in {node.get_code()}.")

        return vectors
    
class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
                if edge.type != self.edge_type:
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)

        return coo

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
            code = codecs.getdecoder("unicode_escape")(no_char_lit_line)[0]

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

def tokenize(data_frame: pd.DataFrame):
    # Apply the tokenizer function to the 'func' column
    data_frame[['tokens', 'line_to_tokens_map']] = data_frame['func'].apply(lambda code: pd.Series(tokenizer_with_mapping(code)))
    
    # Return the DataFrame with both 'tokens' and 'line_to_tokens_map'
    return data_frame[['tokens', 'line_to_tokens_map']]

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
    for function in cpg["functions"]:
        func = Function(function)
        # Only nodes with code and line number are selected
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)
        # Order nodes and get code line map
        ordered_nodes, nodes_by_line_map = order_nodes(nodes, max_nodes)

    return ordered_nodes, nodes_by_line_map

def nodes_to_input(nodes, target, nodes_dim, keyed_vectors, edge_type):
    nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    x, code_embedding_mapping = nodes_embedding(nodes)
    edge_index=graphs_embedding(nodes)

    return Data(x=x, edge_index=edge_index, y=label), code_embedding_mapping

def process_cpg_to_nodes_row(row):
    cpg = eval(row.cpg)
    ordered_nodes, nodes_by_line_map = parse_to_nodes(cpg, NODES_DIM)
    return pd.Series({"nodes": ordered_nodes, "nodes_by_line_map": nodes_by_line_map})

def process_nodes_to_input_row(row, w2vmodel):
    input_series, code_embedding_mapping = nodes_to_input(row.nodes, row.target, NODES_DIM, w2vmodel.wv, EDGE_TYPE)
    return pd.Series({"input": input_series, "code_embedding_mapping": code_embedding_mapping})   

if __name__ == "__main__":

    for dataset in args.dataset:

        print(f"\nGenerating CPG for BigVul dataset")
        print("-----------------------------------------")

        dataset_path = "datasets/BigVul/bigvul_CWE-20.pkl"
        output_path = "datasets/BigVul/bigvul_CWE-20_input.pkl"

        if os.path.exists(output_path):
            dataset_df = pd.read_pickle(dataset_path)
            output_df = pd.read_pickle(output_path)
            df_init = True

            dataset_df = dataset_df[~dataset_df.index.isin(output_df.index)]

        else:
            df_init = False
            dataset_df = pd.read_pickle(dataset_path)

        total_examples = len(dataset_df)

        # Model initialization
        w2vmodel = Word2Vec(**WORD2VEC_ARGS)

        # Setup rich progress bar
        with Progress(
            TextColumn("[bold magenta]Processing {task.fields[dataset]} ({task.completed}/{task.total})..."),
            BarColumn(),
            TextColumn("[bold cyan]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress:
            
            main_task = progress.add_task(f"[magenta]Processing {dataset.upper()} dataset", 
                                            total=total_examples, dataset=dataset.upper(),)

            w2v_init = True
            i = 0
            for index, row_series in dataset_df.copy().iterrows():

                row_df = row_series.to_frame().T
                # Function Tokenization
                tokenized_func_df = tokenize(row_df)
                func_tokens = tokenized_func_df.tokens

                # Build and Train Word2Vec Model
                w2vmodel.build_vocab(corpus_iterable=func_tokens, update=not w2v_init)
                w2vmodel.train(func_tokens, total_examples=w2vmodel.corpus_count, epochs=1)

                # Embed cpg to node representation and pass to graph data structure
                row_df[["nodes", "nodes_by_line_map"]] = row_df.apply(process_cpg_to_nodes_row, axis=1)
                
                # remove rows with no nodes
                row_df = row_df.loc[row_df.nodes.map(len) > 0]
                
                # Apply the function and create both "input" and "map" columns
                row_df[["input", "code_embedding_mapping"]] = row_df.apply(lambda row: process_nodes_to_input_row(row, w2vmodel), axis=1)

                progress.update(main_task, advance=1)
                i += 1
                
                if not df_init:
                    output_df = row_df
                    df_init = True
                else:
                    output_df = pd.concat([output_df, row_df])

                if w2v_init:
                    w2v_init = False 
                
                if i % EXAMPLES_PER_SAVE == 0:
                    output_df.to_pickle(output_path)
                    print(f"Saved dataset at {output_path}")
                    # Save Word2Vec model
                    w2vmodel.save('tmp/BigVul/w2v/w2vmodel.wv')

            w2vmodel.save('tmp/BigVul/w2v/w2vmodel.wv')
            output_df.to_pickle(output_path)
            print(f"Final dataset saved at {output_path}")