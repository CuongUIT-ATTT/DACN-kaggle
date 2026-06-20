import argparse
import os
import sys

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from train import (  # noqa: E402
    DEVICE,
    PROCESS,
    DevignDataset,
    build_devign_model,
    eval_model,
    infer_graph_dims_from_index_and_dir,
)


class FixedNodeDimDataset:
    def __init__(self, dataset, nodes_dim: int):
        self.dataset = dataset
        self.nodes_dim = nodes_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        graph = sample["input"]

        current_nodes = int(graph.x.shape[0])
        if current_nodes > self.nodes_dim:
            graph.x = graph.x[: self.nodes_dim, :]
            if graph.edge_index is not None and graph.edge_index.numel() > 0:
                keep_edges = (graph.edge_index[0] < self.nodes_dim) & (graph.edge_index[1] < self.nodes_dim)
                graph.edge_index = graph.edge_index[:, keep_edges]
        elif current_nodes < self.nodes_dim:
            pad_rows = self.nodes_dim - current_nodes
            padding = torch.zeros(
                pad_rows,
                graph.x.shape[1],
                dtype=graph.x.dtype,
                device=graph.x.device,
            )
            graph.x = torch.cat([graph.x, padding], dim=0)

        sample["input"] = graph
        return sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Devign checkpoint on a processed .pt dataset."
    )
    parser.add_argument(
        "--processed-data-dir",
        required=True,
        help="Directory containing sample_*.pt files and index.csv.",
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Path to index.csv. Defaults to <processed-data-dir>/index.csv.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a trained Devign state_dict checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=PROCESS["batch_size"],
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary classification threshold.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Also compute pair-wise metrics for paired augmented datasets.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save one-row metrics CSV.",
    )
    return parser.parse_args()


def load_state_dict(model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    return checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def infer_checkpoint_nodes_dim(state_dict):
    for key, tensor in state_dict.items():
        if key.endswith("conv1d_1.weight") and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[1])
    return None


def load_checkpoint(model, state_dict):
    model.load_state_dict(state_dict)
    return model


def main():
    args = parse_args()
    index_csv = args.index_csv or os.path.join(args.processed_data_dir, "index.csv")

    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"index.csv not found: {index_csv}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model checkpoint not found: {args.model_path}")

    index_df = pd.read_csv(index_csv)
    required_cols = {"filename", "target", "id"}
    if not required_cols.issubset(index_df.columns):
        raise ValueError(f"{index_csv} missing required columns: {required_cols}")

    data_nodes_dim, emb_size = infer_graph_dims_from_index_and_dir(index_df, args.processed_data_dir)
    state_dict = strip_module_prefix(load_state_dict(args.model_path))
    checkpoint_nodes_dim = infer_checkpoint_nodes_dim(state_dict)
    nodes_dim = checkpoint_nodes_dim or data_nodes_dim

    model = build_devign_model(nodes_dim, emb_size)
    model = load_checkpoint(model, state_dict)
    model.to(DEVICE)
    model.eval()

    dataset = FixedNodeDimDataset(
        DevignDataset(args.processed_data_dir, index_df=index_df),
        nodes_dim=nodes_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Evaluation configuration:")
    print(f"  processed_data_dir: {args.processed_data_dir}")
    print(f"  index_csv: {index_csv}")
    print(f"  model_path: {args.model_path}")
    print(f"  samples: {len(index_df)}")
    print(f"  data_nodes_dim: {data_nodes_dim}")
    print(f"  model_nodes_dim: {nodes_dim}")
    print(f"  emb_size: {emb_size}")
    print(f"  device: {DEVICE}")

    metrics = eval_model(model, loader, threshold=args.threshold, test=args.pairwise)

    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        pd.DataFrame([metrics]).to_csv(args.output_csv, index=False)
        print(f"\nSaved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()
