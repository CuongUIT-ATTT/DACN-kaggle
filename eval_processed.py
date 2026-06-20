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


def load_checkpoint(model, model_path: str):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
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

    nodes_dim, emb_size = infer_graph_dims_from_index_and_dir(index_df, args.processed_data_dir)
    model = build_devign_model(nodes_dim, emb_size)
    model = load_checkpoint(model, args.model_path)
    model.to(DEVICE)
    model.eval()

    dataset = DevignDataset(args.processed_data_dir, index_df=index_df)
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
    print(f"  nodes_dim: {nodes_dim}")
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
