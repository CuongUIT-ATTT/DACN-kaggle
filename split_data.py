import os
import gc
import sys
import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric


DEFAULT_INPUT_PKL = "datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl"
FALLBACK_INPUT_PKL = "datasets/cwe20cfa/we20cfa_CWE-20_augmented_input_balanced.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a large DataFrame .pkl into per-sample .pt files for lazy loading."
    )
    parser.add_argument(
        "--input-pkl",
        default=DEFAULT_INPUT_PKL,
        help=(
            "Path to input DataFrame .pkl "
            f"(default: {DEFAULT_INPUT_PKL})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="processed_data",
        help="Directory to store split .pt files and index.csv",
    )
    parser.add_argument(
        "--input-col",
        default="input",
        help="Column name containing torch_geometric graph objects",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name containing labels (0/1)",
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help="Column name containing sample identifiers",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=200,
        help="Run gc.collect() every N samples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if they exist",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, input_col: str, target_col: str, id_col: str) -> None:
    missing = [c for c in (input_col, target_col, id_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def normalize_target(value: Any) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid target value: {value}") from exc


def main() -> None:
    args = parse_args()

    # Graceful fallback for common filename typo variants.
    if not os.path.exists(args.input_pkl) and args.input_pkl == DEFAULT_INPUT_PKL and os.path.exists(FALLBACK_INPUT_PKL):
        args.input_pkl = FALLBACK_INPUT_PKL

    os.makedirs(args.output_dir, exist_ok=True)
    index_csv_path = os.path.join(args.output_dir, "index.csv")

    print(f"Loading DataFrame from: {args.input_pkl}")
    df = pd.read_pickle(args.input_pkl)
    print(f"Loaded {len(df)} rows")

    validate_columns(df, args.input_col, args.target_col, args.id_col)

    records = []
    skipped = 0

    # Iterate row-by-row to avoid creating extra in-memory copies.
    iterator = df.itertuples(index=False)
    for i, row in enumerate(tqdm(iterator, total=len(df), desc="Splitting samples")):
        try:
            graph = getattr(row, args.input_col)
            target = normalize_target(getattr(row, args.target_col))
            sample_id = str(getattr(row, args.id_col))

            sample = {
                "input": graph,
                "target": target,
                "id": sample_id,
            }

            filename = f"sample_{i}.pt"
            file_path = os.path.join(args.output_dir, filename)

            if not args.overwrite and os.path.exists(file_path):
                records.append({"sample_index": i, "filename": filename, "id": sample_id, "target": target})
                continue

            # Keep tensors on CPU and use protocol 4 for broad compatibility.
            torch.save(sample, file_path, pickle_protocol=4)
            records.append({"sample_index": i, "filename": filename, "id": sample_id, "target": target})

            if (i + 1) % max(1, args.gc_every) == 0:
                gc.collect()

        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skip row {i}: {exc}")

    index_df = pd.DataFrame(records)
    index_df.to_csv(index_csv_path, index=False)

    print(f"Done. Wrote {len(records)} samples to: {args.output_dir}")
    print(f"Index file: {index_csv_path}")
    if skipped:
        print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
