import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from train import (  # noqa: E402
    DEVICE,
    PROCESS,
    DevignDataset,
    build_devign_model,
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


def flatten_batches(values):
    flattened = []
    for value in values:
        flattened.extend(value.reshape(-1).tolist())
    return flattened


def evaluate_logits_model(model, dataloader, threshold: float, pairwise: bool):
    loss_list = []
    labels = []
    predicts = []
    probabilities = []
    logits = []
    pair_groups = {}
    loss_lambda = 1e-6

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            graph = batch["input"].to(DEVICE)
            target = graph.y
            logit = model(graph)
            prob = torch.sigmoid(logit)
            pred = (prob >= threshold).float()
            loss = F.binary_cross_entropy_with_logits(logit, target) + F.l1_loss(logit, target) * loss_lambda

            loss_list.append(float(loss.item()))
            labels.append(target.detach().cpu())
            predicts.append(pred.detach().cpu())
            probabilities.append(prob.detach().cpu())
            logits.append(logit.detach().cpu())

            if pairwise:
                for sample_id, pred_value, true_value in zip(
                    batch["id"],
                    pred.detach().cpu().view(-1).long().tolist(),
                    target.detach().cpu().view(-1).long().tolist(),
                ):
                    pair_groups.setdefault(str(sample_id), []).append((pred_value, true_value))

    y_true = flatten_batches(labels)
    y_pred = flatten_batches(predicts)
    y_prob = flatten_batches(probabilities)
    y_logit = flatten_batches(logits)

    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0.0, 1.0])
    tn, fp, fn, tp = confusion.ravel()

    print(f"\nConfusion matrix:\n{confusion}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    print(
        "Probability summary: "
        f"min={min(y_prob):.6f}, mean={sum(y_prob) / len(y_prob):.6f}, max={max(y_prob):.6f}"
    )
    print(
        "Logit summary: "
        f"min={min(y_logit):.6f}, mean={sum(y_logit) / len(y_logit):.6f}, max={max(y_logit):.6f}"
    )

    metrics_out = {
        "Accuracy": metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
        "Loss": sum(loss_list) / len(loss_list),
        "Precision": metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=0),
        "Recall": metrics.recall_score(y_true=y_true, y_pred=y_pred, zero_division=0),
        "F-measure": metrics.f1_score(y_true=y_true, y_pred=y_pred, zero_division=0),
        "Precision-Recall AUC": metrics.average_precision_score(y_true=y_true, y_score=y_prob),
        "AUC": metrics.roc_auc_score(y_true=y_true, y_score=y_prob),
        "MCC": metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred),
        "Avg. Error (%)": sum(abs(prob - label) * 100 for prob, label in zip(y_prob, y_true)) / len(y_true),
    }

    if pairwise:
        stats = {"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0}
        total = 0
        for group in pair_groups.values():
            if len(group) != 2:
                continue
            sorted_group = sorted(group, key=lambda item: item[1], reverse=True)
            (p1, y1), (p2, y2) = sorted_group
            if y1 != 1 or y2 != 0:
                continue
            total += 1
            if p1 == 1 and p2 == 0:
                stats["P-C"] += 1
            elif p1 == 1 and p2 == 1:
                stats["P-V"] += 1
            elif p1 == 0 and p2 == 0:
                stats["P-B"] += 1
            elif p1 == 0 and p2 == 1:
                stats["P-R"] += 1

        for key in stats:
            stats[key] = stats[key] / total if total > 0 else 0.0
        metrics_out.update(stats)

    return metrics_out


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

    metrics = evaluate_logits_model(model, loader, threshold=args.threshold, pairwise=args.pairwise)

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
