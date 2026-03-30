import os
import sys
import glob

# Add that directory to sys.path if it's not already there
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
import subprocess
import numpy as np

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.utils import resample
import matplotlib.colors as mcolors
from devign.devign import Devign
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler, Subset

SEED = 42

DEVIGN = {
        "learning_rate" : 1e-3,
        "weight_decay" : 1e-6,
        "loss_lambda" : 1e-6,
        "model": {
            "gated_graph_conv_args": {"out_channels" : 200, "num_layers" : 6, "aggr" : "add", "bias": True},
            "conv_args": {
                "conv1d_1" : {"in_channels": 205, "out_channels": 50, "kernel_size": 3, "padding" : 1},
                "conv1d_2" : {"in_channels": 50, "out_channels": 20, "kernel_size": 1, "padding" : 1},
                "maxpool1d_1" : {"kernel_size" : 3, "stride" : 2},
                "maxpool1d_2" : {"kernel_size" : 2, "stride" : 2}
            },
            "emb_size" : 101
        }
    }

PROCESS = {
        "epochs" : 100,
        "patience" : 10,
        "batch_size" : 32,
        "dataset_ratio" : 0.2,
        "shuffle" : False
    }
    
COLORS = {
    '100_0': '#1f77b4',   # blue
    '90_10': '#3c8dbc',   # moderate blue
    '80_20': '#2ca02c',   # green
    '70_30': '#55a65a',   # medium green variant
    '60_40': '#d62728',   # red
    '50_50': '#ff7f0e',   # orange
    '40_60': '#9467bd',   # purple
    '30_70': '#ab82ff',   # lavender
    '20_80': '#8c564b',   # brown
    '10_90': '#c49c94',   # light brown / tan
    '0_100': '#17becf'    # cyan
}

def get_gpu_memory():
    """Returns a list of (total_MB, used_MB, free_MB) tuples for each GPU."""
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
         '--format=csv,nounits,noheader']
    ).decode('utf-8')
    
    memory = []
    for line in result.strip().split('\n'):
        total, used, free = map(int, line.split(','))
        memory.append({'total': total, 'used': used, 'free': free})
    return memory

def select_best_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    memory_info = get_gpu_memory()

    best_gpu = max(enumerate(memory_info), key=lambda x: x[1]['free'])
    device_id = best_gpu[0]
    free_mem = best_gpu[1]['free']

    print(f"Selected GPU {device_id} with {free_mem} MB free memory.")

    return device_id

DEVICE = torch.device(f"cuda:{select_best_gpu()}") if torch.cuda.is_available() else "cpu"

class DevignDataset(Dataset):
    def __init__(self, dataset_source, index_file=None, max_load_retries=3):
        self.max_load_retries = max_load_retries
        self.mode = "pt_dir" if isinstance(dataset_source, str) else "dataframe"

        if self.mode == "pt_dir":
            self.data_dir = dataset_source
            self.index_file = index_file or os.path.join(self.data_dir, "index.csv")
            self.samples = self._build_sample_index()
            self.targets = [int(item["target"]) for item in self.samples]
        else:
            # Backward-compatible path for existing code that still passes a DataFrame.
            self.dataset = dataset_source
            self.targets = [int(t) for t in self.dataset.target.values]

        self.set_sampler()

    def __len__(self):
        if self.mode == "pt_dir":
            return len(self.samples)
        return len(self.dataset)

    def __getitem__(self, index):
        if self.mode == "dataframe":
            row = self.dataset.iloc[index]
            sample = {
                "target": torch.tensor(row["target"], dtype=torch.long),
                "input": row["input"],
                "id": str(row["id"]),
            }
            if hasattr(sample["input"], "y") and sample["input"].y is None:
                sample["input"].y = torch.tensor([float(row["target"])], dtype=torch.float32)
            return sample

        # Lazy loading mode: only load a single file when requested by DataLoader.
        last_exc = None
        n = len(self.samples)
        for offset in range(min(self.max_load_retries, n)):
            current_index = (index + offset) % n
            meta = self.samples[current_index]
            file_path = meta["file_path"]

            try:
                # These files are produced by our own preprocessing script, so allow full unpickling.
                payload = torch.load(file_path, map_location="cpu", weights_only=False)

                if isinstance(payload, dict):
                    graph = payload.get("input")
                    target = int(payload.get("target", meta["target"]))
                    sample_id = str(payload.get("id", meta["id"]))
                else:
                    graph = payload
                    target = int(meta["target"])
                    sample_id = str(meta["id"])

                if graph is None:
                    raise ValueError("Missing 'input' graph in sample payload")

                # Ensure label exists on graph for current training loss logic.
                if not hasattr(graph, "y") or graph.y is None:
                    graph.y = torch.tensor([float(target)], dtype=torch.float32)

                return {
                    "input": graph,
                    "target": torch.tensor(target, dtype=torch.long),
                    "id": sample_id,
                }
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(f"Failed to load sample index {index} after retries: {last_exc}")

    def _build_sample_index(self):
        if os.path.exists(self.index_file):
            index_df = pd.read_csv(self.index_file)
            required_cols = {"filename", "target", "id"}
            if not required_cols.issubset(index_df.columns):
                raise ValueError(f"Index file missing columns {required_cols}: {self.index_file}")

            samples = []
            for row in index_df.itertuples(index=False):
                filename = str(row.filename)
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    samples.append({
                        "file_path": file_path,
                        "target": int(row.target),
                        "id": str(row.id),
                    })
            if samples:
                return samples

        pt_files = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in directory: {self.data_dir}")

        samples = []
        for file_path in pt_files:
            try:
                payload = torch.load(file_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict):
                    target = int(payload.get("target", 0))
                    sample_id = str(payload.get("id", os.path.basename(file_path)))
                else:
                    # If target/id are missing, fallback values keep dataset readable.
                    target = 0
                    sample_id = os.path.basename(file_path)
                samples.append({
                    "file_path": file_path,
                    "target": target,
                    "id": sample_id,
                })
            except Exception:
                # Skip malformed files during indexing; __getitem__ also has retry logic.
                continue

        if not samples:
            raise RuntimeError(f"Unable to index valid .pt samples in: {self.data_dir}")
        return samples

    def set_sampler(self):
        unique_targets = np.unique(self.targets)
        class_sample_count = np.array([len(np.where(np.array(self.targets) == t)[0]) for t in unique_targets])
        weight = 1. / class_sample_count

        class_to_weight = {int(t): weight[i] for i, t in enumerate(unique_targets)}
        samples_weight = np.array([class_to_weight[int(t)] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight).double()

        # Check if all indices exist
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

        self.sampler = sampler

    def get_loader(self, batch_size, shuffle=True, use_sampler=True):
        if use_sampler:
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=False, sampler=self.sampler)
        else:
            # When not using the sampler, use normal shuffling.
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

def group_train_val_test_split(df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=SEED):
    # Get unique ids (each id groups together an original and its adversarial counterpart)
    unique_ids = df['id'].unique()
    # Split ids into train_val and test groups.
    test_size = int(len(df)*test_size)
    train_val_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    # Create dataframes from these groups.
    train_val_df = df[df['id'].isin(train_val_ids)].reset_index(drop=True)
    test_df = df[df['id'].isin(test_ids)].reset_index(drop=True)
    
    # Further split the train_val_df into training and validation, again at group level.
    unique_train_val_ids = train_val_df['id'].unique()
    val_size = int(len(df)*val_size)
    train_ids, val_ids = train_test_split(unique_train_val_ids, test_size=val_size, random_state=random_state)
    
    train_df = train_val_df[train_val_df['id'].isin(train_ids)].reset_index(drop=True)
    val_df = train_val_df[train_val_df['id'].isin(val_ids)].reset_index(drop=True)
    
    return train_df, val_df, test_df

def binary_accuracy(probs, all_labels):
    """
    Calculates the accuracy for binary classification given sigmoid probabilities and true labels.

    Args:
    - probs (torch.Tensor): Model's output probabilities, shape (n_samples,).
    - all_labels (torch.Tensor): True labels, shape (n_samples,).

    Returns:
    - float: The accuracy as a percentage.
    """
    # Round each probability to get binary predictions (0 or 1)
    predicted_classes = torch.round(probs)
    
    # Calculate accuracy
    correct_predictions = (predicted_classes == all_labels).sum()
    accuracy = torch.div(correct_predictions, len(all_labels) + 0.0)
    
    return accuracy

def flatten_list(list_of_lists):
    flattened_list = []
    for _list in list_of_lists:
        for _item in _list:
            flattened_list.append(_item)
    return flattened_list

def compute_pairwise_metrics_from_loader(model, dataloader):
    """
    Compute pair-wise evaluation metrics from a DataLoader.

    Args:
        model (torch.nn.Module): The trained model for vulnerability detection.
        dataloader (torch.utils.data.DataLoader): The dataloader yielding batches.
        device (str): The device to run inference on.

    Returns:
        dict: Dictionary with P-C, P-V, P-B, P-R in percentage.
    """
    model.eval()
    pair_groups = {}  # pair_id -> list of (pred, true)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(DEVICE)
            labels = batch['target'].to(DEVICE)
            pair_ids = batch['id']
            
            # Forward pass
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs).view(-1) > 0.5).long()

            # Group predictions by pair_id
            for pid, pred, true in zip(pair_ids, preds.cpu(), labels.cpu()):
                try:
                    pair_groups[pid].append((pred.item(), true.item()))
                except KeyError:
                    pair_groups[pid] = [(pred.item(), true.item())]

    # Evaluate valid pairs (those with exactly 2 samples: one benign, one vulnerable)
    total = 0
    stats = {"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0}
    for pid, samples in pair_groups.items():
        if len(samples[0]) != 2:
            continue  # skip unpaired or malformed groups

        # Extract predictions and labels
        (p1, y1), (p2, y2) = samples

        # Sort so y1 == 1 means vulnerable
        if y1 == 0 and y2 == 1:
            (p1, y1), (p2, y2) = (p2, y2), (p1, y1)

        if y1 != 1 or y2 != 0:
            continue  # skip invalid pairs

        total += 1

        if p1 == 1 and p2 == 0:
            stats["P-C"] += 1
        elif p1 == 1 and p2 == 1:
            stats["P-V"] += 1
        elif p1 == 0 and p2 == 0:
            stats["P-B"] += 1
        elif p1 == 0 and p2 == 1:
            stats["P-R"] += 1

    # Normalize to percentages
    for k in stats:
        stats[k] = stats[k] / total if total > 0 else 0.0

    return stats

def eval_model(model, dataloader, threshold=0.5, test=False):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataset loader.
        threshold (float): Decision threshold for classification. Default is 0.5.
        info (bool): If True, prints detailed evaluation metrics.

    Returns:
        Tuple: (accuracy, loss)
    """
    loss_list = []
    labels = []
    predicts = []
    scores = []
    
    model.eval()

    # Hyperparameters
    loss_lambda = 1e-06

    # Loss Function
    loss_fc = lambda o, t: F.binary_cross_entropy_with_logits(o, t) + F.l1_loss(o, t) * loss_lambda
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input = batch["input"].to(DEVICE)
            logit = model(input)  # Get raw scores (logits)
            loss = loss_fc(logit, input.y)  # Compute loss
            
            labels.append(input.y.cpu().numpy())  # Store true labels
            scores.append(logit.cpu().numpy())  # Store raw scores
            
            # Apply threshold to determine binary predictions
            preds = (logit >= threshold).float().cpu().numpy()
            predicts.append(preds)  # Store thresholded predictions
            
            # Compute batch accuracy
            loss_list.append(loss.item())
    
    # Flatten lists into NumPy arrays
    labels = np.array(flatten_list(labels))
    predicts = np.array(flatten_list(predicts))
    scores = np.array(flatten_list(scores))

    # Compute final metrics
    acc = metrics.accuracy_score(y_true=labels, y_pred=predicts)
    loss = np.mean(loss_list)
    precision = metrics.precision_score(y_true=labels, y_pred=predicts, zero_division=0)
    recall = metrics.recall_score(y_true=labels, y_pred=predicts)

    # Compute confusion matrix
    confusion = confusion_matrix(y_true=labels, y_pred=predicts)
    tn, fp, fn, tp = confusion.ravel()
    
    print(f"\nConfusion matrix: \n{confusion}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    
    # Compute error percentage
    _errors = [abs(score - label) * 100 for score, label in zip(scores, labels)]
    error = sum(_errors) / len(_errors)

    # Compute evaluation metrics
    eval_metrics = {
        "Accuracy": acc,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
        "F-measure": metrics.f1_score(y_true=labels, y_pred=predicts),
        "Precision-Recall AUC": metrics.average_precision_score(y_true=labels, y_score=scores),
        "AUC": metrics.roc_auc_score(y_true=labels, y_score=scores),
        "MCC": metrics.matthews_corrcoef(y_true=labels, y_pred=predicts),
        "Avg. Error (%)": error
    }

    if test:
        # Pair-wise Accuracy
        eval_metrics.update(compute_pairwise_metrics_from_loader(model, dataloader))

    return eval_metrics

# Balance target within each subset
def balance_targets(df_group, target_col):
    class_0 = df_group[df_group[target_col] == 0]
    class_1 = df_group[df_group[target_col] == 1]
    if len(class_0) > len(class_1):
        class_1 = resample(class_1, replace=True, n_samples=len(class_0), random_state=SEED)
    elif len(class_1) > len(class_0):
        class_0 = resample(class_0, replace=True, n_samples=len(class_1), random_state=SEED)
    return pd.concat([class_0, class_1], ignore_index=True)

def create_balanced_symmetric_benchmark_split(df, orig_frac, size=10000, random_state=13):
    assert 0 <= orig_frac <= 1, "orig_frac must be between 0 and 1"
    ce_frac = round(1 - orig_frac,2)
    N = size

    # Step 0: Split into original and counterexample pools
    orig_df = df[df['adv'] == False]
    ce_df = df[df['adv'] == True]

    # Handle edge cases: fully original or fully counterexample
    if orig_frac == 1.0:
        source_df = orig_df.copy()
    elif orig_frac == 0.0:
        source_df = ce_df.copy()
    else:
        source_df = None  # to be handled below

    if source_df is not None:
        # Ensure target class balance
        benign_df = source_df[source_df['target'] == 0]
        vuln_df = source_df[source_df['target'] == 1]

        half_N = N // 2
        if len(benign_df) < half_N:
            benign_df = resample(benign_df, replace=True, n_samples=half_N, random_state=random_state)
        else:
            benign_df = benign_df.sample(n=half_N, random_state=random_state)

        if len(vuln_df) < half_N:
            vuln_df = resample(vuln_df, replace=True, n_samples=half_N, random_state=random_state + 1)
        else:
            vuln_df = vuln_df.sample(n=half_N, random_state=random_state + 1)

        if N % 2 == 1:
            extra_sample = pd.concat([benign_df, vuln_df]).sample(1, random_state=random_state + 2)
            final_df = pd.concat([benign_df, vuln_df, extra_sample], ignore_index=True)
        else:
            final_df = pd.concat([benign_df, vuln_df], ignore_index=True)

        return final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Non-extreme case: mix of original and counterexample
    M = int(min(orig_frac, ce_frac) * N)

    # Step 1: Select M examples from the minority pool
    if orig_frac > ce_frac:
        pool = ce_df.sort_values(by='target', ascending=False)  # prefer vulnerable
    else:
        pool = orig_df.sort_values(by='target', ascending=True)  # prefer benign

    selected_ids_step1 = pool['id'].drop_duplicates().head(M).tolist()
    step1_df = pool[pool['id'].isin(selected_ids_step1)]

    # Step 2: Add M counterparts from the opposite set
    if orig_frac > ce_frac:
        step2_df = orig_df[orig_df['id'].isin(selected_ids_step1)]
        majority_df = orig_df
    else:
        step2_df = ce_df[ce_df['id'].isin(selected_ids_step1)]
        majority_df = ce_df

    # Step 3: Add (N - 2M) more examples from the majority set (not used before)
    used_ids = set(step1_df['id'].unique()).union(set(step2_df['id'].unique()))
    candidate_df = majority_df[~majority_df['id'].isin(used_ids)]

    benign_df = candidate_df[candidate_df['target'] == 0]
    vuln_df = candidate_df[candidate_df['target'] == 1]

    n_remaining = N - 2 * M
    n_half = n_remaining // 2

    if len(benign_df) < n_half:
        benign_extra = resample(benign_df, replace=True, n_samples=n_half, random_state=random_state + 1)
    else:
        benign_extra = benign_df.sample(n=n_half, random_state=random_state + 1)

    if len(vuln_df) < n_half:
        vuln_extra = resample(vuln_df, replace=True, n_samples=n_half, random_state=random_state)
    else:
        vuln_extra = vuln_df.sample(n=n_half, random_state=random_state)

    if n_remaining % 2 == 1:
        extra_sample = pd.concat([benign_df, vuln_df]).sample(1, random_state=random_state + 2)
        step3_df = pd.concat([benign_extra, vuln_extra, extra_sample], ignore_index=True)
    else:
        step3_df = pd.concat([benign_extra, vuln_extra], ignore_index=True)

    # Combine all and shuffle
    full_df = pd.concat([step1_df, step2_df, step3_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return full_df

def plot_benchmark_distribution_with_duplicates_stacked(benchmark_datasets):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    bench_keys = sorted(benchmark_datasets.keys(), key=lambda x: int(x.split('_')[0]), reverse=True)
    n_bench = len(bench_keys)

    # Initialize data containers
    t0u_orig, t0u_counter = [], []
    t1u_orig, t1u_counter = [], []
    t0dup_orig, t0dup_counter = [], []
    t1dup_orig, t1dup_counter = [], []

    for key in bench_keys:
        df = benchmark_datasets[key]['train']

        # Target 0
        t0_df = df[df['target'] == 0]
        t0_orig_full = t0_df[t0_df['adv'] == False]
        t0_counter_full = t0_df[t0_df['adv'] == True]
        t0u_orig.append(t0_orig_full.drop_duplicates('id').shape[0])
        t0u_counter.append(t0_counter_full.drop_duplicates('id').shape[0])
        t0dup_orig.append(t0_orig_full.shape[0] - t0u_orig[-1])
        t0dup_counter.append(t0_counter_full.shape[0] - t0u_counter[-1])

        # Target 1
        t1_df = df[df['target'] == 1]
        t1_orig_full = t1_df[t1_df['adv'] == False]
        t1_counter_full = t1_df[t1_df['adv'] == True]
        t1u_orig.append(t1_orig_full.drop_duplicates('id').shape[0])
        t1u_counter.append(t1_counter_full.drop_duplicates('id').shape[0])
        t1dup_orig.append(t1_orig_full.shape[0] - t1u_orig[-1])
        t1dup_counter.append(t1_counter_full.shape[0] - t1u_counter[-1])

    x = np.arange(n_bench)
    bar_width = 0.35
    x_t0 = x - bar_width / 2
    x_t1 = x + bar_width / 2

    fig, ax = plt.subplots(figsize=(16, 8))

    # Target 0 bars (original and counterexample, unique and duplicated)
    b1 = ax.bar(x_t0, t0u_orig, bar_width, label='Benign Original', color='#4daf4a', edgecolor='black')
    b2 = ax.bar(x_t0, t0dup_orig, bar_width, bottom=t0u_orig, label='Benign Upsampled Original', color='#74c476', edgecolor='black')
    b3 = ax.bar(x_t0, t0u_counter, bar_width, bottom=np.array(t0u_orig)+np.array(t0dup_orig), label='Benign Counterexample', color='#a6d96a', edgecolor='black')
    b4 = ax.bar(x_t0, t0dup_counter, bar_width, bottom=np.array(t0u_orig)+np.array(t0dup_orig)+np.array(t0u_counter), label='Benign Upsampled Counterexample', color='#c2e699', edgecolor='black')

    # Target 1 bars (original and counterexample, unique and duplicated)
    b5 = ax.bar(x_t1, t1u_orig, bar_width, label='Vulnerable Original', color='#e41a1c', edgecolor='black')
    b6 = ax.bar(x_t1, t1dup_orig, bar_width, bottom=t1u_orig, label='Vulnerable Upsampled Original', color='#fb6a4a', edgecolor='black')
    b7 = ax.bar(x_t1, t1u_counter, bar_width, bottom=np.array(t1u_orig)+np.array(t1dup_orig), label='Vulnerable Counterexample', color='#fc8d59', edgecolor='black')
    b8 = ax.bar(x_t1, t1dup_counter, bar_width, bottom=np.array(t1u_orig)+np.array(t1dup_orig)+np.array(t1u_counter), label='Vulnerable Upsampled Counterexample', color='#fcbba1', edgecolor='black')

    # Annotate bars
    for i in range(n_bench):
        tot_t0 = t0u_orig[i] + t0dup_orig[i] + t0u_counter[i] + t0dup_counter[i]
        tot_t1 = t1u_orig[i] + t1dup_orig[i] + t1u_counter[i] + t1dup_counter[i]
        ax.text(x_t0[i], tot_t0 + 0.01 * max(t0u_orig + t0dup_orig + t0u_counter + t0dup_counter), f"{tot_t0}", ha='center', va='bottom', fontsize=10)
        ax.text(x_t1[i], tot_t1 + 0.01 * max(t1u_orig + t1dup_orig + t1u_counter + t1dup_counter), f"{tot_t1}", ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Number of Examples')
    ax.set_title('Benchmark Dataset Distribution\n(Target Class vs. Original/Counterexample, with Upsampled Examples)')
    ax.set_xticks(x)
    ax.set_xticklabels(bench_keys, fontsize=12)
    ax.legend(loc='center right', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    ax.set_ylim(top=6500) 
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(f'benchmarks/dataset_distribution.png')
    plt.show()

def summarize_benchmark_distribution_compact(benchmark_datasets):
    bench_keys = sorted(benchmark_datasets.keys(), key=lambda x: int(x.split('_')[0]), reverse=True)
    data = []

    for key in bench_keys:
        df = benchmark_datasets[key]['train']

        row = {'Benchmark': key}

        for target, tlabel in zip([0, 1], ['Benign', 'Vulnerable']):
            for adv, adv_label in zip([False, True], ['Original', 'Counterexample']):
                subset = df[(df['target'] == target) & (df['adv'] == adv)]
                unique_ids = subset['id'].drop_duplicates().shape[0]
                total = subset.shape[0]
                duplicated = total - unique_ids

                col_prefix = f"{tlabel}_{adv_label}"
                row[f"{col_prefix}_Unique"] = unique_ids
                row[f"{col_prefix}_Upsampled"] = duplicated
                row[f"{col_prefix}_Total"] = total

        data.append(row)

    compact_df = pd.DataFrame(data)

    return compact_df

def adjust_color(color, factor):
    """
    Adjusts the brightness of a given color.
    factor > 1 will lighten the color.
    factor < 1 will darken the color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    rgb = np.clip(rgb * factor, 0, 1)
    return mcolors.to_hex(rgb)


def split_indices_by_id(index_df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=SEED):
    """
    Split rows by group id so all samples with the same id stay in one split.
    """
    if "id" not in index_df.columns:
        raise ValueError("index.csv must contain an 'id' column")

    id_series = index_df["id"].astype(str)
    unique_ids = id_series.unique()
    if len(unique_ids) < 3:
        raise ValueError("Need at least 3 unique ids for train/val/test split")

    train_val_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

    # Keep val ratio relative to the remaining train_val pool.
    val_ratio_in_train_val = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio_in_train_val, random_state=random_state)

    train_indices = index_df[id_series.isin(train_ids)].index.tolist()
    val_indices = index_df[id_series.isin(val_ids)].index.tolist()
    test_indices = index_df[id_series.isin(test_ids)].index.tolist()

    return train_indices, val_indices, test_indices


def run_lazy_training(processed_dir: str, index_csv: str):
    """
    Train Devign directly from split .pt files using lazy loading.
    """
    os.makedirs("benchmarks", exist_ok=True)

    index_df = pd.read_csv(index_csv)
    required_cols = {"filename", "target", "id"}
    if not required_cols.issubset(index_df.columns):
        raise ValueError(f"{index_csv} missing required columns: {required_cols}")

    print(f"Loaded index file: {index_csv}")
    print(f"Total samples: {len(index_df)}")

    train_indices, val_indices, test_indices = split_indices_by_id(
        index_df,
        test_size=0.1,
        val_size=0.1,
        random_state=SEED,
    )

    print(f"Split sizes -> train: {len(train_indices)}, valid: {len(val_indices)}, test: {len(test_indices)}")

    full_dataset = DevignDataset(processed_dir, index_file=index_csv)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PROCESS["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PROCESS["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=PROCESS["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    model = Devign(
        gated_graph_conv_args={
            'out_channels': 200,
            'num_layers': 6,
            'aggr': 'add',
            'bias': True
        },
        conv_args={
            'conv1d_1': {
                'in_channels': 205,
                'out_channels': 50,
                'kernel_size': 3,
                'padding': 1
            },
            'conv1d_2': {
                'in_channels': 50,
                'out_channels': 20,
                'kernel_size': 1,
                'padding': 1
            },
            'maxpool1d_1': {
                'kernel_size': 3,
                'stride': 2
            },
            'maxpool1d_2': {
                'kernel_size': 2,
                'stride': 2
            }
        },
        emb_size=101
    )

    learning_rate = 5e-4
    weight_decay = 1e-05
    loss_lambda = 1e-06
    loss_fc = lambda o, t: F.binary_cross_entropy_with_logits(o, t) + F.l1_loss(o, t) * loss_lambda
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(DEVICE)

    early_stop_count = 0
    early_stop = int(PROCESS['epochs'] / 2)
    best_loss = float('inf')
    best_acc = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    THRESHOLD = 0.5
    model_save_path = "benchmarks/devign_Accuracy_lazy.pt"

    for epoch in range(PROCESS['epochs']):
        train_acc_list = []
        train_loss_list = []

        model.train()
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            input_graph = batch["input"].to(DEVICE)
            logit = model(input_graph)
            loss = loss_fc(logit, input_graph.y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            preds = (logit >= THRESHOLD).float().cpu().numpy()
            acc = binary_accuracy(torch.tensor(preds), input_graph.y.cpu())

            train_acc_list.append(acc)
            train_loss_list.append(loss.item())

        train_acc = np.mean(train_acc_list)
        train_loss = np.mean(train_loss_list)
        val_metrics = eval_model(model, val_loader, threshold=THRESHOLD)

        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, val_acc={val_metrics['Accuracy']:.4f}, val_loss={val_metrics['Loss']:.4f}")

        scheduler.step(val_metrics["Accuracy"])

        is_best = (val_metrics["Accuracy"] > best_acc) or (
            val_metrics["Loss"] < best_loss and val_metrics["Accuracy"] >= best_acc
        )
        if is_best:
            best_acc = val_metrics["Accuracy"]
            best_loss = val_metrics["Loss"]
            early_stop_count = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model: {model_save_path}")
        else:
            early_stop_count += 1

        if early_stop_count > early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()
    test_metrics = eval_model(model, test_loader, test=True)
    print(
        f"Test: Acc={test_metrics['Accuracy']:.4f}, Loss={test_metrics['Loss']:.4f}, "
        f"Precision={test_metrics['Precision']:.4f}, Recall={test_metrics['Recall']:.4f}"
    )

    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv("benchmarks/lazy_metrics.csv", index=False)
    print("Saved test metrics to benchmarks/lazy_metrics.csv")

if __name__ == "__main__":

    processed_dir = os.getenv("PROCESSED_DATA_DIR", "processed_data")
    index_csv = os.path.join(processed_dir, "index.csv")

    if os.path.exists(index_csv):
        print("Lazy-loading mode enabled (using processed_data/*.pt)")
        run_lazy_training(processed_dir, index_csv)
        sys.exit(0)

    # Load processed datasets
    print("Data Loading")
    print("-----------------------------------------")
    print("Loading...")
    # TRAIN
    dataset_df = pd.read_pickle('datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_balanced.pkl')
    print(f"Dataset loaded. {len(dataset_df)} examples")
    print(dataset_df['target'].value_counts())

    # Step 1: Create the base train/val/test split — all balanced
    train_df, val_df, test_df = group_train_val_test_split(dataset_df, test_size=0.1, val_size=0.1, random_state=SEED)

    # Step 2: Fix the test set (same across all benchmarks)
    fixed_test_df = test_df.copy()  # already balanced
    test_dataset = DevignDataset(fixed_test_df)
    test_loader = test_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=False)

    # Define benchmark splits: (orig, adv)
    benchmark_datasets = {}
    benchmark_splits = [(100-i, i) for i in range(0, 110, 10)]

    for orig_pct, adv_pct in benchmark_splits:
        key = f'{orig_pct}_{adv_pct}'

        benchmark_train_df = create_balanced_symmetric_benchmark_split(
            train_df,
            orig_frac=orig_pct/100,
            size=len(train_df),
            random_state=SEED
        )

        benchmark_valid_df = create_balanced_symmetric_benchmark_split(
            val_df,
            orig_frac=orig_pct/100,
            size=len(val_df),
            random_state=SEED
        )

        # Store both benchmark and fixed test set
        benchmark_datasets[key] = {
            'train': benchmark_train_df,
            'valid': benchmark_valid_df
        }

    # Benchmark dataset distribution
    plot_benchmark_distribution_with_duplicates_stacked(benchmark_datasets)
    benchmark_distribution_df = summarize_benchmark_distribution_compact(benchmark_datasets)
    benchmark_distribution_df.to_csv("benchmarks/dataset_distribution.csv")

    # Lists to store training, validation and test accuracy for plotting
    train_acc_history = {}
    val_acc_history = {}
    val_target_history = {}
    epochs_list = {}
    last_saved_epoch_history = {}
    test_acc_history = {}
    test_target_history = {}

    for key, benchmark_df in benchmark_datasets.items():
        # For each benchmark obtain train and validation splits:
        train_dataset = DevignDataset(benchmark_df['train'])
        val_dataset = DevignDataset(benchmark_df['valid'])

        # Data Loaders
        train_loader = train_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=False)
        val_loader = val_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=False)

        # Model
        model = Devign(
        gated_graph_conv_args={
                'out_channels': 200,
                'num_layers': 6,
                'aggr': 'add',
                'bias': True
                },
        conv_args={
                'conv1d_1': {
                    'in_channels': 205, 
                    'out_channels': 50, 
                    'kernel_size': 3,
                    'padding': 1
                    }, 
                'conv1d_2': {
                    'in_channels': 50, 
                    'out_channels': 20,
                    'kernel_size': 1, 
                    'padding': 1
                    }, 
                'maxpool1d_1': {
                    'kernel_size': 3, 
                    'stride': 2
                    }, 
                'maxpool1d_2': {
                    'kernel_size': 2, 
                    'stride': 2
                    }
                },
        emb_size=101
        )

        # Hyperparameters
        learning_rate = 5e-4
        weight_decay = 1e-05
        loss_lambda = 1e-06

        # Loss Function
        loss_fc = lambda o, t: F.binary_cross_entropy_with_logits(o, t) + F.l1_loss(o, t) * loss_lambda

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Move model to the correct device
        model.to(DEVICE)

        # Early stopping parameters
        early_stop_count = 0
        early_stop = int(PROCESS['epochs']/2)  # Number of epochs without improvement
        best_loss = float('inf')

        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        # Decision threshold for classification
        THRESHOLD = 0.5

        # eval target metric
        TARGET_METRIC = {}
        TARGET_METRIC['name'] = "Accuracy"
        TARGET_METRIC["best"] = 0

        model_save_path = f"benchmarks/devign_{TARGET_METRIC['name']}_{key}.pt"

        # Lists to store training and validation metrics for plotting
        train_acc_history[key] = []
        val_acc_history[key] = []
        val_target_history[key] = []
        epochs_list[key] = []
        last_saved_epoch_history[key] = None
        test_acc_history[key] = None
        test_target_history[key] = None

        # Variables to record the last saved epoch and its metrics.
        last_saved_epoch = None
        last_saved_val_acc = None
        last_saved_val_target = None

        for epoch in range(PROCESS['epochs']):
            train_acc_list = []
            train_loss_list = []

            # Set model to training mode
            model.train()

            for i, batch in enumerate(tqdm(train_loader)):
                input = batch["input"].to(DEVICE)  # Move batch to GPU
                logit = model(input)  # Forward pass
                
                # Compute loss
                loss = loss_fc(logit, input.y)

                # Reset gradients
                optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                # Perform parameter update
                optimizer.step()

                # Apply threshold to determine binary predictions
                preds = (logit >= THRESHOLD).float().cpu().numpy()
                acc = binary_accuracy(torch.tensor(preds), input.y.cpu())

                train_acc_list.append(acc)
                train_loss_list.append(loss.item())

            # Compute train metrics
            train_acc = np.mean(train_acc_list)
            train_loss = np.mean(train_loss_list)

            # Evaluate on validation set
            val_metrics = eval_model(model, val_loader, threshold=THRESHOLD)

            print(f"Epoch {epoch}:")
            print(f" - Train: Acc = {train_acc:.4f} | Loss = {train_loss:.4f}")
            print(f" - Valid: Acc = {val_metrics['Accuracy']:.4f} | Loss = {val_metrics['Loss']:.4f} | Precision = {val_metrics['Precision']:.4f} | Recall = {val_metrics['Recall']:.4f}")

            # Reduce learning rate if validation accuracy plateaus
            scheduler.step(val_metrics["Accuracy"])

            # Save accuracies for plotting
            train_acc_history[key].append(train_acc)
            val_acc_history[key].append(val_metrics["Accuracy"])
            val_target_history[key].append(val_metrics[TARGET_METRIC['name']])
            epochs_list[key].append(epoch + 1)

            # Early Stopping
            is_best = (val_metrics[TARGET_METRIC['name']] > TARGET_METRIC["best"]) or (val_metrics["Loss"] < best_loss and val_metrics[TARGET_METRIC['name']] >= TARGET_METRIC["best"])
            
            if is_best:
                TARGET_METRIC["best"] = val_metrics[TARGET_METRIC['name']]
                best_loss = val_metrics["Loss"]
                early_stop_count = 0  # Reset early stopping counter
                last_saved_epoch = epoch + 1
                last_saved_val_acc = val_metrics["Accuracy"]
                last_saved_val_target = val_metrics[TARGET_METRIC['name']]
                print(f"- Saved model (based on {TARGET_METRIC['name']}) at '{model_save_path}'.")
                torch.save(model.state_dict(), model_save_path)
            else:
                early_stop_count += 1

            if early_stop_count > early_stop:
                print("Early stopping triggered.")
                break
        last_saved_epoch_history[key] = last_saved_epoch

        # Benchmark plot after training
        plt.figure(figsize=(32, 24))
        plt.plot(epochs_list[key], train_acc_history[key], label="Train Accuracy", marker="o")
        plt.plot(epochs_list[key], val_acc_history[key], label="Validation Accuracy", marker="s")
        plt.plot(epochs_list[key], val_target_history[key], label=f"Validation {TARGET_METRIC['name']}", marker="d")
        plt.xlabel("Epochs")
        plt.ylabel("Performance Metrics")
        plt.title(f"Final Training vs. Validation Metrics (Key: {key})")
        plt.legend()
        plt.grid(True)

        # Annotate the point where the model was last saved
        if last_saved_epoch is not None:
            plt.annotate(f"Last Saved\nEpoch: {last_saved_epoch}\nVal Acc: {last_saved_val_acc:.4f}\nVal Recall: {last_saved_val_target:.4f}",
                        xy=(last_saved_epoch, last_saved_val_acc),
                        xytext=(last_saved_epoch, last_saved_val_acc + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=16,    
                        horizontalalignment='center')
            
        plt.savefig(f'benchmarks/training_{key}.png')
        plt.show()

        # TEST
        #-----------------------
        model.load(model_save_path)
        model.eval()
        test_metrics = eval_model(model, test_loader, test=True)
        test_acc_history[key] = test_metrics["Accuracy"]
        test_target_history[key] = test_metrics[TARGET_METRIC['name']]
        print(f" - Test: Acc = {test_metrics['Accuracy']:.4f} | Loss = {test_metrics['Loss']:.4f} | Precision = {test_metrics['Precision']:.4f} | Recall = {test_metrics['Recall']:.4f} | {TARGET_METRIC['name']} = {val_metrics[TARGET_METRIC['name']]:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame()
    metrics_df['benchmark'] = list(benchmark_datasets.keys())
    metrics_df['train_epochs'] = list(epochs_list.values())
    metrics_df['train_acc'] = list(train_acc_history.values())
    metrics_df['last_saved_epoch'] = list(last_saved_epoch_history.values())
    metrics_df['val_acc'] = list(val_acc_history.values())
    metrics_df['val_target'] = list(val_target_history.values())
    metrics_df['test_acc'] = list(test_acc_history.values())
    metrics_df['test_target'] = list(test_target_history.values())
    metrics_df.to_csv('benchmarks/metrics.csv', index=False)

    # Final plot training/eval for all benchmark
    plt.figure(figsize=(32, 24))
    for key in benchmark_datasets:
        base_color = COLORS.get(key, "#000000")
        # Create slight variants for each metric:
        train_color = adjust_color(base_color, 1.2)  # slightly lighter
        val_color   = base_color                    # base color
        target_color = adjust_color(base_color, 0.8)  # slightly darker

        plt.plot(epochs_list[key], train_acc_history[key],
            label=f"Train Accuracy {key}",
            marker="o", color=train_color, linestyle='-')
    
        plt.plot(epochs_list[key], val_acc_history[key],
            label=f"Validation Accuracy {key}",
            marker="s", color=val_color, linestyle='--')
        
        plt.plot(epochs_list[key], val_target_history[key],
            label=f"Validation Recall {key}",
            marker="d", color=target_color, linestyle=':')
        
    plt.xlabel("Epochs")
    plt.ylabel("Performance Metrics")
    plt.title(f"Final Training vs. Validation Metrics Across Benchmarks")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'benchmarks/training_eval_all.png')
    plt.show()

    # Final plot test for all benchmark
    plt.figure(figsize=(32, 24))
    for key in benchmark_datasets:
        # Get the base color for this benchmark from COLORS dictionary.
        base_color = COLORS.get(key, "#000000")
        
        # Plot a scatter point for this benchmark.
        plt.scatter(test_acc_history[key], test_target_history[key],
                    color=base_color, s=200, alpha=0.8, edgecolor='k', label=key)
        # Annotate the point with the benchmark key.
        plt.text(int(test_acc_history[key])*1.01, int(test_target_history[key])*1.01, key, fontsize=12)

    plt.xlabel("Test Accuracy")
    plt.ylabel(f"Test {TARGET_METRIC['name']}")
    plt.title("Comparative Test Metrics Across Benchmarks")
    plt.legend(title="Benchmark")
    plt.grid(True)
    plt.savefig('benchmarks/test_all.png')
    plt.show()