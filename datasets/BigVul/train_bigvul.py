import os
import sys

# Add that directory to sys.path if it's not already there
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.colors as mcolors
from devign.src.devign import Devign
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

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

# DEVICE = torch.device(f"cuda:{select_best_gpu()}") if torch.cuda.is_available() else "cpu"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DevignDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.set_sampler()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]

        # Convert row into a dictionary (or a tensor if needed)
        sample = {
            "func": row["func"],
            "target": torch.tensor(row["target"], dtype=torch.long),
            "input": row["input"],
            "cpg": row["cpg"]
        }

        return sample

    def set_sampler(self):
        # Ensure labels are 0-based integers
        targets = self.dataset.target.values.tolist()
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts

        # Assign a weight to each sample
        sample_weights = class_weights[targets]
        sample_weights = torch.from_numpy(sample_weights).double()

        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def get_loader(self, batch_size, shuffle=True, use_sampler=True):
        if use_sampler:
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=False, sampler=self.sampler)
        else:
            return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)
        
def group_train_val_test_split(df: pd.DataFrame, test_size=0.1, val_size=0.1, random_state=SEED):
    # Split ids into train_val and test groups.
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
    # Further split the train_val_df into training and validation, again at group level.
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
    
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

def eval_model(model, dataloader, threshold=0.5, info=False):
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

    eval_metrics = {
        "Accuracy": acc,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
    }

    if info:
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

        # Print metrics
        # print("\n".join([f"{metric}: {eval_metrics[metric]:.4f}" for metric in eval_metrics]))

    return eval_metrics

def balance_df_by_target(df, target_col='target', random_state=SEED):
    groups = df.groupby(target_col)
    min_count = groups.size().min()
    balanced = groups.sample(n=min_count, random_state=random_state)
    # When using groupby.apply, the index might be a MultiIndex – reset it:
    return balanced.reset_index(drop=True)

def oversample_df_by_target(df, target_col='target', random_state=42):
    groups = df.groupby(target_col)
    max_count = groups.size().max()
    oversampled_list = []
    for label, group in groups:
        # For groups smaller than the maximum, sample with replacement until size equals max_count.
        oversampled = group.sample(n=max_count, replace=True, random_state=random_state)
        oversampled_list.append(oversampled)
    balanced_df = pd.concat(oversampled_list).reset_index(drop=True)
    return balanced_df

def create_paired_benchmark_from_split(split_df: pd.DataFrame, orig_frac: float, adv_frac: float, target_col='target', random_state=SEED):
    """
    Creates a benchmark from the given split (train or validation). It ensures that for the lower group (original or counterexample),
    the samples are taken as pairs (original with its counterpart), and the majority is filled with extra samples.

    For example, if orig_frac < adv_frac, then:
       - Sample orig_frac proportion of original examples.
       - Gather their corresponding counterexample samples (paired, same ids).
       - Then, sample additional counterexample examples to reach the adv_frac proportion.
    """
    # Separate out the original and counterexample samples.
    orig_df = split_df[split_df.adv == False]
    adv_df = split_df[split_df.adv == True]

    n_total = min(len(orig_df), len(adv_df))

    # Determine the pairing based on the lower fraction:
    if orig_frac < adv_frac:
        # Step 1: Choose a fraction of the original samples.
        paired_orig = orig_df.sample(frac=orig_frac, random_state=random_state)
        # Step 2: Select the corresponding counterexample samples (using the same id).
        paired_adv = adv_df[adv_df['id'].isin(paired_orig['id'])]
        
        # Step 3: For the extra counterexample proportion, compute how many additional adv samples are needed.
        desired_adv_count = int(adv_frac * n_total)
        extra_adv_needed = desired_adv_count - len(paired_adv)
        
        # Sample extra counterexample examples from those not already paired
        remaining_adv = adv_df[~adv_df['id'].isin(paired_orig['id'])]
        extra_adv = remaining_adv.sample(n=extra_adv_needed, random_state=random_state) if extra_adv_needed > 0 else pd.DataFrame()
        
        # Combine the paired set with the extra counterexample examples.
        benchmark_df = pd.concat([paired_orig, paired_adv, extra_adv])
    
    elif adv_frac < orig_frac:
        # In the reverse case, sample a fraction of counterexample samples to pair.
        paired_adv = adv_df.sample(frac=adv_frac, random_state=random_state)
        paired_orig = orig_df[orig_df['id'].isin(paired_adv['id'])]
        
        desired_orig_count = int(orig_frac * n_total)
        extra_orig_needed = desired_orig_count - len(paired_orig)
        
        remaining_orig = orig_df[~orig_df['id'].isin(paired_adv['id'])]
        extra_orig = remaining_orig.sample(n=extra_orig_needed, random_state=random_state) if extra_orig_needed > 0 else pd.DataFrame()
        
        benchmark_df = pd.concat([paired_adv, paired_orig, extra_orig])
    
    else:
        # If equal, simply pair them.
        paired_orig = orig_df.sample(frac=orig_frac, random_state=random_state)
        paired_adv = adv_df[adv_df['id'].isin(paired_orig['id'])]
        benchmark_df = pd.concat([paired_orig, paired_adv])
    
    # Balance the benchmark over the target if needed.
    benchmark_df = oversample_df_by_target(benchmark_df, target_col=target_col, random_state=random_state)
    
    return benchmark_df

def create_balanced_test_set(test_df: pd.DataFrame, target_col='target', random_state=SEED):
    # A simple approach may be to take all groups (by id) and then balance by target.
    balanced_test = oversample_df_by_target(test_df, target_col=target_col, random_state=random_state)
    
    return balanced_test

def train_val_fixed_test_split(benchmark_df: pd.DataFrame, fixed_test_df: pd.DataFrame, shuffle=True):
    """
    Split benchmark_df into train and validation splits so that the fixed_test_df remains
    the test split for all benchmarks. Assumes that pairing is based on the 'id' column.

    Args:
      benchmark_df (pd.DataFrame): The benchmark-specific dataframe.
      fixed_test_df (pd.DataFrame): The globally fixed test set dataframe.
      shuffle (bool): Whether to shuffle in the internal splits (default True).

    Returns:
      A tuple of (train_dataset, val_dataset, test_dataset) where each is a DevignDataset.
    """
    # Remove fixed test samples from the benchmark data (by matching 'id')
    train_val_df = benchmark_df[~benchmark_df['id'].isin(fixed_test_df['id'])].reset_index(drop=True)
    
    # Split the remaining data into train and validation splits.
    false = train_val_df[train_val_df.target == 0]
    true  = train_val_df[train_val_df.target == 1]

    train_false, val_false = train_test_split(false, test_size=0.5, shuffle=shuffle, random_state=SEED)
    train_true,  val_true  = train_test_split(true, test_size=0.5, shuffle=shuffle, random_state=SEED)

    train_df = pd.concat([train_false, train_true]).reset_index(drop=True)
    val_df   = pd.concat([val_false, val_true]).reset_index(drop=True)

    return DevignDataset(train_df), DevignDataset(val_df), DevignDataset(fixed_test_df)

def print_dataset_distribution(df, name):
    """ Prints DataFrame distribution."""

    print(f"Dataset Distribution for {name}: \n ----------------------------")
    vuln, benign = df['target'].value_counts()
    print(f" - Vulnerable: {vuln} | Benign: {benign} | Total: {vuln+benign} ({100*vuln/(vuln+benign):.1f}%/{100*benign/(vuln+benign):.1f}%)")

def adjust_color(color, factor):
    """
    Adjusts the brightness of a given color.
    factor > 1 will lighten the color.
    factor < 1 will darken the color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    rgb = np.clip(rgb * factor, 0, 1)
    return mcolors.to_hex(rgb)

def check_balance(loader):
    label_counts = torch.zeros(2)
    total = 0
    for batch in loader:
        targets = batch['target']
        for label in targets:
            label_counts[label.item()] += 1
            total += 1
    print("Sampled class distribution (raw):", label_counts)
    print("Proportions:", label_counts / total)

if __name__ == "__main__":

    # Load processed datasets
    print("Data Loading")
    print("-----------------------------------------")
    print("Loading...")

    dataset_df = pd.read_pickle('datasets/BigVul/bigvul_CWE-20_input.pkl')
    print(f"Dataset loaded. {len(dataset_df)} examples")
    print(dataset_df['target'].value_counts())

    # Divide the dataset into train, val & test
    train_df, val_df, test_df = group_train_val_test_split(dataset_df, test_size=0.1, val_size=0.1)

    # For each benchmark obtain train and validation splits:
    # train_dataset, val_dataset, test_dataset = train_val_fixed_test_split(benchmark_df, fixed_test_df, shuffle=PROCESS["shuffle"])
    train_dataset = DevignDataset(train_df)
    val_dataset = DevignDataset(val_df)
    test_dataset = DevignDataset(test_df)

    # For training, use the weighted sampler.
    train_loader = train_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=True)
    # For validation and test, do not use the sampler—this will preserve the true distribution.
    val_loader = val_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=True)
    test_loader = test_dataset.get_loader(PROCESS["batch_size"], shuffle=False, use_sampler=False)

    check_balance(train_loader)
    check_balance(val_loader)
    check_balance(test_loader)

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
    TARGET_METRIC["name"] = "Accuracy"
    TARGET_METRIC["best"] = 0

    model_save_path = f"datasets/BigVul/train/devign_bigvul_{TARGET_METRIC['name']}.pt"

    # Variables to record the last saved epoch and its metrics.
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_recall_history = []
    epochs_list = []
    last_saved_epoch = None
    last_saved_val_acc = None
    last_saved_val_recall = None

    for epoch in range(PROCESS['epochs']):
        print(f"\n\nEpoch {epoch}:")
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
        print(f" - Train: Acc = {train_acc:.4f} | Loss = {train_loss:.4f}")

        # Evaluate on validation set
        val_metrics = eval_model(model, val_loader, threshold=THRESHOLD, info=True)
        print(f" - Valid: Acc = {val_metrics['Accuracy']:.4f} | Loss = {val_metrics['Loss']:.4f} | Precision = {val_metrics['Precision']:.4f} | Recall = {val_metrics['Recall']:.4f}")

        # Reduce learning rate if validation accuracy plateaus
        scheduler.step(val_metrics["Accuracy"])

        # Save accuracies for plotting
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_metrics["Accuracy"])
        val_recall_history.append(val_metrics["Recall"])
        epochs_list.append(epoch + 1)

        # Early Stopping
        is_best = (val_metrics[TARGET_METRIC["name"]] > TARGET_METRIC["best"]) or (val_metrics["Loss"] < best_loss and val_metrics[TARGET_METRIC["name"]] >= TARGET_METRIC["best"])
        
        if is_best:
            TARGET_METRIC["best"] = val_metrics[TARGET_METRIC["name"]]
            best_loss = val_metrics["Loss"]
            early_stop_count = 0  # Reset early stopping counter
            last_saved_epoch = epoch + 1
            last_saved_val_acc = val_metrics["Accuracy"]
            last_saved_val_recall = val_metrics["Recall"]
            print(f"- Saved model (based on {TARGET_METRIC['name']}) at '{model_save_path}'.")
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stop_count += 1

        if early_stop_count > early_stop:
            print("Early stopping triggered.")
            break
        last_saved_epoch_history = last_saved_epoch

        # Benchmark plot after training
        plt.figure(figsize=(32, 24))
        plt.plot(epochs_list, train_acc_history, label="Train Accuracy", marker="o")
        plt.plot(epochs_list, train_loss_history, label="Train Loss", marker=".")
        plt.plot(epochs_list, val_acc_history, label="Validation Accuracy", marker="s")
        plt.plot(epochs_list, val_recall_history, label="Validation Recall", marker="d")  # Add recall plot
        plt.xlabel("Epochs")
        plt.ylabel("Performance Metrics")
        plt.title(f"Final Training vs. Validation Metrics")
        plt.legend()
        plt.grid(True)

        # Annotate the point where the model was last saved
        if last_saved_epoch is not None:
            plt.annotate(f"Last Saved\nEpoch: {last_saved_epoch}\nVal Acc: {last_saved_val_acc:.4f}\nVal Recall: {last_saved_val_recall:.4f}",
                        xy=(last_saved_epoch, last_saved_val_acc),
                        xytext=(last_saved_epoch, last_saved_val_acc + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=16,    
                        horizontalalignment='center')
            
        plt.savefig(f'datasets/BigVul/train/training.png')
        plt.show()

    # TEST
    #-----------------------
    model.load(model_save_path)
    model.eval()
    test_metrics = eval_model(model, test_loader, info=True)
    test_acc_history = test_metrics["Accuracy"]
    test_recall_history = test_metrics["Recall"]
    print(f" - Test: Acc = {test_metrics['Accuracy']:.4f} | Loss = {test_metrics['Loss']:.4f} | Precision = {test_metrics['Precision']:.4f} | Recall = {test_metrics['Recall']:.4f}")

    # Save metrics
    metrics_df = pd.DataFrame()
    metrics_df['train_epochs'] = [epochs_list]
    metrics_df['train_acc'] = [train_acc_history]
    metrics_df['last_saved_epoch'] = last_saved_epoch_history
    metrics_df['val_acc'] = [val_acc_history]
    metrics_df['val_recall'] = [val_recall_history]
    metrics_df['test_acc'] = test_acc_history
    metrics_df['test_recall'] = test_recall_history
    metrics_df.to_csv('datasets/BigVul/train/metrics.csv', index=False)