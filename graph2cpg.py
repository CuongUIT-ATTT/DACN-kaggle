import os
import re
import gc
import json
import sys
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric

AVAILABLE_DATASETS = ["train", "valid", "test"]
BASE_DIR = os.getcwd()
JOERN_CLI_DIR = os.path.join(BASE_DIR, "joern", "joern-cli")
GRAPH_SCRIPT_PATH = os.path.join(BASE_DIR, "joern", "graph-for-funcs.sc")
PATHS = {
    "cpg": os.path.join(BASE_DIR, "tmp", "cwe20cfa", "cpg"),
    "source": os.path.join(BASE_DIR, "tmp", "cwe20cfa", "source"),
    "input": os.path.join(BASE_DIR, "tmp", "cwe20cfa", "input"),
    "model": os.path.join(BASE_DIR, "tmp", "cwe20cfa", "model"),
    "tokens": os.path.join(BASE_DIR, "tmp", "tokens"),
    "w2v": os.path.join(BASE_DIR, "tmp", "cwe20cfa", "w2v"),
}

DEFAULT_BATCH_SIZE = 50
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
MAX_RETRIES = 3


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    help="Select dataset(s). If not provided, all datasets are used.",
    choices=AVAILABLE_DATASETS,
    default=AVAILABLE_DATASETS,
)
parser.add_argument(
    "--mode",
    type=str,
    default="augmented",
    choices=["augmented", "original"],
    help="Processing mode: 'augmented' (orig_func→orig_cpg) or 'original' (func→cpg).",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help="Number of samples per Joern batch.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=DEFAULT_WORKERS,
    help="Number of worker processes. Recommended <= cpu_count()//2 due to Joern RAM usage.",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=900,
    help="Timeout (seconds) per batch worker.",
)
args = parser.parse_args()


def ensure_directories_exist(paths: Dict[str, str]):
    for path in paths.values():
        os.makedirs(path, exist_ok=True)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_joern_script(cpg_bin_path: str, json_out_path: str, script_path: str):
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write(f'importCpg("{cpg_bin_path}")\n')
        script_file.write(f'cpg.runScript("{GRAPH_SCRIPT_PATH}").toString() |> "{json_out_path}"\n')
        script_file.write(f'delete("{os.path.basename(cpg_bin_path)}")\n')


def run_joern_parse(input_dir: str, output_bin_path: str):
    cmd = [
        os.path.join(JOERN_CLI_DIR, "joern-parse"),
        input_dir,
        "--out",
        output_bin_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def run_joern_script(script_path: str, timeout_seconds: int):
    env = os.environ.copy()
    env["JOERN_INTERACTIVE"] = "false"
    cmd = [os.path.join(JOERN_CLI_DIR, "joern"), "--script", script_path]
    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout_seconds,
        check=True,
    )


def parse_batch_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        return {}

    with open(json_path, "r", encoding="utf-8") as jf:
        cpg_string = jf.read()

    cpg_string = re.sub(r"io\\.shiftleft\\.codepropertygraph\\.generated\\.", "", cpg_string)
    payload = json.loads(cpg_string)

    indexed: Dict[str, Dict[str, Any]] = {}
    for graph in payload.get("functions", []):
        file_path = graph.get("file")
        if not file_path or file_path == "N/A":
            continue

        index_key = os.path.splitext(os.path.basename(file_path))[0]
        if index_key in indexed:
            # Keep first graph for consistency with old single-function flow.
            continue

        graph_copy = dict(graph)
        graph_copy.pop("file", None)
        indexed[index_key] = {"functions": [graph_copy]}

    return indexed


def cleanup_batch_artifacts(batch_source_dir: str, batch_bin_path: str, batch_json_path: str, batch_script_path: str):
    shutil.rmtree(batch_source_dir, ignore_errors=True)
    for path in [batch_bin_path, batch_json_path, batch_script_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def process_batch(task: Tuple[str, List[Tuple[int, str]], int, int]):
    """
    Process one batch in one worker:
    1) write .c files for the batch
    2) run joern-parse once on the whole directory
    3) run joern script once to export all functions as JSON
    4) map JSON results back to original DataFrame indices
    """
    batch_id, batch_examples, max_retries, joern_timeout = task

    batch_source_dir = os.path.join(PATHS["source"], f"batch_{batch_id}")
    os.makedirs(batch_source_dir, exist_ok=True)

    batch_bin_path = os.path.join(PATHS["cpg"], f"batch_{batch_id}.bin")
    batch_json_path = os.path.join(PATHS["cpg"], f"batch_{batch_id}.json")
    batch_script_path = os.path.join(BASE_DIR, "tmp", f"joern_batch_{batch_id}.sc")

    id_to_index: Dict[str, int] = {}

    try:
        for index, code in batch_examples:
            id_str = str(index)
            id_to_index[id_str] = index
            source_path = os.path.join(batch_source_dir, f"{id_str}.c")
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(code)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                run_joern_parse(batch_source_dir, batch_bin_path)
                build_joern_script(batch_bin_path, batch_json_path, batch_script_path)
                run_joern_script(batch_script_path, joern_timeout)
                break
            except Exception as err:
                last_error = err
                if attempt == max_retries:
                    raise

        indexed_graphs = parse_batch_json(batch_json_path)

        success_map: Dict[int, Dict[str, Any]] = {}
        failed_indices: List[int] = []

        for id_str, index in id_to_index.items():
            cpg = indexed_graphs.get(id_str)
            if cpg is None:
                failed_indices.append(index)
            else:
                success_map[index] = cpg

        return {
            "batch_id": batch_id,
            "success_map": success_map,
            "failed_indices": failed_indices,
            "error": str(last_error) if (not success_map and last_error is not None) else None,
        }

    except Exception as e:
        return {
            "batch_id": batch_id,
            "success_map": {},
            "failed_indices": [index for index, _ in batch_examples],
            "error": str(e),
        }
    finally:
        cleanup_batch_artifacts(batch_source_dir, batch_bin_path, batch_json_path, batch_script_path)
        gc.collect()


def build_pending_examples(dataset_df: pd.DataFrame, mode: str = "augmented") -> List[Tuple[int, str]]:
    pending = []
    if mode == "augmented":
        # Mode: orig_func → orig_cpg
        for index, row in dataset_df[pd.isna(dataset_df["orig_cpg"])].iterrows():
            if "orig_func" in row.index and pd.notna(row["orig_func"]):
                pending.append((index, row["orig_func"]))
    else:  # mode == "original"
        # Mode: func → cpg
        for index, row in dataset_df[pd.isna(dataset_df["cpg"])].iterrows():
            if "func" in row.index and pd.notna(row["func"]):
                pending.append((index, row["func"]))
    return pending


if __name__ == "__main__":
    ensure_directories_exist(PATHS)

    for dataset in args.dataset:
        mode = args.mode
        print(f"\nGenerating CPG for {dataset.upper()} dataset ({mode.upper()} mode)")
        print("-" * 60)

        dataset_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_{dataset}.pkl"
        filepath = os.path.join(BASE_DIR, dataset_path)

        dataset_df = pd.read_pickle(filepath)
        
        # Initialize CPG columns for chosen mode
        if mode == "augmented":
            if "orig_cpg" not in dataset_df.columns:
                dataset_df["orig_cpg"] = pd.NA
            dataset_df["orig_cpg"] = dataset_df["orig_cpg"].astype(object)
            cpg_col_name = "orig_cpg"
        else:  # original
            if "cpg" not in dataset_df.columns:
                dataset_df["cpg"] = pd.NA
            dataset_df["cpg"] = dataset_df["cpg"].astype(object)
            cpg_col_name = "cpg"

        pending_examples = build_pending_examples(dataset_df, mode=mode)
        if not pending_examples:
            print(f"No pending rows. Dataset already has {cpg_col_name} for all samples.")
            continue

        batch_size = max(1, args.batch_size)
        workers = max(1, args.workers)
        batches = chunk_list(pending_examples, batch_size)

        print(f"Pending rows: {len(pending_examples)}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {len(batches)}")
        print(f"Workers: {workers}")

        task_list: List[Tuple[str, List[Tuple[int, str]], int, int]] = []
        for batch_idx, batch_examples in enumerate(batches):
            task_list.append((f"{dataset}_{batch_idx}", batch_examples, MAX_RETRIES, args.timeout))

        completed_batches = 0
        dropped_rows = 0

        with Progress(
            TextColumn("[bold magenta]Processing {task.fields[dataset]} batches ({task.completed}/{task.total})..."),
            BarColumn(),
            TextColumn("[bold cyan]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                f"[magenta]Processing {dataset.upper()} batch jobs",
                total=len(task_list),
                dataset=dataset.upper(),
            )

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_batch, task): task[0] for task in task_list}

                for future in as_completed(futures):
                    batch_id = futures[future]
                    try:
                        result = future.result(timeout=args.timeout + 120)
                    except TimeoutError:
                        batch_examples = next((t[1] for t in task_list if t[0] == batch_id), [])
                        failed_indices = [idx for idx, _ in batch_examples]
                        dataset_df = dataset_df.drop(index=failed_indices, errors="ignore")
                        dropped_rows += len(failed_indices)
                        print(f"[ERROR] Batch {batch_id} timed out. Dropped {len(failed_indices)} rows.")
                    except Exception as e:
                        batch_examples = next((t[1] for t in task_list if t[0] == batch_id), [])
                        failed_indices = [idx for idx, _ in batch_examples]
                        dataset_df = dataset_df.drop(index=failed_indices, errors="ignore")
                        dropped_rows += len(failed_indices)
                        print(f"[ERROR] Batch {batch_id} crashed: {e}. Dropped {len(failed_indices)} rows.")
                    else:
                        success_map = result.get("success_map", {})
                        failed_indices = result.get("failed_indices", [])

                        for idx, cpg in success_map.items():
                            dataset_df.at[idx, cpg_col_name] = cpg

                        if failed_indices:
                            dataset_df = dataset_df.drop(index=failed_indices, errors="ignore")
                            dropped_rows += len(failed_indices)

                        if result.get("error"):
                            print(f"[WARN] Batch {batch_id}: {result['error']}")

                    completed_batches += 1
                    progress.update(main_task, advance=1)
                    progress.refresh()

                    # Save once per completed batch to avoid per-example I/O bottleneck.
                    dataset_df.to_pickle(filepath)
                    print(f"Saved dataset after batch {completed_batches}/{len(task_list)} at {filepath}")

        dataset_df.to_pickle(filepath)
        print(f"\nFinal dataset saved at {filepath}")
        print(f"Dropped rows: {dropped_rows}")
        print(f"Remaining rows: {len(dataset_df)}")
