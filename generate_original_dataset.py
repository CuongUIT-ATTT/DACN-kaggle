import os
import re
import json
import sys
import argparse
import subprocess
import threading
import signal
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric

AVAILABLE_DATASETS = ["train", "valid", "test"]
JOERN_CLI_DIR = "joern/joern-cli/"
PATHS = {
    "cpg": "tmp/cwe20cfa/cpg/",
    "source": "tmp/cwe20cfa/source/",
    "input": "tmp/cwe20cfa/input/",
    "model": "tmp/cwe20cfa/model/",
    "tokens": "tmp/tokens/",
    "w2v": "tmp/cwe20cfa/w2v/",
}
MAX_RETRIES = 3
DEFAULT_JOERN_TIMEOUT = 300
DEFAULT_SAVE_EVERY = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_LOCK_TIMEOUT = 30
DEFAULT_SAMPLE_TIMEOUT = 240

# Running many Joern export scripts at once is unstable; serialize this step.
JOERN_CREATE_LOCK = threading.Lock()

# Protect shared failed-examples list under multithreading.
FAILED_EXAMPLES_LOCK = threading.Lock()

# Track failed examples for later inspection
FAILED_EXAMPLES = []

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
    "--workers",
    type=int,
    default=4,
    help="Number of parallel workers.",
)
parser.add_argument(
    "--cwe",
    type=str,
    default="CWE-20",
    help="CWE to keep (default: CWE-20).",
)
parser.add_argument(
    "--joern-timeout",
    type=int,
    default=DEFAULT_JOERN_TIMEOUT,
    help="Timeout (seconds) for a single Joern export script run.",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=DEFAULT_SAVE_EVERY,
    help="Save partial pickle every N processed examples.",
)
parser.add_argument(
    "--max-retries",
    type=int,
    default=DEFAULT_MAX_RETRIES,
    help="Max retry attempts per example before skipping (default: 2 to avoid hangs).",
)
parser.add_argument(
    "--lock-timeout",
    type=int,
    default=DEFAULT_LOCK_TIMEOUT,
    help="Max seconds to wait for JOERN_CREATE_LOCK before failing the current sample.",
)
parser.add_argument(
    "--sample-timeout",
    type=int,
    default=DEFAULT_SAMPLE_TIMEOUT,
    help="Hard timeout (seconds) for processing one sample end-to-end.",
)
args = parser.parse_args()


def ensure_directories_exist(paths: Dict[str, str]):
    for path in paths.values():
        os.makedirs(path, exist_ok=True)


def load_cwe20cfa_dataset(path: str) -> pd.DataFrame:
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df = df[["func", "target", "cwe"]].dropna()
    return df


def filter_dataset_by_cwe(df: pd.DataFrame, cwe: str) -> pd.DataFrame:
    cwe_df = df[df.cwe.apply(lambda x: isinstance(x, list) and cwe in x)]
    cwe_df = cwe_df.sort_values(by="target", ascending=True, kind="stable")
    return cwe_df


def save_pickle(df: pd.DataFrame, file_path: str):
    df.to_pickle(file_path)


def joern_parse(joern_cli_path: str, input_path: str, output_path: str, file_name: str) -> str:
    out_file = file_name + ".bin"
    out_path = os.path.join(output_path, out_file)
    cmd = [
        "./" + os.path.join(joern_cli_path, "joern-parse"),
        input_path,
        "--out",
        out_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=max(1, args.joern_timeout),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"joern-parse timed out after {max(1, args.joern_timeout)}s for {file_name}."
        ) from e

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"joern-parse produced missing/empty bin file: {out_path}")

    return out_file


def joern_create(
    joern_path: str,
    in_path: str,
    out_path: str,
    cpg_file: str,
    unique_id: Optional[int] = None,
    timeout_seconds: int = DEFAULT_JOERN_TIMEOUT,
    lock_timeout_seconds: int = DEFAULT_LOCK_TIMEOUT,
) -> Tuple[str, str]:
    json_file = f"{cpg_file.split('.')[0]}.json"

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    if unique_id is not None:
        commands_script_path = os.path.abspath(f"tmp/joern_temp_script_{unique_id}.sc")
    else:
        commands_script_path = os.path.abspath("tmp/joern_temp_script.sc")

    graph_script_path = os.path.abspath("joern/graph-for-funcs.sc")
    json_out = os.path.join(os.path.abspath(out_path), json_file)

    with open(commands_script_path, "w", encoding="utf-8") as script_file:
        script_file.write(f'importCpg("{os.path.abspath(in_path)}/{cpg_file}")\n')
        script_file.write(f'cpg.runScript("{graph_script_path}").toString() |> "{json_out}"\n')
        script_file.write(f'delete("{cpg_file}")\n')

    env = os.environ.copy()
    env["JOERN_INTERACTIVE"] = "false"

    acquired = JOERN_CREATE_LOCK.acquire(timeout=max(1, lock_timeout_seconds))
    if not acquired:
        raise RuntimeError(
            f"Timed out waiting {max(1, lock_timeout_seconds)}s for JOERN_CREATE_LOCK while exporting {cpg_file}."
        )

    try:
        joern_process = subprocess.Popen(
            [os.path.join(joern_path, "joern"), "--script", commands_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            start_new_session=True,
        )

        try:
            _, stderr = joern_process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as e:
            # Kill the entire process group so no orphan Java child remains.
            try:
                os.killpg(joern_process.pid, signal.SIGKILL)
            except Exception:
                joern_process.kill()
            _, stderr = joern_process.communicate()
            raise RuntimeError(
                f"Joern export timed out after {timeout_seconds}s for {cpg_file}. "
                f"Script: {commands_script_path}. stderr tail: {str(stderr)[-500:]}"
            ) from e

        if joern_process.returncode != 0:
            raise RuntimeError(
                f"Joern export failed with exit code {joern_process.returncode} for {cpg_file}. "
                f"stderr tail: {str(stderr)[-500:]}"
            )
    finally:
        JOERN_CREATE_LOCK.release()

    return json_file, commands_script_path


def graph_indexing(graph: Dict) -> Tuple[str, Dict]:
    func_name = graph["file"].split(".c")[0].split("/")[-1]
    del graph["file"]
    return func_name, {"functions": [graph]}


def json_process(in_path: str, json_file: str, debug_index: Optional[int] = None):
    json_path = os.path.join(in_path, json_file)
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found: {json_path} for example {debug_index}")
        return None

    try:
        with open(json_path, encoding="utf-8") as jf:
            cpg_string = jf.read()
            if not cpg_string.strip():
                print(f"[ERROR] Empty JSON file at {json_path} for example {debug_index}")
                return None
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", "", cpg_string)
            cpg_json = json.loads(cpg_string)

            functions = cpg_json.get("functions", [])
            if not functions:
                print(f"[ERROR] No functions field/entries in JSON {json_path} for example {debug_index}")
                return None

            valid_graphs = [graph for graph in functions if graph.get("file") != "N/A"]
            selected_graphs = valid_graphs if valid_graphs else functions

            if not valid_graphs:
                print(
                    f"[WARNING] All function.file are N/A in {json_path} for example {debug_index}; "
                    "using fallback graphs to avoid empty result."
                )

            container = [graph_indexing(graph) for graph in selected_graphs]
            return container
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_path} for example {debug_index}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process JSON {json_path} for example {debug_index}: {e}")
        return None


def process_single_example(index: int, example: pd.Series, max_retries: int = DEFAULT_MAX_RETRIES):
    start_time = time.monotonic()
    sample_timeout = max(1, args.sample_timeout)

    def ensure_not_timed_out(stage: str):
        elapsed = time.monotonic() - start_time
        if elapsed > sample_timeout:
            raise TimeoutError(
                f"Sample {index} exceeded sample-timeout {sample_timeout}s at stage '{stage}' (elapsed={elapsed:.1f}s)."
            )

    temp_script_path = None

    # Step 1: Parse original source to CPG bin
    for attempt in range(1, max_retries + 1):
        try:
            ensure_not_timed_out("parse")
            source_file_path = os.path.join(PATHS["source"], f"{index}.c")
            with open(source_file_path, "w", encoding="utf-8") as f:
                f.write(example.func)
            cpg_file = joern_parse(JOERN_CLI_DIR, source_file_path, PATHS["cpg"], f"{index}_cpg")
            ensure_not_timed_out("parse")
            break
        except Exception as e:
            print(f"[ERROR] Example {index} - Parse attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                with FAILED_EXAMPLES_LOCK:
                    FAILED_EXAMPLES.append({"index": index, "stage": "parse", "reason": str(e)[:100]})
                return None

    # Step 2: Export CPG JSON
    for attempt in range(1, max_retries + 1):
        try:
            ensure_not_timed_out("export_json")
            json_file, temp_script_path = joern_create(
                JOERN_CLI_DIR,
                PATHS["cpg"],
                PATHS["cpg"],
                cpg_file,
                unique_id=index,
                timeout_seconds=max(1, args.joern_timeout),
                lock_timeout_seconds=max(1, args.lock_timeout),
            )
            json_path = os.path.join(PATHS["cpg"], json_file)
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                raise RuntimeError(f"Joern failed to create valid JSON at {json_path}")
            ensure_not_timed_out("export_json")
            break
        except Exception as e:
            print(f"[ERROR] Example {index} - Export JSON attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                with FAILED_EXAMPLES_LOCK:
                    FAILED_EXAMPLES.append({"index": index, "stage": "export_json", "reason": str(e)[:100]})
                if temp_script_path and os.path.exists(temp_script_path):
                    os.remove(temp_script_path)
                return None

    # Step 3: Read JSON as CPG object
    for attempt in range(1, max_retries + 1):
        try:
            ensure_not_timed_out("read_json")
            graphs = json_process(PATHS["cpg"], json_file, debug_index=index)
            if graphs is None or len(graphs) == 0:
                raise RuntimeError("json_process returned empty result")
            cpg = graphs[0][1]
            out_row = pd.Series(
                {
                    "func": example.func,
                    "target": int(example.target),
                    "cwe": example.cwe,
                    "cpg": cpg,
                }
            )
            out_row.name = example.name
            ensure_not_timed_out("read_json")
            break
        except Exception as e:
            print(f"[ERROR] Example {index} - Process CPG attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                with FAILED_EXAMPLES_LOCK:
                    FAILED_EXAMPLES.append({"index": index, "stage": "read_json", "reason": str(e)[:100]})
                if temp_script_path and os.path.exists(temp_script_path):
                    os.remove(temp_script_path)
                return None

    # Cleanup temp files
    try:
        for filename in [f"{index}_cpg.bin", f"{index}_cpg.json"]:
            filepath = os.path.join(PATHS["cpg"], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        source_path = os.path.join(PATHS["source"], f"{index}.c")
        if os.path.exists(source_path):
            os.remove(source_path)
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    except Exception as cleanup_e:
        print(f"[WARNING] Could not remove temp files for example {index}: {cleanup_e}")

    return index, out_row


def generate_original_dataset(
    df: pd.DataFrame,
    output_path: str,
    workers: int = 4,
    save_every: int = DEFAULT_SAVE_EVERY,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    with FAILED_EXAMPLES_LOCK:
        FAILED_EXAMPLES.clear()

    df_filtered = df[df["cwe"].apply(lambda lst: len(lst) > 0 and all(item.startswith("CWE-") for item in lst))]

    if os.path.exists(output_path):
        output_df = pd.read_pickle(output_path)
        print(f"Loaded existing dataset with {len(output_df)} examples.")
    else:
        output_df = pd.DataFrame(columns=["func", "target", "cwe", "cpg"])

    # Skip indices already processed
    pending_rows = [(idx, row) for idx, row in df_filtered.iterrows() if idx not in output_df.index]
    if not pending_rows:
        print("No pending rows to process.")
        return output_df

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor, Progress() as progress:
        main_task = progress.add_task(
            f"[magenta]Generating ORIGINAL dataset (0/{len(pending_rows)})...",
            total=len(pending_rows),
            bar_style="magenta",
        )

        futures = {
            executor.submit(process_single_example, index, row, max_retries): index
            for index, row in pending_rows
        }

        done = 0
        failed_count = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                idx, row = result
                output_df.loc[idx] = row
            else:
                failed_count += 1

            done += 1
            progress.update(
                main_task,
                advance=1,
                description=f"[magenta]Generating ORIGINAL dataset ({done}/{len(pending_rows)}, {failed_count} failed)...",
            )

            if done % max(1, save_every) == 0:
                save_pickle(output_df, output_path)
                print(f"Saved dataset at {output_path}")
                with FAILED_EXAMPLES_LOCK:
                    failed_snapshot = list(FAILED_EXAMPLES)
                if failed_snapshot:
                    failed_df = pd.DataFrame(failed_snapshot)
                    failed_csv_path = output_path.replace(".pkl", "_failed.csv")
                    failed_df.to_csv(failed_csv_path, index=False)
                    print(f"Updated failed examples log at {failed_csv_path}")

    save_pickle(output_df, output_path)

    with FAILED_EXAMPLES_LOCK:
        failed_snapshot = list(FAILED_EXAMPLES)
    if failed_snapshot:
        failed_df = pd.DataFrame(failed_snapshot)
        failed_csv_path = output_path.replace(".pkl", "_failed.csv")
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"Final failed examples log: {failed_csv_path}")

    return output_df


if __name__ == "__main__":
    ensure_directories_exist(PATHS)

    cwe20cfa_datasets = {}
    for dataset in args.dataset:
        print(f"Loading {dataset} dataset...")
        cwe20cfa_dataset = load_cwe20cfa_dataset(f"datasets/cwe20cfa/raw/cwe20cfa_{dataset}.jsonl")
        cwe20cfa_datasets[dataset] = cwe20cfa_dataset

    for dataset, dataset_df in cwe20cfa_datasets.items():
        print(f"\nGenerating ORIGINAL dataset for {dataset.upper()}")
        print("-----------------------------------------")
        print(f"CWE: {args.cwe}")

        filtered_df = filter_dataset_by_cwe(dataset_df, args.cwe)
        out_path = f"datasets/cwe20cfa/cwe20cfa_{args.cwe}_original_{dataset}.pkl"

        final_df = generate_original_dataset(
            filtered_df,
            out_path,
            workers=args.workers,
            save_every=args.save_every,
            max_retries=args.max_retries,
        )
        save_pickle(final_df, out_path)
        print(f"Saved ORIGINAL {dataset.upper()} dataset at {out_path}\n")
