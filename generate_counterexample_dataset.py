import os
import re
import json
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

# Compatibility shim for pickle files created under NumPy 2.x module paths.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.numeric" not in sys.modules:
    sys.modules["numpy._core.numeric"] = np.core.numeric

# Define the available dataset choices
AVAILABLE_DATASETS = ["train", "valid", "test"]
JOERN_CLI_DIR = "joern/joern-cli/"
PATHS = {
        "cpg" : "tmp/cwe20cfa/cpg/",
        "source" : "tmp/cwe20cfa/source/",
        "input" : "tmp/cwe20cfa/input/",
        "model" : "tmp/cwe20cfa/model/",
        "tokens" : "tmp/tokens/",
        "w2v" : "tmp/cwe20cfa/w2v/"
    }
MAX_RETRIES = 3 # Maximum retry attempts

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset",
    nargs="*",
    help="Select dataset(s). If not provided, all datasets are used.",
    choices=AVAILABLE_DATASETS,
    default=AVAILABLE_DATASETS
)

args = parser.parse_args()

def ensure_directories_exist(paths):
    """
    Check if directories in the given dictionary exist, and create them if they don't.

    :param paths: Dictionary containing directory paths as values.
    """
    for key, path in paths.items():
        if not os.path.exists(path):  # Check if directory exists
            os.makedirs(path, exist_ok=True)  # Create directory (including parents if needed)

# DATA PROCESSING FUNCTIONS
def load_cwe20cfa_dataset(path: str):
    # Store data in a list
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line))  # Convert JSON string to Python dict
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Filter out columns
    df = df[["func","target","cwe"]].dropna()

    return df

def get_cwe_dict(dfs: List[pd.DataFrame]) -> Dict[str, int]:
    # Get CWE distribution
    cwe_dict = {}
    for df in dfs:
        for id, number in df.cwe.value_counts().items():
            try: 
                cwe = id[0]
                try:
                    cwe_dict[cwe] += number
                except KeyError:
                    cwe_dict[cwe] = number
            except IndexError:
                pass
    
    return cwe_dict

def save_pickle(df: pd.DataFrame, file_path: str):
    df.to_pickle(file_path)

def filter_dataset_by_cwe(df: pd.DataFrame, cwe: str) -> pd.DataFrame:

    # Filter out the other CWEs
    cwe_df = df[df.cwe.apply(lambda x: cwe in x )]
    # Target 0 first
    cwe_df = cwe_df.sort_values(by="target", ascending=True, kind="stable")

    return cwe_df

# JOERN FUNCTIONS
def joern_parse(joern_cli_path, input_path, output_path, file_name):
    """
    Parses source code files into Joern's intermediate representation (Coded Property Graph - CPG).
    
    This function runs the `joern-parse` command-line tool to convert source code into a binary CPG file, 
    which is later used for code analysis in Joern. The function executes the command using `subprocess.run` 
    and returns the generated binary file's name.

    Parameters:
    - joern_cli_path (str): Path to the Joern CLI installation directory.
    - input_path (str): Path to the directory containing the source code to be parsed.
    - output_path (str): Path where the parsed CPG binary file should be saved.
    - file_name (str): Base name for the output binary file (without extension).

    Returns:
    - str: The name of the generated binary file.
    """
    out_file = file_name + ".bin"
    # Subprocess calling
    joern_parse_call = subprocess.run(["./" + os.path.join(joern_cli_path, "joern-parse"), input_path, "--out", os.path.join(output_path, out_file)],
                                      stdout=subprocess.PIPE, text=True, check=True)
    # print(joern_parse_call.stdout)
    
    return out_file

def joern_create(joern_path, in_path, out_path, cpg_file, unique_id=None):
    """
    Executes a Joern script to extract function-level code property graphs (CPGs) 
    and saves the results as a JSON file using direct communication with the process.

    Instead of writing the script to a file, this function sends commands directly to 
    the Joern process via `stdin`. It:
    - Loads a previously generated CPG binary file.
    - Runs a predefined Joern script (`graph-for-funcs.sc`) to extract function-level graph data.
    - Exports the results to a JSON file.

    Parameters:
    - joern_path (str): Path to the Joern CLI installation directory.
    - in_path (str): Path where the input CPG binary file is located.
    - out_path (str): Directory where the generated JSON file should be stored.
    - cpg_file (str): Name of the CPG binary file (e.g., "example.bin").
    - unique_id (str/int): Unique identifier for creating thread-safe temporary files.

    Returns:
    - tuple: (json_file, temp_script_path) - The name of the generated JSON file and path to temp script.
    """

    # Generate JSON output file name
    json_file = f"{cpg_file.split('.')[0]}.json"

    # Path for temporary script with commands
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    
    # Create unique temporary script file per thread/process
    if unique_id is not None:
        commands_script__path = os.path.abspath(f"tmp/joern_temp_script_{unique_id}.sc")
    else:
        commands_script__path = os.path.abspath("tmp/joern_temp_script.sc")

    # Paths for script execution
    graph_script_path = os.path.abspath("joern/graph-for-funcs.sc")
    json_out = os.path.join(os.path.abspath(out_path), json_file)

    # Write commands to the script file
    with open(commands_script__path, 'w') as script_file:
        # Import CPG project
        script_file.write(f'importCpg("{os.path.abspath(in_path)}/{cpg_file}")\n')
        # Generate json graph
        script_file.write(f'cpg.runScript("{graph_script_path}").toString() |> "{json_out}"\n')
        # Delete project
        script_file.write(f'delete("{cpg_file}")\n')

    # Set environment variables to avoid interactive mode
    env = os.environ.copy()
    env["JOERN_INTERACTIVE"] = "false"

    # Run Joern process and communicate via stdin (forcing non-interactive mode)
    joern_process = subprocess.Popen(
        [os.path.join(joern_path, "joern"), "--script", commands_script__path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,  # Pass modified environment variables
        bufsize=1,  # Enable line-buffering
    )

    # Send script commands to the process and close stdin
    outs, errs = joern_process.communicate(timeout=180)

    # Print any output or errors
    # if outs:
        # print(f"Outs: {outs}")
    # if errs:
        # print(f"Errs: {errs}")

    return json_file, commands_script__path

def graph_indexing(graph):
    func_name = graph["file"].split(".c")[0].split("/")[-1]
    del graph["file"]
    return func_name, {"functions": [graph]}

def json_process(in_path, json_file, debug_index=None):
    json_path = os.path.join(in_path, json_file)
    if os.path.exists(json_path):
        try:
            with open(json_path) as jf:
                cpg_string = jf.read()
                if not cpg_string.strip():
                    print(f"[ERROR] Empty JSON file at {json_path}" + (f" for example {debug_index}" if debug_index else ""))
                    return None
                cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
                cpg_json = json.loads(cpg_string)
                container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
                return container
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {json_path}" + (f" for example {debug_index}" if debug_index else "") + f": {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to process JSON {json_path}" + (f" for example {debug_index}" if debug_index else "") + f": {e}")
            return None
    else:
        print(f"[ERROR] JSON file not found: {json_path}" + (f" for example {debug_index}" if debug_index else ""))
    return None

# ADVERSARIAL FUNCTIONS
def generate_counterexample_example(example: pd.Series) -> pd.Series:
    """
    Use OpenAI API to generate a modified version of input_code.
    If the target_label is 1 (vulnerable), request a benign version.
    If the target_label is 0 (benign), introduce a vulnerability.
    """
    cwe = example.cwe[0]

    prompt_template = f"""
    The following is a {'vulnerable' if example.target == 1 else 'benign'} C function. 
    Please {'remove the vulnerability to make it safe' if example.target == 1 else f'introduce a security vulnerability (cwe: {cwe})'}.

    Original function:
    ```c
    {example.func}
    ```

    Modified function:
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_template}],
            stream=False,
            temperature=0.7
        )
        # Get response content
        response_content = response.choices[0].message.content
        # Extract function code
        if "```" in response_content:
            ce_func = response_content.split("```")[1].strip()
            
            if ce_func.startswith("c\n"):
                ce_func = ce_func[2:]
        # Create pandas series
        ce = pd.Series(data=[ce_func, 0 if example.target else 1, cwe, example.func], index=["func", "target", "cwe", "orig_func"])

        return ce
    
    except Exception as e:
        print("Error:", e)

        return None

def generate_counterexample_dataset(df: pd.DataFrame, output_path: str):
    # Drop rows where "cwe" is an empty list
    df_filtered = df[df["cwe"].apply(lambda lst: len(lst) > 0 and all(item.startswith("CWE-") for item in lst))]
    
    # Load existing counterexample dataset if it exists; otherwise, create an empty DataFrame
    if os.path.exists(output_path):
        ce_dataset = pd.read_pickle(output_path)
        print(f"Loaded existing counterexample dataset with {len(ce_dataset)} examples.")
    else:
        ce_dataset = pd.DataFrame(columns=['func', 'target', 'cwe', 'orig_func', 'cpg'])
    
    # Counterexample examples generation
    with Progress() as progress:

        # Main Task (Total Work)
        main_task = progress.add_task(
            f"[magenta]Generating counterexample dataset (0/{len(df_filtered)})...",
            total=len(df_filtered),
            bar_style="magenta"
        )
        
        # Create Sub-Task Once (Reused)
        secondary_task = progress.add_task(
            "[cyan]Coding counterexample example...",
            total=4,
            bar_style="cyan"
        )

        i = 0
        for index, example in df_filtered.iterrows():
            # Check if this example has already been processed.
            if i < len(ce_dataset) and not ce_dataset.iloc[i].empty and (ce_dataset.iloc[i].name== example.name).any():
                progress.update(main_task, advance=1, 
                    description=f"[magenta]Skipping already processed example ({i}/{len(df_filtered)})...")
                i += 1
                continue

            retry_attempts = 0
            success = False

            while retry_attempts < MAX_RETRIES and not success:
                try:
                    progress.update(secondary_task, completed=0, description=f"[cyan]Coding counterexample example... (Attempt {retry_attempts+1})")

                    # Subtask 1: Generate counterexample example
                    ce_example = generate_counterexample_example(example)
                    progress.update(secondary_task, advance=1)

                    # Subtask 2: Code Parsing
                    progress.update(secondary_task, advance=1, description=f"[cyan]Parsing source code...")

                    # Save counterexample func as C file
                    source_file_path = os.path.join(PATHS["source"], f"{index}.c")
                    with open(source_file_path, 'w') as f:
                        f.write(ce_example.func)

                    # Parsing function to .bin
                    cpg_file = joern_parse(JOERN_CLI_DIR, source_file_path, PATHS['cpg'], f"{index}_cpg")

                    # Subtask 3: Create CPG graphs JSON file
                    progress.update(secondary_task, advance=1, description=f"[cyan]Creating CPG with Joern...")
                    json_file = joern_create(JOERN_CLI_DIR, PATHS['cpg'], PATHS['cpg'], cpg_file)

                    # Subtask 4: Get CPG obj from JSON
                    progress.update(secondary_task, advance=1, description=f"[cyan]Processing CPG...")
                    ce_graphs = json_process(PATHS['cpg'], json_file)
                    ce_cpg = ce_graphs[0][1]  # Get the CPG from graphs
                    ce_example["cpg"] = ce_cpg
                    ce_example.name = example.name
                    save_pickle(ce_example, os.path.join(PATHS['cpg'], f"{index}_cpg.pkl"))
                    progress.update(secondary_task, description=f"[green]CPG Completed.")

                    # Remove unused files
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.bin"))
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.json"))
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.pkl"))
                    os.remove(os.path.join(PATHS['source'],f"{index}.c"))

                    # Add counterexample example
                    ce_dataset = pd.concat([ce_dataset, ce_example.to_frame().T])

                except Exception as e:
                    retry_attempts += 1
                    progress.update(secondary_task, description=f"[red]Error occurred! Retrying ({retry_attempts}/{MAX_RETRIES})...")

                    if retry_attempts == MAX_RETRIES:
                        print(f"[ERROR] Failed to process example {index} after {MAX_RETRIES} attempts: {e}")

            # Update main task progress
            progress.update(main_task, advance=1, description=f"[magenta]Generating counterexample dataset ({i+1}/{len(df_filtered)})...")
            i += 1
            # Mark as successful and exit retry loop
            success = True

            # Save dataset (every 10 data points)
            if not i % 10:
                save_pickle(ce_dataset, output_path)
                print(f"Saved dataset at {output_path}")

def process_single_example(index: int, example: pd.Series):
    """
    Process one example (a row from df) to generate an counterexample example.
    Each subtask is retried up to MAX_RETRIES on failure.
    
    Returns a tuple (index, ce_example) on success, or None on failure.
    """
    # -------------------------------
    # Step 1: Generate counterexample example
    for attempt in range(1, MAX_RETRIES+1):
        try:
            ce_example = generate_counterexample_example(example)
            break  # Successful, exit retry loop.
        except Exception as e:
            print(f"[ERROR] Example {index} - Step 1 (Generate counterexample example) attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed Step 1 for example {index}. Skipping this example.")
                return None

    # -------------------------------
    # Step 2: Code Parsing (writing file and parsing)
    for attempt in range(1, MAX_RETRIES+1):
        try:
            source_file_path = os.path.join(PATHS["source"], f"{index}.c")
            with open(source_file_path, 'w') as f:
                f.write(ce_example.func)
            # Parsing function to .bin
            cpg_file = joern_parse(JOERN_CLI_DIR, source_file_path, PATHS['cpg'], f"{index}_cpg")
            break
        except Exception as e:
            print(f"[ERROR] Example {index} - Step 2 (Code Parsing) attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed Step 2 for example {index}. Skipping this example.")
                return None

    # -------------------------------
    # Step 3: Create CPG graphs JSON file (with unique temp file per thread)
    temp_script_path = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            json_file, temp_script_path = joern_create(JOERN_CLI_DIR, PATHS['cpg'], PATHS['cpg'], cpg_file, unique_id=index)
            # Verify the JSON file was actually created
            json_path = os.path.join(PATHS['cpg'], json_file)
            if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
                raise Exception(f"Joern failed to create valid JSON file at {json_path}")
            break
        except Exception as e:
            print(f"[ERROR] Example {index} - Step 3 (Create CPG JSON) attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed Step 3 for example {index}. Skipping this example.")
                # Cleanup temp script if it exists
                if temp_script_path and os.path.exists(temp_script_path):
                    try:
                        os.remove(temp_script_path)
                    except Exception:
                        pass
                return None

    # -------------------------------
    # Step 4: Process CPG
    for attempt in range(1, MAX_RETRIES+1):
        try:
            ce_graphs = json_process(PATHS['cpg'], json_file, debug_index=index)
            # Safety check: verify ce_graphs is not None and has expected structure
            if ce_graphs is None or len(ce_graphs) == 0:
                raise Exception(f"json_process returned None or empty list for example {index}")
            ce_cpg = ce_graphs[0][1]  # Get the CPG from graphs
            ce_example["cpg"] = ce_cpg
            break
        except (TypeError, IndexError, KeyError) as e:
            print(f"[ERROR] Example {index} - Step 4 (Process CPG) attempt {attempt}/{MAX_RETRIES} failed with data structure error: {e}")
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed Step 4 for example {index}. Skipping this example.")
                # Cleanup temp script if it exists
                if temp_script_path and os.path.exists(temp_script_path):
                    try:
                        os.remove(temp_script_path)
                    except Exception:
                        pass
                return None
        except Exception as e:
            print(f"[ERROR] Example {index} - Step 4 (Process CPG) attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Failed Step 4 for example {index}. Skipping this example.")
                # Cleanup temp script if it exists
                if temp_script_path and os.path.exists(temp_script_path):
                    try:
                        os.remove(temp_script_path)
                    except Exception:
                        pass
                return None

    # -------------------------------
    # Finalize and cleanup
    try:
        ce_example.name = example.name  # set unique key
        save_pickle(ce_example, os.path.join(PATHS['cpg'], f"{index}_cpg.pkl"))
    except Exception as e:
        print(f"[ERROR] Example {index} - Finalization failed: {e}")
        return None

    # Remove temporary files (best effort cleanup; errors here won't stop processing)
    try:
        for filename in [f"{index}_cpg.bin", f"{index}_cpg.json", f"{index}_cpg.pkl"]:
            filepath = os.path.join(PATHS['cpg'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        filepath = os.path.join(PATHS['source'], f"{index}.c")
        if os.path.exists(filepath):
            os.remove(filepath)
        # Cleanup thread-specific temporary script file
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    except Exception as cleanup_e:
        print(f"[WARNING] Could not remove temporary files for example {index}: {cleanup_e}")

    return (index, ce_example)

def generate_counterexample_dataset(df: pd.DataFrame, output_path: str):
    # Filter out rows with empty or invalid "cwe"
    df_filtered = df[df["cwe"].apply(lambda lst: len(lst) > 0 and all(item.startswith("CWE-") for item in lst))]
    
    # Load existing counterexample dataset if it exists; otherwise, create an empty DataFrame
    if os.path.exists(output_path):
        ce_dataset = pd.read_pickle(output_path)
        print(f"Loaded existing counterexample dataset with {len(ce_dataset)} examples.")
    else:
        ce_dataset = pd.DataFrame(columns=['func', 'target', 'cwe', 'orig_func', 'cpg'])
    
    # Use a ThreadPoolExecutor to process examples concurrently
    with ThreadPoolExecutor(max_workers=4) as executor, Progress() as progress:
        # Main progress task for the total number of examples
        main_task = progress.add_task(
            f"[magenta]Generating counterexample dataset (0/{len(df_filtered)})...",
            total=len(df_filtered),
            bar_style="magenta"
        )

        i = 0
        for index, example in df_filtered.iterrows():
            # Check if this example has already been processed.
            if i < len(ce_dataset) and not ce_dataset.iloc[i].empty and (ce_dataset.iloc[i].name== example.name).any():
                progress.update(main_task, advance=1, 
                    description=f"[magenta]Skipping already processed example ({i}/{len(df_filtered)})...")
                i += 1
                continue
        
        # Submit one task per row in the filtered DataFrame.
        # We assume that each row's index is unique.
        futures = {
            executor.submit(process_single_example, index, example): index 
            for index, example in df_filtered.iterrows()
        }
        
        # As each future completes, update progress and add the result to the dataset
        for future in as_completed(futures):
            progress.update(main_task, advance=1, description=f"[magenta]Generating counterexample dataset ({i}/{len(df_filtered)})...")
            result = future.result()
            if result is not None:
                idx, adv_example = result
                # Avoid duplicates: if idx is already in the dataset, skip
                if idx not in ce_dataset.index:
                    ce_dataset = pd.concat([ce_dataset, adv_example.to_frame().T]) 
            i += 1
            # Save dataset (every 10 data points)
            if not i % 10:
                save_pickle(ce_dataset, output_path)
                print(f"Saved dataset at {output_path}")
    
    # Finally, reindex ce_dataset if needed (here we assume each row's index is the key)
    return ce_dataset

if __name__ == "__main__":

    # API Key Loading
    load_dotenv()
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Files path checking
    ensure_directories_exist(PATHS)

    cwe20cfa_datasets = {}
    # Dataset loading
    for dataset in args.dataset:
        print(f"Loading {dataset} dataset...")
        cwe20cfa_dataset = load_cwe20cfa_dataset(f"datasets/cwe20cfa/raw/cwe20cfa_{dataset}.jsonl")
        cwe20cfa_datasets[dataset] = cwe20cfa_dataset

    # Get cwe keys
    cwe_dict = get_cwe_dict(list(cwe20cfa_datasets.values()))
    
    # Filter out the other CWEs
    cwes = ["CWE-20"] # Should be the easier to detect with AST by a GNN

    # Counterexample dataset generation for each CWE
    for dataset in cwe20cfa_datasets:
        print(f"\nGenerating counterexample dataset for {dataset.upper()}")
        print("-----------------------------------------")
        cwe20cfa_dataset = cwe20cfa_datasets[dataset]
        for cwe in cwes:
            print("\n===============")
            print(f"CWE: {cwe}")
            print("===============")

            cwe_cwe20cfa = filter_dataset_by_cwe(cwe20cfa_dataset, cwe)
            generate_counterexample_dataset(cwe_cwe20cfa, f"datasets/cwe20cfa/cwe20cfa_{cwe}_augmented_{dataset}.pkl")
            print(f"Saved Counterexample {dataset.upper()} dataset (CWE: {cwe}).\n")
