import os
import re
import json
import argparse
import subprocess
import pandas as pd
import multiprocessing
from rich.progress import Progress
from multiprocessing import Pool, cpu_count
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

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
MAX_RETRIES = 10 # Maximum retry attempts
EXAMPLES_PER_SAVE = 1
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

def joern_create(joern_path, in_path, out_path, cpg_file):
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

    Returns:
    - str: The name of the generated JSON file.
    """

    # Generate JSON output file name
    json_file = f"{cpg_file.split('.')[0]}.json"

    # Path for temporary script with commands
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
        
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

    try:
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

        outs, errs = joern_process.communicate(timeout=120)  # Waits for completion with a 120s timeout

    except subprocess.TimeoutExpired:
        joern_process.kill()  # Kill the stuck process
        outs, errs = joern_process.communicate()  # Capture output after termination
        print(f"[ERROR] Joern process timed out after 120 seconds.")

    except Exception as e:
        print(f"[ERROR] Joern process failed: {e}")

    return json_file

def graph_indexing(graph):
    func_name = graph["file"].split(".c")[0].split("/")[-1]
    del graph["file"]
    return func_name, {"functions": [graph]}

def json_process(in_path, json_file):
    if os.path.exists(in_path+"/"+json_file):
        with open(in_path+"/"+json_file) as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None

def process_example(args):
    index, example, paths, joern_cli_dir, max_retries = args

    # Skip if already processed
    if not pd.isna(example["orig_cpg"]):
        return index, example["orig_cpg"]

    try:
        # -------------------------------
        # Step 2: Code Parsing
        source_file_path = os.path.join(paths["source"], f"{index}.c")
        with open(source_file_path, 'w') as f:
            f.write(example.orig_func)

        cpg_file = joern_parse(joern_cli_dir, source_file_path, paths['cpg'], f"{index}_cpg")

        # -------------------------------
        # Step 3: Create CPG graphs JSON file (retry separately)
        for attempt in range(1, max_retries + 1):
            try:
                json_file = joern_create(joern_cli_dir, paths['cpg'], paths['cpg'], cpg_file)
                if os.path.exists(os.path.join(paths['cpg'],json_file)) and os.path.getsize(os.path.join(paths['cpg'],json_file)) > 0:
                    break  # Success, exit retry loop
                else:
                    raise Exception(f"Empty or missing JSON file after attempt {attempt}")
            except Exception as e:
                if attempt == max_retries:
                    print(f"[ERROR] Failed Step 3 for example {index}. Max retries reached ({max_retries}). Skipping this example.")
                    
                    # Cleanup temporary files
                    for filename in [f"{index}_cpg.bin", f"{index}_cpg.json"]:
                        filepath = os.path.join(paths['source'], f"{index}.c")
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        filepath = os.path.join(paths['cpg'], filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                    
                    return index, None

        # -------------------------------
        # Step 4: Process CPG (retry separately)
        for attempt in range(1, max_retries + 1):
            try:
                graphs = json_process(paths['cpg'], json_file)
                if graphs and isinstance(graphs, list) and len(graphs) > 0 and len(graphs[0]) > 1:
                    cpg = graphs[0][1]  # Extract CPG
                    break  # Success, exit retry loop
                else:
                    raise Exception(f"Invalid graphs output after attempt {attempt}")
            except Exception as e:
                if attempt == max_retries:
                    print(f"[ERROR] Failed Step 4 for example {index}. Max retries reached ({max_retries}). Skipping this example.")
                    
                    # Cleanup temporary files
                    for filename in [f"{index}_cpg.bin", f"{index}_cpg.json"]:
                        filepath = os.path.join(paths['source'], f"{index}.c")
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        filepath = os.path.join(paths['cpg'], filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)         
                    
                    return index, None

        # Cleanup temporary files
        for filename in [f"{index}_cpg.bin", f"{index}_cpg.json"]:
            filepath = os.path.join(paths['source'], f"{index}.c")
            if os.path.exists(filepath):
                os.remove(filepath)
            filepath = os.path.join(paths['cpg'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)

        print(f"CPG generated for example {index}.")
        return index, cpg  # Successful processing

    except Exception as e:
        print(f"[ERROR] Example {index} - Unhandled failure: {e}")
        return index, None  # Fail-safe return

if __name__ == "__main__":
    for dataset in args.dataset:
        print(f"\nGenerating CPG for {dataset.upper()} dataset")
        print("-----------------------------------------")

        dataset_path = f"datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_{dataset}.pkl"
        print(dataset_path)
        filepath = os.path.join(os.getcwd(), dataset_path)
        dataset_df = pd.read_pickle(filepath)
        try:
            dataset_df["orig_cpg"] = dataset_df["orig_cpg"].astype(object)
        except KeyError:
            dataset_df["orig_cpg"] = pd.NA

        # Select only rows that still need processing (e.g., where "cpg" is NaN)
        filtered_df = dataset_df[pd.isna(dataset_df["orig_cpg"])]
        task_list = [(index, row, PATHS, JOERN_CLI_DIR, MAX_RETRIES) for index, row in filtered_df.iterrows()]

        num_workers = max(1,1)
        # num_workers = max(1, int(multiprocessing.cpu_count()/2))  # Use half of available CPUs, leaving one free
        timeout_per_task = 60  # seconds, adjust as needed

        # Setup rich progress bar
        with Progress(
            TextColumn("[bold magenta]Processing {task.fields[dataset]} ({task.completed}/{task.total})..."),
            BarColumn(),
            TextColumn("[bold cyan]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                f"[magenta]Processing {dataset.upper()} dataset", 
                total=len(task_list),
                dataset=dataset.upper()
            )

            results = {}
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_example, task): task[0] for task in task_list}

                i = 0
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        idx, cpg = future.result(timeout=timeout_per_task)

                        if cpg is not None:
                            dataset_df.at[idx, "orig_cpg"] = cpg  # Assign as dict directly
                        else:
                            dataset_df.drop(index=idx, inplace=True)  # Drop failed row

                        i += 1
                        progress.update(main_task, advance=1)
                        progress.refresh()

                        # Save dataset every X examples
                        if i % EXAMPLES_PER_SAVE == 0:
                            dataset_df.to_pickle(filepath)
                            print(f"Saved dataset at {filepath}")
                        
                    except TimeoutError:
                        print(f"[ERROR] Example {index} timed out. Skipping this example.")
                        continue
                    except Exception as e:
                        print(f"[ERROR] Example {index} raised an exception: {e}")
                        continue

        # Save the final dataset after all processing is complete
        dataset_df.to_pickle(filepath)
        print(f"Final dataset saved at {filepath}")