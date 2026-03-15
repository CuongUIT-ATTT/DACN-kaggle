import os
import re
import json
import pandas as pd
import subprocess
from rich.progress import Progress

JOERN_CLI_DIR = "joern/joern-cli/"
PATHS = {
        "cpg" : "tmp/cwe20cfa/cpg/",
        "source" : "tmp/cwe20cfa/source/",
        "input" : "tmp/cwe20cfa/input/",
    }
MAX_RETRIES = 3 # Maximum retry attempts

N = 9 #TODO

def load_cwe20cfa_dataset(path: str):
    df = pd.read_pickle(path)
    # Filter out columns
    df = df[["func","target","cwe"]].dropna()

    return df

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
    outs, errs = joern_process.communicate(timeout=60)

    # Print any output or errors
    # if outs:
        # print(f"Outs: {outs}")
    # if errs:
        # print(f"Errs: {errs}")

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

def process_dataset(df: pd.DataFrame, output_path: str):
    
    # Load existing dataset if it exists; otherwise, create an empty DataFrame
    if os.path.exists(output_path):
        dataset = pd.read_csv(output_path, index_col=0)
        print(f"Loaded existing dataset with {len(dataset)} examples.")
    else:
        dataset = pd.DataFrame(columns=['func', 'target', 'cwe', 'cpg'])
    
    with Progress() as progress:

        # Main Task (Total Work)
        main_task = progress.add_task(
            f"[magenta]Generating input examples dataset (0/{len(df)})...",
            total=len(df),
            bar_style="magenta"
        )
        
        # Create Sub-Task Once (Reused)
        secondary_task = progress.add_task(
            "[cyan]Example...",
            total=4,
            bar_style="cyan"
        )

        i = 0
        for index, example in df.iterrows():
        # for index, example in df.iterrows():
            
            # Check if this example has already been processed.
            matches = (dataset[dataset['func'] == example['func']]).index.tolist()
            if matches:
                idx_match = matches[0]
                progress.update(main_task, advance=1, 
                    description=f"[magenta]Skipping already processed example ({i}/{len(df)})...")
                if index != idx_match:
                    dataset.loc[index] = dataset.loc[dataset.index == idx_match].iloc[0].copy()
                    dataset = dataset.drop(index=idx_match)
                i += 1
                continue

            retry_attempts = 0
            success = False

            while retry_attempts < MAX_RETRIES and not success:
                try:
                    progress.update(secondary_task, completed=0, description=f"[cyan]Generating example... (Attempt {retry_attempts+1})")

                    # Subtask 2: Code Parsing
                    progress.update(secondary_task, advance=1, description=f"[cyan]Parsing source code...")

                    # Save func as C file
                    source_file_path = os.path.join(PATHS["source"], f"{index}.c")
                    with open(source_file_path, 'w') as f:
                        f.write(example.func)

                    # Parsing function to .bin
                    cpg_file = joern_parse(JOERN_CLI_DIR, source_file_path, PATHS['cpg'], f"{index}_cpg")

                    # Subtask 3: Create CPG graphs JSON file
                    progress.update(secondary_task, advance=1, description=f"[cyan]Creating CPG with Joern...")
                    json_file = joern_create(JOERN_CLI_DIR, PATHS['cpg'], PATHS['cpg'], cpg_file)

                    # Subtask 4: Get CPG obj from JSON
                    progress.update(secondary_task, advance=1, description=f"[cyan]Processing CPG...")
                    graphs = json_process(PATHS['cpg'], json_file)
                    cpg = graphs[0][1]  # Get the CPG from graphs
                    example["cpg"] = cpg
                    example.to_pickle(os.path.join(PATHS['cpg'], f"{index}_cpg.pkl"))
                    progress.update(secondary_task, description=f"[green]CPG Completed.")

                    # Remove unused files
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.bin"))
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.json"))
                    os.remove(os.path.join(PATHS['cpg'],f"{index}_cpg.pkl"))
                    os.remove(os.path.join(PATHS['source'],f"{index}.c"))

                    # Add example
                    dataset = pd.concat([dataset, example.to_frame().T])
                    success = True

                except Exception as e:
                    retry_attempts += 1
                    progress.update(secondary_task, description=f"[red]Error occurred! Retrying ({retry_attempts}/{MAX_RETRIES})...")
                    

                    if retry_attempts == MAX_RETRIES:
                        print(f"[ERROR] Failed to process example {index} after {MAX_RETRIES} attempts: {e}")

                        example["cpg"] = None
                        dataset = pd.concat([dataset, example.to_frame().T])
                        
            # Update main task progress
            progress.update(main_task, advance=1, description=f"[magenta]Generating dataset ({i+1}/{len(df)})...")
            i += 1
            # Mark as successful and exit retry loop
            success = True

            # Save dataset (every 10 data points)
            if not i % 100:
                dataset.to_csv(output_path)
                print(f"Saved dataset at {output_path}")

if __name__ == "__main__":

    bigvul_df = load_cwe20cfa_dataset(f"datasets/BigVul/bigvul_CWE-20.pkl")
    bigvul_df = bigvul_df.reset_index(drop=True)
    bigvul_df = bigvul_df.iloc[8200:]
    bigvul_df = bigvul_df.iloc[N*1000:(N+1)*(1000)]

    # Dataset generation for each CWE
    print(f"\nProcessing BigVul dataset")
    print("-----------------------------------------")
    process_dataset(bigvul_df, f"datasets/BigVul/bigvul_CWE-20_input_{N}.csv")
    print(f"Dataset Processing Completed!")
