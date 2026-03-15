import os
import re
import io
import sys
import torch
import subprocess
import numpy as np
import gradio as gr
import pandas as pd
from PIL import Image
import networkx as nx
from io import BytesIO
import random
from graphviz import Source
from pygments import highlight
from pygments.token import Token
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import colormaps as cm
from pygments.lexers import PythonLexer
from pygments.formatter import Formatter
from matplotlib.colors import Normalize

# Add project absolute path
sys.path.append(os.path.abspath(""))

from explainer import IlluminatiExplainer
from devign.src.utils.objects.cpg.function import Function
from devign.devign import Devign

# Add Graphviz dot path to env
conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
os.environ["GRAPHVIZ_DOT"] = os.path.join(conda_env_path, "bin", "dot")
os.environ["PATH"] += os.pathsep + os.path.join(conda_env_path, "bin")

# Increase the pixels limit
Image.MAX_IMAGE_PIXELS = 250_000_000 

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

try:
    DEVICE = torch.device(f"cuda:{select_best_gpu()}") if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(input_path: str) -> pd.DataFrame:
    dataset = pd.read_pickle(input_path)
    dataset.info(memory_usage='deep')
    return dataset

# Explainer Params
EXPLAINER = {
    "mode": 2,
    "node": 1,
    "node_rate": 10,
    "synchronize": 0,
    "epochs": 10,
    "lr": 5e-2,
    "agg1": "max",
    "agg2": "",
    "sample": 128
}

MODEL_PATH = 'benchmarks/devign_Accuracy_50_50.pt'

RANDOM_SEED = 42

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# VS Code color palette mapping
VS_CODE_COLORS = {
    Token.Keyword: "#569CD6",
    Token.String: "#D69D85",
    Token.Comment: "#6A9955",
    Token.Name.Function: "#DCDCAA",
    Token.Operator: "#C586C0",
    Token.Number: "#B5CEA8",
}

class VSCodeHTMLFormatter(Formatter):
    def __init__(self):
        super().__init__()
        self.styles = {token: f"color: {color};" for token, color in VS_CODE_COLORS.items()}

    def format(self, tokensource, outfile):
        result = []
        for token, text in tokensource:
            style = self.styles.get(token, "")
            result.append(f'<span style="{style}">{text}</span>')
        outfile.write("".join(result))

def order_nodes(nodes, max_nodes):
    nodes_by_column = sorted(nodes.items(), key=lambda n: int(n[1].get_column_number()))
    nodes_by_line = sorted(nodes_by_column, key=lambda n: int(n[1].get_line_number()))
    for i, node in enumerate(nodes_by_line):
        node[1].order = i
    if len(nodes) > max_nodes:
        nodes_by_line = nodes_by_line[:max_nodes]
    nodes_by_line_map = {}
    for node in nodes_by_line:
        line = node[1].get_line_number()
        code = node[1].get_code()
        try:
            nodes_by_line_map[line].append(code)
        except KeyError:
            nodes_by_line_map[line] = [code]
    return OrderedDict(nodes_by_line), nodes_by_line_map

def filter_nodes(nodes):
    return {n_id: node for n_id, node in nodes.items() if node.has_code() and
            node.has_line_number() and node.label not in ["Comment", "Unknown"]}

def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    for function in cpg["functions"]:
        func = Function(function)
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)
        ordered_nodes, nodes_by_line_map = order_nodes(nodes, max_nodes)
    return ordered_nodes, nodes_by_line_map

def syntax_highlight(code):
    formatter = VSCodeHTMLFormatter()
    highlighted_code = highlight(code, PythonLexer(), formatter)
    return f"<pre style='font-family: monospace; background-color: #FFFFFF; color: #D4D4D4; padding: 10px;'>{highlighted_code}</pre>"

def get_node_names(nodes_by_line_map):
    nodes = []
    for line in nodes_by_line_map.values():
        for value in line:
            nodes.append(value)
    return nodes

def plot_graph_with_masks(data, node_names, node_mask, edge_mask):
    edge_index = data.edge_index
    # non_zero_nodes = [i for i in range(x.shape[0]) if not torch.all(x[i] == 0.0)]
    non_zero_nodes = np.nonzero(node_mask.numpy())[0]
    print(f" - Non-zero Nodes: {len(non_zero_nodes)}")
    edge_index = edge_index.cpu()
    filtered_edges = [(u, v) for u, v in zip(edge_index[0].numpy(), edge_index[1].numpy())
                      if u in non_zero_nodes and v in non_zero_nodes]
    G = nx.Graph()
    G.add_nodes_from(non_zero_nodes)
    G.add_edges_from(filtered_edges)
    node_labels = {node: f"{node}:{node_names[node]}" for node in non_zero_nodes}
    node_sizes = [300 + 700 * node_mask[i].item() for i in non_zero_nodes]
    node_colors = [node_mask[i].item() for i in non_zero_nodes]
    edge_weights = [edge_mask[idx].item() for idx, (u, v) in enumerate(filtered_edges)]
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_weights = [3 * (w / max_weight) for w in edge_weights]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(20, 14))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.coolwarm,
        edgecolors="black",
        vmin=0,
        vmax=1
    )
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")
    cbar = plt.colorbar(nodes)
    cbar.set_label("Node Importance")
    plt.title("Graph Visualization with Node and Edge Importance")
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer

def generate_ast(ast_data):
    dot = "digraph AST {\n"
    dot += '    node [shape="box", style="filled", color="#cccccc", fontname="Courier"];\n'
    dot += '    edge [fontname="Courier", fontsize="10"];\n'
    for node in ast_data:
        node_id = node['id']
        label = f"{node_id}\\n"
        for prop in node['properties']:
            label += f"{prop['key']}: {prop['value']}\\n"
        dot += f'    "{node_id}" [label="{label}"];\n'
    for node in ast_data:
        for edge in node.get('edges', []):
            dot += f'    "{edge["out"]}" -> "{edge["in"]}" [label="{edge["id"]}"];\n'
    dot += "}\n"
    source = Source(dot)
    img_bytes = source.pipe(format='png')
    return Image.open(io.BytesIO(img_bytes))

def generate_cfg(cfg_data):
    dot = "digraph CFG {\n"
    dot += '    node [shape="ellipse", style="filled", color="#d9e8f5", fontname="Courier"];\n'
    dot += '    edge [fontname="Courier", fontsize="10"];\n'
    for node in cfg_data:
        node_id = node['id']
        label = f"{node_id}\\n"
        for prop in node['properties']:
            label += f"{prop['key']}: {prop['value']}\\n"
        dot += f'    "{node_id}" [label="{label}"];\n'
    for node in cfg_data:
        for edge in node.get('edges', []):
            dot += f'    "{edge["out"]}" -> "{edge["in"]}" [label="{edge["id"]}"];\n'
    dot += "}\n"
    source = Source(dot)
    img_bytes = source.pipe(format='png')
    return Image.open(io.BytesIO(img_bytes))

def get_color_from_score(score, min_score, max_score):
    norm = Normalize(vmin=min_score, vmax=max_score)
    cmap = cm.get_cmap("coolwarm")
    rgba = cmap(norm(score))
    return rgba

def replace_leading_spaces_with_nbsp(sentence):
    match = re.match(r"^[ \t]*", sentence)
    leading_spaces = match.group(0)
    nbsp_replacement = '&nbsp;' * 4 * len(leading_spaces)
    result = nbsp_replacement + sentence[len(leading_spaces):]
    return result

def highlight_code_with_scores(nodes_and_masks_by_line_map, raw_func):
    """
    For each line of source code (raw_func), replace tokens with colored spans
    (without individual tooltips) and, if nodes exist on that line, aggregate
    their information into one tooltip that is displayed when the line is hovered.
    """
    lines = raw_func.split("\n")
    highlighted_lines = []
    
    # Gather all node scores to compute the global min and max.
    all_scores = [score for token_list in nodes_and_masks_by_line_map.values()
                  for _, score in token_list]
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
    else:
        min_score, max_score = 0, 1

    node = 0
    # Process each line.
    for i, line in enumerate(lines):
        line_number = str(i+1)
        line = line.replace('    ', '\t')
        aggregate_tooltip = ""
        # If there are any nodes for this line.
        if line_number in nodes_and_masks_by_line_map:
            tokens_info = nodes_and_masks_by_line_map[line_number]
            # Replace each token occurrence (only once per token in the order provided).
            nodes_info = []
            for token_info in tokens_info:
                token_text, score = token_info
                nodes_info.append((node, token_text, score))
                # Append the token information for the aggregated tooltip.
                aggregate_tooltip += f"Node {node}: {token_text} (Score: {score:.2f})\n"
                node += 1
            # Sort in ascending order based on the score (second element)
            sorted_nodes_by_score = sorted(nodes_info, key=lambda token: token[1])
            for node, token_text, score in nodes_info:
                if score:
                    color = get_color_from_score(score, min_score, max_score)
                    rgba_color = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})"
                    # Replace token_text (first occurrence) with a colored span.
                    line = re.sub(re.escape(token_text),
                                f'<span style="background-color:{rgba_color};">{token_text}</span>',
                                line, count=1)
            # Wrap the entire line in a container that includes the aggregated tooltip.
            highlighted_line = (
                f'<div class="line-with-tooltip">'
                f'{line}'
                f'<span class="tooltiptext">{aggregate_tooltip.strip()}</span>'
                f'</div>'
            )
        else:
            highlighted_line = f"<div>{line}</div>"
        highlighted_line = highlighted_line.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
        highlighted_lines.append(highlighted_line)
    return "\n".join(highlighted_lines)

def map_nodes_and_masks_by_line(nodes_by_line_map, node_mask):
    result = {}
    idx = 0
    for line, items in nodes_by_line_map.items():
        mapped_items = []
        for item in items:
            mapped_items.append((item, node_mask[idx].item()))
            idx += 1
        result[line] = mapped_items
    return result

def generate_counterexample_source(data):
    """ Function to generate an counterexample version of the source code. """
    if data is None:
        return ""
    raw_function = data.func
    ce_function = "# CounterExample\n" + raw_function.replace("def ", "def adv_")
    
    return ce_function

# --------------
# App Callbacks
# --------------

# Randomize now takes the current tab label as input.
def randomize(current_tab):

    # Original or Counterexample example:
    adv = True if current_tab.strip() == "Counterexample Source Code" else False
    orig_example = dataset_df[dataset_df['adv'] == False].sample(n=1).iloc[0]
    adv_example = dataset_df[(dataset_df["id"] == orig_example.id) & (dataset_df["adv"] == True)].iloc[0]
    id = orig_example.id
    example = adv_example if adv else orig_example
    details = f"- **Label:** {'Vulnerable' if example.target else 'Benign'} \n- **Id:** {id} \n- **Counterexample:** {adv}"
    data = {}
    data["Selected"] = example
    
    return orig_example.func, adv_example.func, details, id, data

def clear_input():
    return "", "", "", "", None

def submit_code(data):
    data = data["Selected"]
    graph = data.input
    raw_func = data.func
    cpg = data.cpg
    ordered_nodes, nodes_by_line_map = parse_to_nodes(cpg, 205)
    node_names = get_node_names(nodes_by_line_map)
    label = int(data.target)

    feat_mask, edge_mask, node_mask = explainer.explain_graph(
        graph, loss_fc=None, node=False, synchronize=True)
    
    node_mask_image = explainer.plot_node_scores_distribution(node_mask)
    graph_image_bytes = plot_graph_with_masks(graph, node_names, node_mask, edge_mask)
    graph_image = Image.open(io.BytesIO(graph_image_bytes.getvalue()))
    # Create a node mask mapping
    nodes_and_masks_by_line_map = map_nodes_and_masks_by_line(nodes_by_line_map, node_mask)
    # Enhance function code with color coding
    highlighted_code = highlight_code_with_scores(nodes_and_masks_by_line_map, raw_func)
    
    # ----------------------------------------------
    # Method 1: Minimal Subgraph by Greedy Addition
    # ----------------------------------------------
    # This function gradually adds nodes (sorted by importance) until
    # the subgraph prediction equals the original prediction.
    subgraph_mask_add, selected_nodes, conf_add, ep_add = explainer.minimal_subgraph_by_adding(graph, node_mask, model)
    subgraph_add_image_bytes = plot_graph_with_masks(graph, node_names, subgraph_mask_add, edge_mask)
    subgraph_add_image = Image.open(io.BytesIO(subgraph_add_image_bytes.getvalue()))

    # ----------------------------------------------
    # Method 2: Minimal Subgraph by Greedy Removal
    # ----------------------------------------------
    # This function starts with the full graph and removes the least important nodes
    # one by one while checking that the prediction remains unchanged.
    subgraph_mask_rem, remaining_nodes, conf_rem, ep_rem = explainer.minimal_subgraph_by_removal(graph, node_mask, model)
    subgraph_rem_image_bytes = plot_graph_with_masks(graph, node_names, subgraph_mask_rem, edge_mask)
    subgraph_rem_image = Image.open(io.BytesIO(subgraph_rem_image_bytes.getvalue()))

    # ----------------------------------------------
    # Method 3: Optimal Subgraph
    # ----------------------------------------------
    # This function finds the combination of nodes that provides the best confidence score
    # while checking that the prediction remains unchanged.
    subgraph_mask_opt, opt_nodes, conf_opt, ep_opt = explainer.optimal_minimal_subgraph(graph, node_mask, model, threshold=0.5)
    subgraph_opt_image_bytes = plot_graph_with_masks(graph, node_names, subgraph_mask_opt, edge_mask)
    subgraph_opt_image = Image.open(io.BytesIO(subgraph_opt_image_bytes.getvalue()))

    # ---------------------------
    # Method 4: Counterfactual Explanation
    # ---------------------------
    # This new method finds a minimal change that flips the prediction.
    # cf_submask, changed_nodes, cf_confidence, cf_prediction = explainer.counterfactual_graph_explanation(graph, node_mask, model, threshold=0.5)
    # cf_image_bytes = plot_graph_with_masks(graph, node_names, cf_submask, edge_mask)
    # cf_image = Image.open(io.BytesIO(cf_image_bytes.getvalue()))

    model.eval()
    prediction_confidence = float(model(graph.to(DEVICE)))
    predicted_label = "Vulnerable" if prediction_confidence > 0.5 else "Benign"
    true_label_text = "Vulnerable" if label == 1 else "Benign"
    is_correct = predicted_label == true_label_text
    prediction_box_class = "correct-prediction" if is_correct else "incorrect-prediction"
    return (
        graph_image,
        gr.update(elem_id='prediction_text', value=predicted_label, elem_classes=prediction_box_class),
        # prediction_confidence,
        highlighted_code, 
        node_mask_image,
        subgraph_add_image,
        subgraph_rem_image,
        subgraph_opt_image,
        # cf_image
    )

def show_graph_type(choice, data):
    data = data["Selected"]
    cpg = data.cpg
    if choice == "AST":
        ast = cpg['functions'][0]['AST']
        try:
            ast_image = generate_ast(ast)
        except Exception as e:
            print(e)
            ast_image = None
        return gr.update(value=ast_image, label="AST Image",visible=True)
    elif choice == "CFG":
        cfg = cpg['functions'][0]['CFG']
        try:
            cfg_image = generate_cfg(cfg)
        except Exception as e:
            print(e)
            cfg_image = None
        return gr.update(value=cfg_image, label="CFG Image", visible=True)
    else:
        return gr.update(visible=False)
    
def select_example_by_id(id, current_tab):
    try:
        id = int(id)
        adv = True if current_tab.strip() == "Counterexample Source Code" else False
        orig_example = dataset_df[(dataset_df["id"] == id) & (dataset_df["adv"] == False)].iloc[0]
        adv_example = dataset_df[(dataset_df["id"] == id) & (dataset_df["adv"] == True)].iloc[0]
        example = adv_example if adv else orig_example
        details = f"- **Label:** {'Vulnerable' if example.target else 'Benign'} \n- **Id:** {id} \n- **Counterexample:** {adv}"
        data = {}
        data["Selected"] = example
        
        return orig_example.func, adv_example.func, details, id, data
    
    except IndexError:
        raise gr.Error(f"Example with ID {id} not found", duration=5)

# --------------
# CSS Styles
# --------------
custom_css = """
#prediction_text.correct-prediction textarea {
    background-color: lightgreen;
}
#prediction_text.incorrect-prediction textarea {
    background-color: lightcoral;
}

/* Styles for aggregated tooltip on each line */
.line-with-tooltip {
  position: relative;
  display: block;
}

.line-with-tooltip .tooltiptext {
  visibility: hidden;
  width: 400px;               /* Increase width as needed */
  background-color: #444;     /* Background color different from black */
  color: #fff;
  text-align: left;
  padding: 5px;
  border-radius: 5px;
  position: absolute;
  z-index: 1;
  bottom: 100%;               /* Position above the line */
  left: 0;                    
  margin-bottom: 5px;
  opacity: 0;
  transition: opacity 0.3s;
  white-space: pre-wrap;      /* Wrap multiple lines */
  font-size: 0.9em;
}

.line-with-tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
"""

# -------------------------
# Devign Model Initialization
# -------------------------
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
model.load(MODEL_PATH)
model.to(DEVICE)

explainer = IlluminatiExplainer(
    model,
    agg1=EXPLAINER.get("agg1"),
    agg2=EXPLAINER.get("agg2"),
    lr=EXPLAINER.get("lr"),
    epochs=EXPLAINER.get("epochs"),
    device=DEVICE
)

dataset_df = load_dataset("datasets/cwe20cfa/cwe20cfa_CWE-20_augmented_input_test.pkl")

# -------------------------
# Gradio Interface
# -------------------------

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("# VISION Visualization Inferface")
    with gr.Row():
        with gr.Column():
            with gr.Tabs() as tabs:
                with gr.TabItem("Original Source Code") as orig_code_tab:
                    orig_input_code_box = gr.Textbox(lines=10, max_lines=15, label="Input Source Code", interactive=False)                    
                with gr.TabItem("Counterexample Source Code") as adv_code_tab:
                    adv_input_code_box = gr.Textbox(lines=10, max_lines=15, label="Input Source Code", interactive=False)
            details_box = gr.Markdown(value="", label="Additional Details")
            with gr.Row():
                id_box = gr.Textbox(label="Example ID", interactive=True)
            with gr.Row():
                random_button = gr.Button("Random")
                select_button = gr.Button("Select")
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
        with gr.Column():
            with gr.Row():
                prediction_details = gr.Textbox(
                    lines=1, 
                    label="Prediction Details", 
                    elem_id="prediction_text",
                    interactive=False
                )
                # prediction_score = gr.Number(label="Model Confidence Score", interactive=False, visible=False)
            with gr.Tab("Source Code"):
                with gr.Row():
                    source_code_output = gr.HTML(label="Source Code Visualization")
            with gr.Tab("Node Graph"):
                with gr.Row():
                    graph_image = gr.Image(label="Node Graph Visualization")
                with gr.Row():
                    node_distribution_image = gr.Image(label="Node Score Distribution")
            with gr.Tab("Subgraph Explanation"):
                subgraph_add_image = gr.Image(label="Positive Subgraph (Add)")
                subgraph_rem_image = gr.Image(label="Negative Subgraph (Remove)")
                subgraph_opt_image = gr.Image(label="Optimal Subgraph")
            # with gr.Tab("Counterfactual Analysis"):
            #     cf_image = gr.Image(label="Counterfactual Graph")
            with gr.Row():
                cpg_dropdown = gr.Dropdown(
                    choices=["None", "AST", "CFG"],
                    label="Choose Graph Type", 
                    interactive=True
                )
            with gr.Row():
                cpg_image = gr.Image(visible=False)
    
    # Create state components.
    data_state = gr.State()  # Holds the current example (original or ce).
    current_tab_state = gr.State(value=orig_code_tab.label)         

    # Bind callbacks.
    random_button.click(
        randomize, 
        inputs=[current_tab_state],
        outputs=[orig_input_code_box, adv_input_code_box, details_box, id_box, data_state]
    )
    clear_button.click(
        clear_input, 
        outputs=[orig_input_code_box, adv_input_code_box, details_box, id_box, data_state]
    )
    select_button.click(
        select_example_by_id,
        inputs=[id_box, current_tab_state],
        outputs=[orig_input_code_box, adv_input_code_box, details_box, id_box, data_state]
    )
    submit_button.click(
        submit_code, 
        inputs=data_state,
        outputs=[graph_image, prediction_details, source_code_output, node_distribution_image, subgraph_add_image, subgraph_rem_image, subgraph_opt_image]
    )
    cpg_dropdown.change(
        show_graph_type, 
        inputs=[cpg_dropdown, data_state],
        outputs=[cpg_image]
    )
    # Attach select event for tabs.
    orig_code_tab.select(lambda: "Original Source Code", inputs=[], outputs=current_tab_state)
    adv_code_tab.select(lambda: "Counterexample Source Code", inputs=[], outputs=current_tab_state)
    
    # Optionally, trigger the select_example_by_id callback when the current_tab_state changes.
    current_tab_state.change(
        select_example_by_id,
        inputs=[id_box, current_tab_state],
        outputs=[orig_input_code_box, adv_input_code_box, details_box, id_box, data_state]
    )

interface.launch()