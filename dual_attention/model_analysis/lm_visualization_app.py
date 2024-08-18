"""
This module implements a graphical user interface for generating visualizations of the internal representations of a Dual Attention Transformer Language Model.

You will be prompted to load a model checkpoint from Huggingface Hub, and then you can input a text prompt to generate different visualizations.

To run the app, simply run:

.. code-block::

    python -m dual_attention.model_analysis.lm_visualization_app

"""

import torch
import tiktoken
from tqdm import tqdm
from huggingface_hub import HfApi
import psutil

try:
    import gradio as gr
except ImportError:
    raise ImportError("Please install gradio to use this script")
import tempfile
from pathlib import Path

from .datlm_utils import datlm_forward_w_intermediate_results
from .. import attention_utils

try:
    from bertviz import head_view, model_view
except ImportError:
    raise ImportError("Please install bertviz to use this script")

from ..hf import DualAttnTransformerLM_HFHub

api = HfApi()
models = api.list_models(author='awni00', search='DAT')
models = [model.modelId for model in models] # list of models I have uploaded to the Hugging Face Hub


# Global variables to store the loaded model and tokenizer
loaded_model = None
loaded_model_name = None
tokenizer = tiktoken.get_encoding("gpt2") # TODO: in the future, different models may require different tokenizers
is_model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intermediate_results = None
tokenized_text = None

static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)


def load_model(selected_model_path, progress=gr.Progress(track_tqdm=True)):
    global loaded_model, loaded_model_name, tokenizer, is_model_loaded

    try:
        loaded_model = DualAttnTransformerLM_HFHub.from_pretrained(selected_model_path).to(device)
        loaded_model_name = selected_model_path.split('/')[-1]
        is_model_loaded = True
        load_status = f"Model `{selected_model_path}` loaded successfully. 😁"
        run_forward_button = gr.Button("Run Forward Pass", visible=True)
        head_selection = gr.update(choices=list(range(loaded_model.n_heads_ra)))
        load_button = gr.update(visible=False)
        return load_status, load_button, run_forward_button, head_selection
    except Exception as e:
        load_status = f"Failed to load model '{selected_model_path}' 🥲. Error: {str(e)}"
        return load_status, gr.update(), gr.update(), gr.update()

def run_forward_pass(prompt_text):
    global intermediate_results, tokenized_text
    if loaded_model is None:
        forward_pass_status = "No model loaded yet. Please load a model first."
        generate_viz_button = gr.Button("Generate Visualization", visible=False)
        return forward_pass_status, generate_viz_button

    prompt_tokens = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(device)
    tokenized_text = [tokenizer.decode_single_token_bytes(i).decode('utf-8') for i in prompt_tokens[0]]
    logits, intermediate_results = datlm_forward_w_intermediate_results(loaded_model, prompt_tokens)
    forward_pass_status = "Forward pass computed successfully. 😁"
    generate_viz_button = gr.Button("Generate Visualization", visible=True)
    return forward_pass_status, generate_viz_button

def causal_softmax(scores, temperature=1.0):
    scores = scores / temperature
    bsz, nh, l, _ = scores.size()
    attn_mask = attention_utils.compute_causal_mask(l, device=scores.device)
    attn_mask_ = torch.zeros(l, l, dtype=scores.dtype, device=scores.device).masked_fill(attn_mask.logical_not(), float('-inf'))
    scores = scores + attn_mask_
    scores = torch.nn.functional.softmax(scores, dim=-1)
    return scores

rel_processor_map = {
    "Clip": lambda x: x.clip(0, 1),
    "Sign": lambda x: (x > 0).float(),
    "Sigmoid-Normalize": torch.nn.functional.sigmoid,
    "Softmax-Normalize": causal_softmax
}
def generate_html_visualization(viz_type, view_type, head_selection, rel_processing, rel_scale=1.0):

    if intermediate_results is None or tokenized_text is None:
        return "Please run forward pass first."

    if viz_type == "Self-Attention Attention Scores":
        scores = [x.cpu() for x in intermediate_results['sa_attn_scores']]
    elif viz_type == "Relational-Attention Attention Scores":
        scores = [x.cpu() for x in intermediate_results['ra_attn_scores']]
    elif viz_type == "Relational-Attention Relations":
        rel_processor = rel_processor_map[rel_processing]
        scores = [rel_processor(rels.transpose(-1, 1).cpu() * rel_scale) for rels in intermediate_results['ra_rels']]
    elif viz_type == "Relational-Attention Relations (Scaled by Attention Scores)":
        h = head_selection
        rel_processor = rel_processor_map[rel_processing]
        scores = [rel_processor(rels.transpose(-1, 1).cpu() * rel_scale) * attn[:, h].cpu() for rels, attn in
            zip(intermediate_results['ra_rels'], intermediate_results['ra_attn_scores'])]
    else:
        raise ValueError(f"Invalid visualization type: {viz_type}")

    if view_type == "Head View":
        html_out = head_view(scores, tokenized_text, html_action='return')
    elif view_type == "Model View":
        html_out = model_view(scores, tokenized_text, html_action='return', display_mode="light")
    else:
        raise ValueError(f"Invalid view type: {view_type}")

    return html_out.data

def create_download_link(file_path):
    return f"<a href='file://{file_path}' target='_blank'>Click here to view the generated HTML</a>"

def print_machine_info():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "No GPU available"
    num_cpus = psutil.cpu_count()
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  # Convert bytes to GB

    print(f"CUDA Available: {cuda_available}")
    print(f"GPU: {gpu_name}")
    print(f"Number of CPUs: {num_cpus}")
    print(f"Total Memory: {total_memory:.2f} GB")


def run_app(share=True):
    markdown_header = """# DAT-LM Visualization App
    This app visualizes the internal representations inside of a trained Dual Attention Transformer Language Model.

    With this app, you can visualize: 1) the attention scores in the Self-Attention heads, 2) the attention scores in the Relational-Attention heads, 3) the relations in the Relational-Attention heads, and 4) the relations in the Relational-Attention heads, scaled by the attention scores.

    To use the app, first load a model from the Hugging Face Hub by entering the path to the model in the "Select Model" dropdown and clicking "Load Model". Then, enter a text prompt in the "Input Prompt" textbox and click "Run Forward Pass" to compute the forward pass through the model. Finally, select the type of visualization you want to generate and click "Generate Visualization" to view the visualization.
    """

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(markdown_header)

        model_dropdown = gr.Dropdown(label="Select Model", choices=models, info="Path to model on Hugging Face Hub", allow_custom_value=True)
        model_status = gr.Markdown(label="Model Status",
            value="No model loaded yet. Please enter a path to a model on HF Hub above and click \"Load Model\". "
            "Note that loading a model may take a while since it involves downloading model weights.")

        load_button = gr.Button("Load Model")
        model_dropdown.change(lambda: gr.update(visible=True), outputs=[load_button])

        text_prompt = gr.Textbox(label="Input Prompt", placeholder="Enter your text prompt here...")

        forward_status = gr.Markdown(label="Forward Pass Status", value="Forward pass not computed yet. Load model, enter prompt, then run forward pass.")
        run_forward_button = gr.Button("Run Forward Pass", visible=False)

        viz_type = gr.Dropdown(label="What to visualize?",
            choices=["Self-Attention Attention Scores", "Relational-Attention Attention Scores",
                     "Relational-Attention Relations", "Relational-Attention Relations (Scaled by Attention Scores)"],
            value="Relational-Attention Relations")

        rel_processing = gr.Radio(
            label="Relation Post-Processing", info="Normalize relations to be in (0,1) for visualization.",
            choices=list(rel_processor_map.keys()), value="Clip", visible=True)

        rel_scale = gr.Slider(
            label="Relation Scale Factor", info="Used to scale relations before applying post-processor",
            minimum=0.1, maximum=50., step=0.1, value=5.0, visible=True)

        def viz_type_change_handler(viz_type):
            viz_type_is_rel = "Relations" in viz_type
            return gr.update(visible=viz_type_is_rel), gr.update(visible=viz_type_is_rel)

        viz_type.change(viz_type_change_handler, inputs=[viz_type], outputs=[rel_processing, rel_scale])

        head_selection = gr.Radio(label="Head Selection", value=0, visible=False,
            info="Select a head in relational attention. The attention scores from this head will be used to scale the relations. This can help visualize the relations and the attention criterion simultaneously.")
        def update_head_selection_visiblity(viz_type):
            viz_type == "Relational-Attention Relations (Scaled by Attention Scores)"
            return gr.update(visible=(viz_type == "Relational-Attention Relations (Scaled by Attention Scores)"))
        viz_type.change(update_head_selection_visiblity, inputs=[viz_type], outputs=[head_selection])
        view_type = gr.Dropdown(label="View Type",
            choices=["Head View", "Model View"], value="Head View",
            info="Head view visualizes each layer separately in more detail. Model view visualizes all layers simultaneously at a higher level.")

        generate_viz_button = gr.Button("Generate Visualization", visible=False)

        visualization_file = gr.File(label="HTML Visualization", visible=False)

        html_output = gr.HTML(label="Generated HTML", visible=False)

        def generate_visualization(viz_type, view_type, head_selection, rel_processing, rel_scale):
            html_content = generate_html_visualization(viz_type, view_type, head_selection, rel_processing, rel_scale)

            file_path = static_dir / f"{loaded_model_name}-{viz_type}-{view_type}.html"
            with open(file_path, 'w') as f:
                f.write(html_content)

            visualization_file = gr.update(label="HTML Visualization", value=str(file_path), visible=True)
            return visualization_file#, html_content # create_download_link(file_path)

        load_button.click(load_model, inputs=[model_dropdown], outputs=[model_status, load_button, run_forward_button, head_selection])
        run_forward_button.click(run_forward_pass, inputs=text_prompt, outputs=[forward_status, generate_viz_button])
        generate_viz_button.click(generate_visualization, inputs=[viz_type, view_type, head_selection, rel_processing, rel_scale], outputs=[visualization_file])

    # Launch the app
    demo.launch(share=share)

if __name__ == '__main__':
    print_machine_info()
    run_app()