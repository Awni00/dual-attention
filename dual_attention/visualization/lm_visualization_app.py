"""
This module implements a graphical user interface for generating visualizations of the internal representations of a Dual Attention Transformer Language Model.

You will be prompted to load a model checkpoint from Huggingface Hub, and then you can input a text prompt to generate different visualizations.

To run the app, simply run:
```bash
python -m dual_attention.visualization.lm_visualization_app
```
"""

import torch
import tiktoken
from tqdm import tqdm
import gradio as gr
import tempfile
from pathlib import Path

from .datlm_utils import datlm_forward_w_intermediate_results

try:
    from bertviz import head_view, model_view
except ImportError:
    raise ImportError("Please install bertviz to use this script")

from ..hf import DualAttnTransformerLM_HFHub

# A dictionary of available models
models = [
    "awni00/DAT-sa8-ra8-ns1024-sh8-nkvh4-343M",
]

# Global variables to store the loaded model and tokenizer
loaded_model = None
tokenizer = tiktoken.get_encoding("gpt2") # TODO: in the future, different models may require different tokenizers
is_model_loaded = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intermediate_results = None
tokenized_text = None

static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)


def load_model(selected_model_path):
    global loaded_model, tokenizer, is_model_loaded

    try:
        loaded_model = DualAttnTransformerLM_HFHub.from_pretrained(selected_model_path).to(device)
        is_model_loaded = True
        return f"Model `{selected_model_path}` loaded successfully. 😁"
    except Exception as e:
        return f"Failed to load model '{selected_model_path}' 🥲. Error: {str(e)}"

def compute_intermediate_results(prompt_text):
    global intermediate_results, tokenized_text
    if loaded_model is None:
        return "No model loaded yet. Please load a model first."

    prompt_tokens = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(device)
    tokenized_text = [tokenizer.decode_single_token_bytes(i).decode('utf-8') for i in prompt_tokens[0]]
    logits, intermediate_results = datlm_forward_w_intermediate_results(loaded_model, prompt_tokens)
    return "Forward pass computed successfully. 😁"

def generate_visualization(viz_type, view_type):

    if intermediate_results is None or tokenized_text is None:
        return "Please run forward pass first."

    if viz_type == "SA Attention Scores":
        scores = [x.cpu() for x in intermediate_results['sa_attn_scores']]
    elif viz_type == "RA Attention Scores":
        scores = [x.cpu() for x in intermediate_results['ra_attn_scores']]
    elif viz_type == "RA Relations":
        scores = [rels.transpose(-1, 1).cpu() for rels in intermediate_results['ra_rels']]
    else:
        raise ValueError(f"Invalid visualization type: {viz_type}")

    if view_type == "Head View":
        html_out = head_view(scores, tokenized_text, html_action='return')
    elif view_type == "Model View":
        html_out = model_view(scores, tokenized_text, html_action='return', display_mode="light")
    else:
        raise ValueError(f"Invalid view type: {view_type}")

    return html_out.data

def serve_html(viz_type, view_type):
    # Generate HTML content
    html_content = generate_visualization(viz_type, view_type)

    file_path = static_dir / "visualization.html"
    with open(file_path, 'w') as f:
        f.write(html_content)

    # # Create a temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
    #     temp_file.write(html_content.encode('utf-8'))
    #     temp_file_path = temp_file.name

    # Return the path to the file, which can be used to create a download link
    # return temp_file_path
    return file_path

# Function to create the download link
def create_download_link(file_path):
    return f"<a href='file://{file_path}' target='_blank'>Click here to view the generated HTML</a>"


# Function to return HTML for Gradio app
# def serve_html(text_prompt):
#     return generate_visualization(text_prompt)

if __name__ == '__main__':
    # Gradio Interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# DAT-LM Visualization App")

        model_dropdown = gr.Dropdown(label="Select Model", choices=models, info="Path to model on Hugging Face Hub", allow_custom_value=True)
        model_status = gr.Markdown(label="Model Status", value="No model loaded yet.")

        load_button = gr.Button("Load Model")

        # Text prompt input
        text_prompt = gr.Textbox(label="Input Prompt", placeholder="Enter your text prompt here...")

        # Status of forward pass
        forward_status = gr.Markdown(label="Forward Pass Status", value="Forward pass not computed yet. 😐")
        # Button to run forward pass
        run_forward_button = gr.Button("Run Forward Pass")

        # Type of Visualization to generate
        viz_type = gr.Dropdown(label="What to visualize?", choices=["SA Attention Scores", "RA Attention Scores", "RA Relations"], value="RA Relations")
        view_type = gr.Dropdown(label="View Type", choices=["Head View", "Model View"], value="Head View")

        # Button to generate and serve HTML
        generate_button = gr.Button("Generate HTML")

        # HTML output
        html_output = gr.HTML(label="Generated HTML")

        def generate_link(viz_type, view_type):
            file_path = serve_html(viz_type, view_type)
            return create_download_link(file_path)

        # Connect the button click to the function
        load_button.click(load_model, inputs=[model_dropdown], outputs=model_status)
        run_forward_button.click(compute_intermediate_results, inputs=text_prompt, outputs=forward_status)
        generate_button.click(generate_link, inputs=[viz_type, view_type], outputs=html_output)

    # Launch the app
    demo.launch(share=True)
