"""
This module implements a graphical user interface for generating text using a Dual Attention Transformer Language Model.

You will be prompted to load a model checkpoint from Huggingface Hub, and then you can input a text prompt to generate text.

To run the app, simply run:
```bash
python -m dual_attention.visualization.lm_inference_app
```
"""

import torch
import tiktoken
from tqdm import tqdm
import gradio as gr

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

def load_model(selected_model_path):
    global loaded_model, tokenizer, is_model_loaded

    try:
        loaded_model = DualAttnTransformerLM_HFHub.from_pretrained(selected_model_path).to(device)
        is_model_loaded = True
        return f"Model `{selected_model_path}` loaded successfully. 😁"
    except Exception as e:
        return f"Failed to load model '{selected_model_path}' 🥲. Error: {str(e)}"


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    use_tqdm=False):
    """
    Generate max_new_tokens new tokens, conditioning on the input idx.

    Parameters
    ----------
    idx : Tensor[int]
        tensor of shape (batch_size, seq_len) with input tokens.
    max_new_tokens : int
        number of new tokens to generate
    temperature : float, optional
        temperature parameter of softmax, by default 1.0
    top_k : int, optional
        top-k sampling parameter, by default None

    Returns
    -------
    Tensor[int]
        tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
    """

    iterator = tqdm(range(max_new_tokens)) if use_tqdm else range(max_new_tokens)
    for _ in iterator:
        # crop the sequence if it is longer thanblock_size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, model.block_size:]
        logits, _ = model(idx_cond) # forward pass
        logits = logits[:, -1, :] / temperature # scale by temperature

        # optionally, crop logits to top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
        idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
        idx = torch.cat((idx, idx_next), dim=1) # append to sequence

    return idx

def generate_text(prompt_text, max_new_tokens, temperature, top_k):
    if not is_model_loaded:
        return "Please load a model first."

    top_k = int(top_k)
    max_new_tokens = int(max_new_tokens)

    prompt_tokens = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(device)
    generated_tokens = generate(loaded_model, prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_tqdm=True)
    return tokenizer.decode(generated_tokens[0].cpu().numpy())

if __name__ == '__main__':

    # Gradio Interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Dual Attention Language Model Inference App")

        with gr.Row():
            with gr.Column():
                # Left column: Model selection and input controls
                model_dropdown = gr.Dropdown(label="Select Model", choices=models, info="Path to model on Hugging Face Hub", allow_custom_value=True)
                model_status = gr.Markdown(label="Model Status", value="No model loaded yet.")

                load_button = gr.Button("Load Model")

                # Text prompt input
                text_prompt = gr.Textbox(label="Input Prompt", placeholder="Enter your prompt here...", lines=5)

                # Inputs for temperature, top-k, and number of tokens to generate
                temperature_input = gr.Slider(0.01, 1.0, label="Temperature", value=0.95)
                top_k_input = gr.Number(label="Top-K", value=50, precision=0)
                num_tokens_input = gr.Number(label="Number of tokens to generate", value=100, precision=0)

                # Generate button
                generate_button = gr.Button("Generate")

            with gr.Column():
                # Right column: Generated output
                output_text = gr.Textbox(label="Generated Text", interactive=False, lines=20,
                    placeholder="Generated text will appear here...", show_copy_button=True)

        # Function bindings
        load_button.click(load_model, inputs=[model_dropdown], outputs=model_status)
        generate_button.click(generate_text, inputs=[text_prompt, num_tokens_input, temperature_input, top_k_input], outputs=output_text)

    # Launch the app
    demo.launch(share=True)