import os
import time
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import gradio as gr
from threading import Thread

MODEL_LIST = ["mistralai/mathstral-7B-v0.1"]
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL = os.environ.get("MODEL_ID")

TITLE = "<h1><center>MathΣtral</center></h1>"

PLACEHOLDER = """
<center>
<p>MathΣtral - I'm MisMath,Your Math advisor</p>
</center>
"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""

device = "cuda" # for GPU usage or "cpu" for CPU usage

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config)

@spaces.GPU()
def stream_chat(
    message: str, 
    history: list,
    system_prompt: str,
    temperature: float = 0.8, 
    max_new_tokens: int = 1024, 
    top_p: float = 1.0, 
    top_k: int = 20, 
    penalty: float = 1.2,
):
    print(f'message: {message}')
    print(f'history: {history}')

    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": answer},
        ])

    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=input_ids, 
        max_new_tokens = max_new_tokens,
        do_sample = False if temperature == 0 else True,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        eos_token_id=[128001,128008,128009],
        streamer=streamer,
    )

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

            
chatbot = gr.Chatbot(height=500, placeholder=PLACEHOLDER)

footer = """
<div style="text-align: center; margin-top: 20px;">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a>
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

with gr.Blocks(css=CSS, theme="Ajaxon6255/Emerald_Isle") as demo:
    gr.HTML(TITLE)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    gr.ChatInterface(
        fn=stream_chat,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Textbox(
                value="You are a helpful assistant in mathematical and scientific tasks",
                label="System Prompt",
                render=False,
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=8192,
                step=1,
                value=1024,
                label="Max new tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="top_p",
                render=False,
            ),
            gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=20,
                label="top_k",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
                label="Repetition penalty",
                render=False,
            ),
        ],
        examples=[
            ["Can you explain the Pythagorean theorem?"],
            ["What is the derivative of sin(x)?"],
            ["Solve the integral of e^(2x) dx."],
            ["How does quantum entanglement work?"],
        ],
        cache_examples=False,
    )
    gr.HTML(footer)


if __name__ == "__main__":
    demo.launch()