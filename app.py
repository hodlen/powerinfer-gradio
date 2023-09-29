import os
import gradio as gr
import copy
import time
import llama_cpp
from llama_cpp import Llama
from huggingface_hub import hf_hub_download  


llm = Llama(
    model_path=hf_hub_download(
        repo_id=os.environ.get("REPO_ID", "TheBloke/Llama-2-7b-Chat-GGUF"),
        filename=os.environ.get("MODEL_FILE", "llama-2-7b-chat.Q5_0.gguf"),
    ),
    n_ctx=2048,
    n_gpu_layers=50, # change n_gpu_layers if you have more or less VRAM 
) 

history = []

system_message = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""


def generate_text(message, history):
    temp = ""
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in history:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "

    output = llm(
        input_prompt,
        temperature=0.15,
        top_p=0.1,
        top_k=40, 
        repeat_penalty=1.1,
        max_tokens=1024,
        stop=[
            "<|prompter|>",
            "<|endoftext|>",
            "<|endoftext|> \n",
            "ASSISTANT:",
            "USER:",
            "SYSTEM:",
        ],
        stream=True,
    )
    for out in output:
        stream = copy.deepcopy(out)
        temp += stream["choices"][0]["text"]
        yield temp

    history = ["init", input_prompt]


demo = gr.ChatInterface(
    generate_text,
    title="llama-cpp-python on GPU",
    description="Running LLM with https://github.com/abetlen/llama-cpp-python",
    examples=["tell me everything about llamas"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)
demo.queue(concurrency_count=1, max_size=5)
demo.launch()