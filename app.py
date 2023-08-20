import gradio as gr
import copy
import time
import ctypes  # to run on C api directly
import llama_cpp
from llama_cpp import Llama
from huggingface_hub import hf_hub_download  # load from huggingfaces


llm = Llama(
    model_path=hf_hub_download(
        # repo_id="TheBloke/WizardLM-7B-uncensored-GGML",
        repo_id="TheBloke/Llama-2-7B-Chat-GGML",
        # filename="WizardLM-7B-uncensored.ggmlv3.q4_0.bin",
        filename="llama-2-7b-chat.ggmlv3.q5_0.bin",
    ),
    n_ctx=2048,
    n_gpu_layers=50
)  # download model from hf/ n_ctx=2048 for high ccontext length

history = []

pre_prompt = " The user and the AI are having a conversation : <|endoftext|> \n "


def generate_text(input_text, history):

    temp = ""
    if history == []:
        input_text_with_history = (
            f"SYSTEM:{pre_prompt}"
            + "\n"
            + f"USER: {input_text} "
            + "\n"
            + " ASSISTANT:"
        )
    else:
        input_text_with_history = f"{history[-1][1]}" + "\n"
        input_text_with_history += f"USER: {input_text}" + "\n" + " ASSISTANT:"

    output = llm(
        input_text_with_history,
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

    history = ["init", input_text_with_history]


demo = gr.ChatInterface(
    generate_text,
    title="llama-cpp-python on GPU",
    description="Running LLM with https://github.com/abetlen/llama-cpp-python. btw the text streaming thing was the hardest thing to impliment",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)
demo.queue(concurrency_count=1, max_size=5)
demo.launch()