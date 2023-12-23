import os
import gradio as gr
import copy
import time
import llama_cpp
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


llm = Llama(
    model_path=os.environ.get("MODEL_PATH"),
    idx_path=os.environ.get("IDX_PATH"),
    n_gpu_layers=63,
    n_ctx=512,
    n_batch=1,
    n_threads=8,
    verbose=True,
)

history = []

system_message = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_text(message, history):
    input_prompt = "" # f"System: {system_message}\n"
    for user_msg, assistant_msg in history:
        input_prompt += f"User: {user_msg}\n"
        input_prompt += f"Assistant: {assistant_msg}\n"

    input_prompt += f"User: {message}\n"
    input_prompt += f"Assistant: "

    print('input_prompt:\n' + input_prompt)

    output = llm(
        input_prompt,
        temperature=0.15,
        top_p=0.1,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=512,
        stop=[
            "<|endoftext|>",
            "Assistant: ",
            "User: ",
            "System: ",
        ],
        stream=True,
    )
    temp = ""
    for out in output:
        stream = copy.deepcopy(out)
        temp += stream["choices"][0]["text"]
        yield temp

    print('response: ', temp)



demo = gr.ChatInterface(
    generate_text,
    title="PowerInfer with Falcon-40B(ReLU) on 4090",
    description="Running LLM with <a href=https://github.com/SJTU-IPADS/PowerInfer>PowerInfer</a> on Consumer-grade GPUs",
    examples=["Tell me a story about the middle earth", "Tell a fairy tale about PowerInfer"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
    concurrency_limit=4,
)
demo.queue(max_size=20)
demo.launch(share=False)
