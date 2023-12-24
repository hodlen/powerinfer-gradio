import os
import gradio as gr
import copy
import time
import llama_cpp
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import pandas as pd


llm = Llama(
    model_path=os.environ.get("MODEL_PATH"),
    path_idx=os.environ.get("IDX_PATH"),
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
file_path = 'sensitive_words.xlsx'
df = pd.read_excel(file_path, usecols=[0, 1, 2])
sensitive_words = df.values.flatten()
#sensitive_words = set(word.lower() for word in sensitive_words if pd.notna(word))
sensitive_words = set(str(word).lower() for word in sensitive_words if pd.notna(word))

def check_sensitive_words(text, sensitive_words):
    words = text.split()
    for word in words:
        if word.lower() in sensitive_words:
            return True
    return False

def generate_text(message, history):
    input_prompt = "" # f"System: {system_message}\n"
    '''
    for user_msg, assistant_msg in history:
        input_prompt += f"Question: {user_msg}\n"
        input_prompt += f"Answer: {assistant_msg}\n"
    '''
    input_prompt += f"Question: {message}\n"
    if len(input_prompt) > 128:
            input_prompt = input_prompt[: 128]
    input_prompt += f"Answer: "

    if check_sensitive_words(input_prompt, sensitive_words) == True:
        return
    print('input_prompt:\n' + input_prompt)
    if (len(input_prompt) > 256):
        print('input prompt too long')
        return 


    output = llm(
        input_prompt,
        temperature=0.3,
        top_p=0.1,
        top_k=20,
        repeat_penalty=1.1,
        max_tokens=128,
        stop=[
            "<|endoftext|>",
            "Question: ",
            "Answer: ",
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
    title="<a href=https://github.com/SJTU-IPADS/PowerInfer>PowerInfer</a>&nbsp; with &nbsp;<a href=https://huggingface.co/SparseLLM/ReluFalcon-40B>Falcon-40B(ReLU)-FP16</a>&nbsp; on a single RTX 4090",
    description=(
	"Paper: <a href=https://arxiv.org/abs/2312.12456>PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU</a>.<br>"
        "<sub>Note: This system utilizes 8 CPU threads and one NVIDIA RTX 4090.</sub><br>"
	    "<sub>Note: To maintain Gradio stability, the maximum input length has been set to 128 tokens, and the output length should be less than 128 tokens as well.</sub><br>"
        "<sub>Note: Currently, we only support single-round testing. The prompt template is: <br>`Question: Your input prompt. Answer: Model output.`</sub><br>"
        "<sub>Source Code: <a href=https://github.com/hodlen/powerinfer-gradio/tree/main>hodlen/powerinfer-gradio</a></sub><br>"
        "<hr>**Notice: AI Model Limitations**. This model, re-trained on a dataset of 5 billion tokens, has not been fine-tuned, aligned, or subjected to comprehensive safety checks. It may produce responses that are inaccurate, biased, or objectionable. The model is provided without any warranties or guarantees, express or implied.<br>"
        '**Terms of Use**. By using this AI model demo, you acknowledge its provision "as is" and without warranties. You accept that we are not liable for any damages arising from its use. You agree to use the model lawfully and at your own risk, acknowledging its limitations.'
    ),
    examples=["Tell me a story about the middle earth", "What is the theory of relativity?"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
    concurrency_limit=1,
)
demo.queue(max_size=32)
demo.launch(share=False, root_path="/gradio")
