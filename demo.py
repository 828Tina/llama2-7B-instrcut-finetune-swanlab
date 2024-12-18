import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TOTAL_GPU_NUMS = 4
TOKENIZE_PATH = "./model/LLM-Research/llama-2-7b"
MODEL_LIST = {
    "office-llama2-7b": "model/LLM-Research/llama-2-7b",  # 官方模型
    "alpaca_llama2-7b_lora": "./output/llama2-7b-alpaca-en-52k-epoch-3-merge-model",  # cot微调
}

model_names = list(MODEL_LIST.keys())  # Ensure this is a list to easily index


# 推理函数
def llama_inference(model_path: str, prompt: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    # 将输入文本转换为模型的输入格式
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 推理过程
    with torch.no_grad():
        # 生成输出，调整参数以控制生成长度
        output = model.generate(
            inputs['input_ids'],
            max_length=2048,  # 设置最大生成长度
            num_return_sequences=1,  # 生成一个序列
            no_repeat_ngram_size=2,  # 防止重复的n-gram
            top_p=0.95,  # nucleus sampling
            temperature=0.7  # 控制输出的随机性
        )

    # 解码并打印生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# 更新 generate_response 函数，使用 llama_inference 进行推理
def generate_response(instruct_text, input_text):
    prompt = instruct_text+input_text

    # 获取所有模型的输出，指定每个模型使用不同的 GPU
    outputs = [
        llama_inference(MODEL_LIST[model_name], prompt, device=f"cuda:{i % TOTAL_GPU_NUMS}")
        for i, model_name in enumerate(model_names)
    ]

    return tuple(outputs)


# 创建 Gradio 界面
demo = gr.Interface(
    fn=generate_response,  # 函数名
    inputs=[
        gr.Textbox(label="instruction"),
        gr.Textbox(label="input"),
    ],  # 输入文本框
    outputs=[gr.Textbox(label=model_name) for model_name in model_names],
)

if __name__ == "__main__":
    demo.launch(share=True)
