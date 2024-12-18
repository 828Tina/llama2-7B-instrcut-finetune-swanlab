import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def llama_inference(model_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda:0")

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



if __name__ == "__main__":
    model_path = "./model/LLM-Research/llama-2-7b"
    merge_path = "./output/llama2-7b-alpaca-en-52k-epoch-3-merge-model"
    input_texts = "Describe a time when you had to make a difficult decision."

    print(llama_inference(model_path, input_texts))
    print(llama_inference(merge_path, input_texts))