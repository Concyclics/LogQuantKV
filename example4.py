import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.LogQuant import QuantoPartialLogQuantizedCache, PartialLogQuantizedCacheConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = PartialLogQuantizedCacheConfig(
            backend="quanto",
            nbits=2,
            window_length=16,
            compute_dtype="auto",
            device=device,
        )
cache = QuantoPartialLogQuantizedCache(config)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user", 
        "content": "如果把脏话都说出来了，那么嘴是不是就干净了"
    }]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

output = model.generate(input_ids, max_new_tokens=128, do_sample=False, past_key_values=cache)

print(tokenizer.decode(output[0], skip_special_tokens=True))