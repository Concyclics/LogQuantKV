import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.LogQuant import QuantoPartialStreamingQuantizedCache, PartialStreamingQuantizedCacheConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config =  PartialStreamingQuantizedCacheConfig(
            backend="quanto",
            nbits=2,
            window_length=4,
            sink_length=2,
            compute_dtype="auto",
            device=device,
        )
cache = QuantoPartialStreamingQuantizedCache(config)

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

output = model.generate(input_ids, max_new_tokens=256, past_key_values=cache)

print(tokenizer.decode(output[0], skip_special_tokens=True))