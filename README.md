# LogQuantKV
LogQuantKV, a high precision 2-bit Quantization method.

## how to use

### requirements

* torch>=2.3
* transformers>=4.42
* quanto==0.2.0

```bash
pip install -U transformers quanto==0.2.0
```

### implemented methods

* LogQuant: total full precision tokens memory cost will be [3*window_length]
* PartialLogQuant: total full precision tokens memory cost will be [2*window_length]

### use with huggingface models

* example of LogQuant

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.LogQuant import QuantoLogQuantizedCache, LogQuantizedCacheConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen1.5-7B-Chat"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = LogQuantizedCacheConfig(
            backend="quanto",
            nbits=2,
            window_length=4,
            compute_dtype="auto",
            device=device,
        )
cache = QuantoLogQuantizedCache(config)

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

output = model.generate(input_ids, max_new_tokens=128, past_key_values=cache)

print(tokenizer.decode(output[0], skip_special_tokens=True))
'''
system
You are a helpful assistant.
user
如果把脏话都说出来了，那么嘴是不是就干净了
assistant
不，把脏话或不尊重的语言说出口并不会让嘴巴变得干净。恰恰相反，这通常会显示出粗俗、不礼貌或者情绪失控。真正的“干净”是通过言语和行为展现出尊重、理解、宽容和礼貌。即使在压力或冲突的情况下，我们也应该努力控制自己的言辞，用更建设性的方式表达我们的想法。
'''
```

