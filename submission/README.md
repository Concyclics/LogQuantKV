# LogQuantKV

## Evaluation

To see the detail output and result, go to [test_acc/GSM8K](./test_acc/GSM8K/README.md) and [test_acc/LongBench](./test_acc/LongBench/README.md).

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
        "content": "Where is the nearest coffee shop?"
    }]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

output = model.generate(input_ids, max_new_tokens=128, past_key_values=cache)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

