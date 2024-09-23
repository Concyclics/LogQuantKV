# %%
# %%
#-----------------------------------------------------------------------------
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = "cuda:0"
output_path = "speed_logquant_48.csv"
methods = ["LogQuant"]
#methods = ["LogQuant"]
#methods = ["KiVi"]
n_bit_set = [2]
full_precision_lengths = [48]

import sys
sys.path.append('../')
from src.LogQuant import (QuantoLogQuantizedCache, LogQuantizedCacheConfig, 
                          QuantoStreamingQuantizedCache, StreamingQuantizedCacheConfig, 
                          QuantoPartialStreamingQuantizedCache, PartialStreamingQuantizedCacheConfig,
                          QuantoPartialLogQuantizedCache, PartialLogQuantizedCacheConfig,
                          QuantoKiViSinkQuantizedCache, KiViSinkQuantizedCacheConfig)
#-----------------------------------------------------------------------------

import torch
from torch import nn
data_type = 'auto'
input_length = 512
new_token_length = 8000

# %%
from transformers import Cache, CacheConfig, QuantizedCacheConfig, DynamicCache, QuantizedCache, QuantoQuantizedCache
import torch

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Cache, CacheConfig, QuantizedCacheConfig, QuantizedCache
import torch
from torch import nn

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=data_type,
    device_map=device,
    trust_remote_code=True,
    attn_implementations="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# %%
def to_chat_template(input_text, context):
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": f"fact: {context}\nquestion: {input_text}\n"
    }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



# %%
def get_cache(method, n_bits, dense):
    if method == "KiVi":
        config =  QuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            residual_length=dense,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoQuantizedCache(config)
    elif method == "StreamingQuant":
        config =  StreamingQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=(dense - 4)//2,
            sink_length=4,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoStreamingQuantizedCache(config)
    elif method == "LogQuant":
        config =  LogQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=dense//3,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoLogQuantizedCache(config)
    elif method == "PartialStreamingQuant":
        config =  PartialStreamingQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=int((dense - 4)/1.5),
            sink_length=4,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoPartialStreamingQuantizedCache(config)
    elif method == "PartialLogQuant":
        config =  PartialLogQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=dense//2,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoPartialLogQuantizedCache(config)
    elif method == "KiViSink":
        config =  KiViSinkQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=dense,
            compute_dtype=data_type,
            device=device,
        )
        cache = QuantoKiViSinkQuantizedCache(config)
    else:
        cache = None
    return cache

# %%
def run_with_cache(model, cache, batch_size):
    text = "t" * input_length
    input_ids = tokenizer([text for _ in range(batch_size)], return_tensors="pt").input_ids.to(device)
    length = input_ids.shape[1]
    output_lens = 0
    import time
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        torch.cuda.synchronize()
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        start = time.time()
        if cache is None:
            output = model.generate(input_ids, max_new_tokens=new_token_length, do_sample=False, pad_token_id=tokenizer.eos_token_id, cache_implementation="quantized", cache_config={"backend": "quanto", "nbits": 2})
        else:
            output = model.generate(input_ids, past_key_values=cache, max_new_tokens=new_token_length, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        end = time.time()

    for out in output:
        output_lens += len(out) - length
    used_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

    del input_ids
    del output
    torch.cuda.empty_cache()
    
    return length, output_lens, end - start, used_mem


# %%
import pandas as pd
import tqdm
df = pd.DataFrame(columns=["method", "n_bit", "dense", "batch_size", "input_len", "output_len", "time", "speed", "memory"])
for method in methods:
    if method == "baseline":
        batch_sizes = list(range(1, 4))+list(range(4, 32, 4))+list(range(32, 128, 16))
    else:
        batch_sizes = list(range(1, 4))+list(range(4, 32, 4))+list(range(32, 128, 16))+list(range(128, 385, 32))
    for n_bit in n_bit_set:
        for dense in full_precision_lengths:
            for batch_size in tqdm.tqdm(batch_sizes):
                cache = get_cache(method, n_bit, dense)
                length, output_lens, time, mem = run_with_cache(model, cache, batch_size)
                df = pd.concat([df, pd.DataFrame([[method, n_bit, dense, batch_size, length, output_lens, time, output_lens/time, mem]], columns=df.columns)], ignore_index=True)
                df.to_csv(output_path, index=False)
                torch.cuda.empty_cache()


