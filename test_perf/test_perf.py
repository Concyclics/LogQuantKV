# %%
# %%
#-----------------------------------------------------------------------------
model_name = "Qwen/Qwen1.5-0.5B-Chat-AWQ"
device = "cuda:0"
output_path = "speed.csv"
methods = ["baseline", "KiVi", "LogQuant", "PartialLogQuant"]
methods = ["KiVi"]
n_bit_set = [2]
full_precision_lengths = [128]

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
new_token_length = 4096

from datasets import load_dataset
ds = load_dataset("THUDM/LongBench", "2wikimqa_e", split="test")

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
    trust_remote_code=True
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
def run_with_cache(model, qid, cache, batch_size):
    chat_template = to_chat_template(ds[qid]["input"], ds[qid]["context"])
    input_ids = tokenizer([chat_template for _ in range(batch_size)], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    length = input_ids.shape[1]
    output_lens = 0
    import time
    start = time.time()
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    output = model.generate(input_ids, past_key_values=cache, pad_token_id=tokenizer.eos_token_id, max_new_tokens=new_token_length, do_sample=False)
    end = time.time()
    for out in output:
        output_lens += len(out)

    del input_ids
    del output
    
    return length, output_lens, end - start


# %%
import pandas as pd
import tqdm
df = pd.DataFrame(columns=["method", "n_bit", "dense", "batch_size", "qid", "input_len", "output_len", "time", "speed"])
for qid in tqdm.tqdm(range(10)):
    for method in methods:
        if method == "baseline":
            batch_sizes = [1, 2, 4, 8, 12, 16]
        else:
            batch_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128]
        for n_bit in n_bit_set:
            for dense in full_precision_lengths:
                for batch_size in batch_sizes:
                    cache = get_cache(method, n_bit, dense)
                    length, output_lens, time = run_with_cache(model, qid, cache, batch_size)
                    df = pd.concat([df, pd.DataFrame([[method, n_bit, dense, batch_size, qid, length, output_lens, time, output_lens/time]], columns=df.columns)], ignore_index=True)
                    df.to_csv(output_path, index=False)


