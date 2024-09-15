# %%
#-----------------------------------------------------------------------------
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = "cuda:0"
output_path = "KiVi-4-bit-llama31.csv"
#methods = ["baseline", "KiVi", "StreamingQuant", "LogQuant", "PartialStreamingQuant", "PartialLogQuant"]
#methods = ["baseline", "KiVi", "StreamingQuant", "LogQuant", "PartialStreamingQuant", "PartialLogQuant", "KiViSink"]
methods = ["KiVi"]
#methods = ["PartialLogQuant"]
n_bit_set = [4]
full_precision_lengths = [128]

import sys
sys.path.append('../../')
from src.LogQuant import (QuantoLogQuantizedCache, LogQuantizedCacheConfig, 
                          QuantoStreamingQuantizedCache, StreamingQuantizedCacheConfig, 
                          QuantoPartialStreamingQuantizedCache, PartialStreamingQuantizedCacheConfig,
                          QuantoPartialLogQuantizedCache, PartialLogQuantizedCacheConfig,
                          QuantoKiViSinkQuantizedCache, KiViSinkQuantizedCacheConfig)
#-----------------------------------------------------------------------------

import torch
from torch import nn
data_type = 'auto'
new_token_length = 1024

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main")

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
def to_chat_template(text):
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user", 
        "content": f"Please answer the following question by the format, mark the final answer by ####.\n\
        example 1: \n\
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72\n\
        example 2: \n\
        question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\
        answer: Weng earns $12/60 minutes = $<<12/60=0.2>>0.2 per minute. Weng earned $0.2*50 minutes = $<<0.2*50=10>>10. #### 10\n\
        example 3: \n\
        question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\n\
        answer: He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624\n\
        example 4: \n\
        question: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\n\
        answer: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n\
        example 5: \n\
        question: John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?\n\
        answer: He wants to write 3*400=<<3*400=1200>>1200 pages So it will take him 1200/20=<<1200/20=60>>60 days #### 60\n\
        Question for you: \n\
        question: {text}\n"
    }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_with_cache(model, input_ids, cache):
    # Run the model
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    output = model.generate(input_ids, do_sample=False, past_key_values=cache, pad_token_id=tokenizer.eos_token_id, max_new_tokens=new_token_length)
    return output

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

def get_bits_per_token(length, dense, n_bits, method):
    if method == "KiVi":
        bits_per_token = (n_bits * (length - dense) + 16 * dense) / length
    elif method == "StreamingQuant":
        window_length = (dense - 4)//2
        bits_per_token = (n_bits * (length - 2 * window_length - 4) + 16 * (2 * window_length + 4)) / length
    elif method == "LogQuant":
        window_length = dense//3
        bits_per_token = (n_bits * (length - 3 * window_length) + 16 * 3 * window_length) / length
    elif method == "PartialStreamingQuant":
        window_length = int((dense - 4)/1.5)
        bits_per_token = (n_bits * (length - 1.5 * window_length - 4) + 16 * (1.5 * window_length + 4)) / length
    elif method == "PartialLogQuant":
        window_length = dense//2
        bits_per_token = (n_bits * (length - 2 * window_length) + 16 * 2 * window_length) / length
    elif method == "KiViSink":
        window_length = dense
        bits_per_token = (n_bits * (length - window_length - 2) + 16 * window_length + 32) / length
    else:
        bits_per_token = 16
    return bits_per_token

def manage_output(output, ground_truth_answer, chat_template):
    length = output.shape[-1]
    model_output = tokenizer.decode(output[0], skip_special_tokens=True)
    model_output = model_output[len(chat_template):]
    if "####" not in model_output:
        print(model_output)
        model_output_answer = 'None'
    else:
        model_output_answer = model_output.split("####")[-1].strip()
        model_output_answer = ''.join(filter(str.isdigit, model_output_answer))  
    if ground_truth_answer == model_output_answer:
        accuracy = 1
    else:
        accuracy = 0
    return model_output, model_output_answer, accuracy, length

import pandas as pd
import numpy as np
from tqdm import trange

df = pd.DataFrame(columns=["question", "qid", "ground_truth", "ground_truth_answer", "model_output", "model_output_answer", "method", "accuracy", "bit_per_token", "total_length", "model_name"])
#df = pd.read_csv(output_path)

#for i in trange(10):#len(ds['test'])):
for i in trange(len(ds['test'])):
    question = ds['test'][i]['question']
    ground_truth = ds['test'][i]['answer']
    ground_truth_answer = ds['test'][i]['answer'].split("####")[-1].strip()
    chat_template = to_chat_template(question)
    input_ids = tokenizer(chat_template, return_tensors="pt").input_ids.to(device)
    if "baseline" in methods:
        output = model.generate(input_ids, do_sample=False, pad_token_id=tokenizer.eos_token_id, max_new_tokens=new_token_length)
        model_output, model_output_answer, accuracy, length = manage_output(output, ground_truth_answer, chat_template)

        df = pd.concat([df, pd.DataFrame([[question, i, ground_truth, ground_truth_answer, model_output, model_output_answer, "baseline", accuracy, 16, length, model_name]],
                                     columns=["question", "qid", "ground_truth", "ground_truth_answer", "model_output", "model_output_answer", "method", "accuracy", "bit_per_token", "total_length", "model_name"])])
        del output
        torch.cuda.empty_cache()
        
    for method in methods:
        if method == "baseline":
            continue
        for n_bits in n_bit_set:
            for dense in full_precision_lengths:
                cache = get_cache(method, n_bits, dense)
                output = run_with_cache(model, input_ids, cache)
                model_output, model_output_answer, accuracy, length = manage_output(output, ground_truth_answer, chat_template)
                bits_per_token = get_bits_per_token(length, dense, n_bits, method)
                df = pd.concat([df, pd.DataFrame([[question, i, ground_truth, ground_truth_answer, model_output, model_output_answer, f"{method}_{n_bits}_{dense}", accuracy, bits_per_token, length, model_name]],
                                                columns=["question", "qid", "ground_truth", "ground_truth_answer", "model_output", "model_output_answer", "method", "accuracy", "bit_per_token", "total_length", "model_name"])])
                del cache, output
                torch.cuda.empty_cache()

                df.to_csv(output_path, index=False)



    
        

    



