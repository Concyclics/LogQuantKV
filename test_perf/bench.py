# tested on https://github.com/zucchini-nlp/transformers/tree/quant (commit_id 5f3046a)

import os
import argparse
from pathlib import Path
from time import perf_counter

import sys
sys.path.append('../')
from src.LogQuant import (QuantoLogQuantizedCache, LogQuantizedCacheConfig, 
                          QuantoStreamingQuantizedCache, StreamingQuantizedCacheConfig, 
                          QuantoPartialStreamingQuantizedCache, PartialStreamingQuantizedCacheConfig,
                          QuantoPartialLogQuantizedCache, PartialLogQuantizedCacheConfig,
                          QuantoKiViSinkQuantizedCache, KiViSinkQuantizedCacheConfig)

import numpy as np
from matplotlib import pyplot as plt

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

os.environ["TOKENIZERS_PARALLELISM"] = "0"

class TorchTracemalloc():
    track_memory_consumption = []
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        peak = torch.cuda.max_memory_allocated()
        peaked = (peak - self.begin) // 1024 ** 2
        TorchTracemalloc.track_memory_consumption.append(peaked)
        #print(f"peak: {peaked}; reserved: {torch.cuda.max_memory_reserved() // 1024 ** 2}")


@torch.no_grad()
def prefill(model, inputs, cache_implementation, nbits=4):

    if cache_implementation == "quantized":
        config =  LogQuantizedCacheConfig(
            backend="quanto",
            nbits=2,
            window_length=42,
            compute_dtype=torch.bfloat16,
            device="cuda:0",
        )
        past_key_values = QuantoLogQuantizedCache(config)
    else:
        past_key_values = DynamicCache()

    outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    next_input_ids = torch.cat([inputs["input_ids"], next_tokens[:, None]], dim=-1)
    next_model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            inputs,
            is_encoder_decoder=False,
        )
    return next_input_ids, next_model_kwargs


def save_bar_chart(title, x, y, ylabel, xlabel, output_path):
    width = 0.4
    xs = np.arange(len(x))
    plt.bar(xs, height=y, width=width)
    plt.title(title)
    plt.xticks(xs, x)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.savefig(output_path)


def eval_generated_lengths(model, tokenizer, dataset, cache_implementation, nbits, feature, plot_title, output_path):

    # warm up
    generate_kwargs = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
    for _ in range(3):
        inputs_warmup = tokenizer(["Today a dragon flew over Paris"] * 2, return_tensors="pt").to(model.device)
        model.generate(**inputs_warmup, max_new_tokens=20, **generate_kwargs)

    memory_avg, tokens_per_sec_avg = [], []
    time_to_first_token_avg = []
    TTFT, TIME_PER_DECODING = [], []

    # set default values, only one of them will be changing
    parameters = {"max_new_tokens": 500, "batch_size": 1, "input_length": 100} 
    num_batches = 2 # NOTE: 200 samples total only in dataset

    if feature == "batch_size":
        x_iterable = [1, 20, 50, 100, 200]
    else:
        x_iterable = [500, 1000, 4000, 10_000]
    
    for item in x_iterable:
        parameters[feature] = item
        generate_kwargs_curr = generate_kwargs.copy()
        generate_kwargs_curr["min_new_tokens"] = parameters["max_new_tokens"]
        generate_kwargs_curr["max_new_tokens"] = parameters["max_new_tokens"]

        batch_size = parameters["batch_size"]
        with TorchTracemalloc() as tt:
            for batch in range(num_batches):
                start = perf_counter()
                torch.cuda.synchronize()

                # chunk this way since we do not have many data samples
                curr_chunk = dataset[batch: batch+batch_size]
                inputs = tokenizer(
                    curr_chunk['prompt'],
                    padding="max_length",
                    max_length=parameters["input_length"],
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # pre-fill stage
                next_input_ids, next_model_kwargs = prefill(model, inputs, cache_implementation, nbits)
                TTFT.append(perf_counter() - start)
                next_model_kwargs.pop("input_ids")
                torch.cuda.synchronize()
    
                # decoding stage
                out = model.generate(
                    next_input_ids,
                    **next_model_kwargs,
                    **generate_kwargs_curr
                )
                TIME_PER_DECODING.append((perf_counter() - start - TTFT[-1]) / batch_size / parameters["max_new_tokens"])

                del out
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        memory_avg.append(TorchTracemalloc.track_memory_consumption[-1])
        tokens_per_sec_avg.append(1 / (sum(TIME_PER_DECODING) / len(TIME_PER_DECODING)))
        time_to_first_token_avg.append(sum(TTFT) / len(TTFT))

    save_bar_chart(
        title=plot_title,
        x=x_iterable,
        y=memory_avg,
        ylabel=feature,
        xlabel="GPU Memory comsumption in MiB",
        output_path=f"{output_path}/memory.png",
    )

    save_bar_chart(
        title=plot_title,
        x=x_iterable,
        y=tokens_per_sec_avg,
        ylabel=feature,
        xlabel="Tokens per second",
        output_path=f"{output_path}/latency.png",
    )

    print(f"Tokens per sec (avg) - one per condition: {tokens_per_sec_avg}")
    print(f"Time to first token (avg) - one per condition: {tokens_per_sec_avg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_implementation", type=str, default="quantized")
    parser.add_argument("--nbits", type=int, default=4)

    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--dtype", type=str, default="fp16")

    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--feature", type=str, default="batch_size", choices=["batch_size", "input_length", "max_new_tokens"])
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--plot_title", type=str, default="Quantized cache in int4")

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code), padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def collate_fn(example):
        prompt=f"Question: {example['input']}\nContext: {example['context']}\nAnswer:"
        example['prompt'] = prompt
        return example

    dataset = load_dataset('THUDM/LongBench', "samsum", split='test')
    dataset = dataset.map(collate_fn, batched=False)

    eval_generated_lengths(
        model,
        tokenizer,
        dataset,
        cache_implementation=args.cache_implementation,
        nbits=args.nbits,
        feature=args.feature,
        plot_title=args.plot_title,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()