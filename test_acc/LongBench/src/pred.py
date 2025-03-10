import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
#----------------------------------------------
import sys
sys.path.append('../../../')
from src.LogQuant import (QuantoLogQuantizedCache, LogQuantizedCacheConfig, 
                          QuantoStreamingQuantizedCache, StreamingQuantizedCacheConfig, 
                          QuantoPartialStreamingQuantizedCache, PartialStreamingQuantizedCacheConfig,
                          QuantoPartialLogQuantizedCache, PartialLogQuantizedCacheConfig)
from transformers import Cache, CacheConfig, QuantizedCacheConfig, DynamicCache, QuantizedCache, QuantoQuantizedCache
device = "cuda" if torch.cuda.is_available() else "cpu"
#----------------------------------------------

def get_cache(method, n_bits, dense):
    if method == "KiVi":
        config =  QuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            residual_length=dense,
            compute_dtype='auto',
            device=device,
        )
        cache = QuantoQuantizedCache(config)
    elif method == "StreamingQuant":
        config =  StreamingQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=(dense - 4)//2,
            sink_length=4,
            compute_dtype='auto',
            device=device,
        )
        cache = QuantoStreamingQuantizedCache(config)
    elif method == "LogQuant":
        config =  LogQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=dense//3,
            compute_dtype='auto',
            device=device,
        )
        cache = QuantoLogQuantizedCache(config)
    elif method == "PartialStreamingQuant":
        config =  PartialStreamingQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=int((dense - 4)/1.5),
            sink_length=4,
            compute_dtype='auto',
            device=device,
        )
        cache = QuantoPartialStreamingQuantizedCache(config)
    elif method == "PartialLogQuant":
        config =  PartialLogQuantizedCacheConfig(
            backend="quanto",
            nbits=n_bits,
            window_length=dense//2,
            compute_dtype='auto',
            device=device,
        )
        cache = QuantoPartialLogQuantizedCache(config)
    else:
        cache = None
    return cache

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--cache', type=str, default="baseline", choices=["baseline", "KiVi", "StreamingQuant", "LogQuant", "PartialStreamingQuant", "PartialLogQuant"])
    parser.add_argument('--n_bit', type=int, default=2, choices=[2, 4])
    parser.add_argument('--window_length', type=int, default=128)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_inputs(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    else:
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)[0]
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, cache_method, n_bits, dense):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        cache = get_cache(method, n_bits, dense)
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                inputs = prompt.to(device)
        else:
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                past_key_values=cache,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=cache,
            )
        pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        print(f"Rank {rank} finished!")
        print(f"Lenght of pred: {len(pred)}")
        del cache
        del pred
        del output
        del inputs
        torch.cuda.empty_cache()
    del model
    del tokenizer
    torch.cuda.empty_cache()
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype='auto').to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype='auto').to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype='auto').to(device)

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = 1

    method = args.cache
    n_bits = args.n_bit
    dense = args.window_length

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            if not os.path.exists(f"pred_e/{model_name}/{method}_{n_bits}_{dense}"):
                os.makedirs(f"pred_e/{model_name}/{method}_{n_bits}_{dense}")
            out_path = f"pred_e/{model_name}/{f'{method}_{n_bits}_{dense}'}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if not os.path.exists(f"pred/{model_name}/{method}_{n_bits}_{dense}"):
                os.makedirs(f"pred/{model_name}/{method}_{n_bits}_{dense}")
            out_path = f"pred/{model_name}/{f'{method}_{n_bits}_{dense}'}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        for rank in range(world_size):
            get_pred(rank, world_size, data_subsets[rank], max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, method, n_bits, dense)
            
