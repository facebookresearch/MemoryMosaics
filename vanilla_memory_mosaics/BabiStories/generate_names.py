# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
from tqdm import tqdm

import json 
import os 
import math 
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

#torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()

# general 
parser.add_argument('--max_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--top_p', type=float, default=1, help='top_p')
parser.add_argument('--top_k', type=int, default=0, help='top_k, 0 disables top_k.')
parser.add_argument('--temperature', type=float, default=1, help='temperature, high = random.')
parser.add_argument('--idx', type=str, default='0',help='idx')
parser.add_argument('--prompt', type=str, default="Please give me 100 different first names regardless of gender?")
parser.add_argument('--out_dir', type=str, default='data/')
args = parser.parse_args()

if not os.path.exists(os.path.join(args.out_dir,f'names{args.idx}.txt')):

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, pad_token_id=tokenizer.eos_token_id)


    f = open(os.path.join(args.out_dir,f'names{args.idx}.txt'), 'w')

    text = f"[INST] {args.prompt} [/INST]"
    inputs = tokenizer([text]*args.max_batch_size, return_tensors="pt").to(0)
    for i in range(20):
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, pad_token_id=tokenizer.eos_token_id)
        for j in  range(len(outputs)):
            output = tokenizer.decode(outputs[j], skip_special_tokens=True)[len(text):]
            f.write(output + '\n')
    f.close()

