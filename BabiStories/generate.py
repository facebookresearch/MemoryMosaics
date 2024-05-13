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
import numpy as np 
parser = argparse.ArgumentParser()

# general 
parser.add_argument('--max_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--top_p', type=int, default=1, help='top_p')
parser.add_argument('--top_k', type=int, default=0, help='top_k, 0 disables top_k.')
parser.add_argument('--temperature', type=int, default=1, help='temperature, high = random.')
parser.add_argument('--data_idx', type=int, default=0, help='data index')
parser.add_argument('--sub_data_idx', type=int, default=0, help='sub data idx')
parser.add_argument('--out_dir', type=str, default='data/')
parser.add_argument('--metaprompt_dir', type=str, default='metaprompt/')
args = parser.parse_args()


with torch.no_grad():
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
    
    f = open(os.path.join(args.out_dir, 'allnames.txt'),'r')
    allnames = f.readlines()
    allnames = [var.strip() for var in allnames]
    f.close()

    f = open(os.path.join(args.out_dir, 'allstartings.txt'),'r')
    allstartings = f.readlines()
    allstartings = [var.strip() for var in allstartings]
    f.close() 

    max_batch_size=args.max_batch_size
    data_idx = args.data_idx
    mark = '0%d'  % data_idx if data_idx < 10 else  '%d' % data_idx 
    with open(os.path.join(args.metaprompt_dir, f'data{mark}_{args.sub_data_idx}.json'), 'r') as f:
        data = json.load(f)
    alldata = data

    master_process = True
    try: 
        start_iter = int(np.loadtxt(os.path.join( args.out_dir, f'{mark}_{args.sub_data_idx}.log')))
    except:
        start_iter = 0 
    print('start_iter', start_iter)


    for storyid in tqdm(range(start_iter, math.ceil(len(alldata) / max_batch_size))):
        stories = []
        names   = []
        starts  = []
        prompts  = []
        for story in alldata[storyid*max_batch_size:(storyid+1)*max_batch_size]: 
            prompt = story['instruction']['prompt:']
            split = "the verb"
            name = allnames[np.random.randint(len(allnames))]
            start = allstartings[np.random.randint(len(allstartings))]
            prompt = prompt.split(split)
            if len(prompt) == 2:
                prompt = prompt[0] + f"the character name \"{name}\", the verb" + prompt[1]
            stories.append(f'[INST] {prompt} [/INST] {start}')
            names.append(name)
            starts.append(start)
            prompts.append(prompt)
        if master_process:
            f = open(os.path.join( args.out_dir, f'{mark}_{args.sub_data_idx}.txt'), 'a+')   
            promptf = open(os.path.join( args.out_dir, f'prompt{mark}_{args.sub_data_idx}.txt'),'a+')

        inputs =  tokenizer(stories, return_tensors="pt", padding=True).to(0) # pad at the start. so fine. 
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, pad_token_id=tokenizer.eos_token_id)

        if master_process:
            generated_stories = []
            for i, output in enumerate(outputs):
                story = tokenizer.decode(output, skip_special_tokens=True)
                generated_stories.append(json.dumps(story)+'\n')
            
            metainfo = []  
            for i in range(len(outputs)):
                metainfo.append(json.dumps({"name":names[i],"starting":starts[i],"prompt":prompts[i]})+'\n')

            f.writelines(generated_stories)
            f.close()
            promptf.writelines(metainfo)
            promptf.close()
            start_iter=storyid+1
            np.savetxt(os.path.join( args.out_dir, f'{mark}_{args.sub_data_idx}.log'), np.array([start_iter]))
            
