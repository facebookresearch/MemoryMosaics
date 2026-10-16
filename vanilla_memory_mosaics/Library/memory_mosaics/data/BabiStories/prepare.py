# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import numpy as np
import os
from tqdm import tqdm
import tiktoken
import json
    

f = open('traindataset.txt', 'r')
traindata = f.readlines()
f = open('valdataset.txt', 'r')
valdata = f.readlines()

split_dataset = {}
split_dataset['train'] = traindata
split_dataset['val'] = valdata 

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


for split in ['val','train']:
    dataset = split_dataset[split]
    idx = []
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    for story in tqdm(dataset): 
        idx.extend(enc.encode_ordinary(json.loads(story)))
        idx.extend([enc.eot_token])

    arr_len = len(idx)
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    arr[:] = np.array(idx)
    arr.flush()
    print(len(arr))
    


