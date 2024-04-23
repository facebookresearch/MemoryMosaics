# saves the TinyStories dataset to a binary file for training. following was helpful:
# https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
# The code is modified from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py


import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 1

# takes ~2GB in huggingface .cache dir
split_dataset = load_dataset("JeanKaddour/minipile")
split_dataset["val"] = split_dataset.pop("test")
split_dataset.pop('val')

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")


def process(example):
    # print(example['text'])
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids, "len": len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=["text"],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

eot_token = enc.eot_token
# concatenate all the ids in each dataset into one large file we can use for training
cur_idx = 0
for split, dset in tokenized.items():
    arr_len = np.sum(dset["len"])
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    #idxfilename = os.path.join(os.path.dirname(__file__), f"{split}.idx.bin")

    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    #idxarr = np.memmap(idxfilename, dtype=np.int64, mode="w+", shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        idxarr_batch = []
        for var in arr_batch:
            idxarr_batch.append(cur_idx)
            if var == eot_token:
                cur_idx += 1
        idxarr_batch = np.array(idxarr_batch)

        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        #idxarr[idx : idx + len(idxarr_batch)] = idxarr_batch
        idx += len(arr_batch)
    arr.flush()
    #idxarr.flush()

# train.bin is ~905MB, val.bin ~9.1MB


# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
