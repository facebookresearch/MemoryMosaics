# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Interactive chat/generation CLI for every model family.

Usage:
    python -m inference.chat --model_path /path/to/checkpoint
"""

import os
import sys

import torch

from inference.generator import Generator
from model import build_model, config_from_dict
from tokenizer.chat_template import build_generation_prompt
from tokenizer.tokenizer import Tokenizer
from training.checkpoint import load_checkpoint
from training.distributed import TORCH_DTYPE_MAP, _apply_blockwise_compile


def load_model(checkpoint_dir: str, model_dtype: str = "bf16", compile: bool = True,
               compile_fullgraph: bool = True):
    """(model, tokenizer) from a training checkpoint, in eval mode.

    The same model serves evaluation and generation: a forward pass without a cache
    computes exactly as in training (minus the training-only random long-memory skip),
    one with a cache generates. compile=True compiles each block; models containing a
    DeltaNet compile with fullgraph=False regardless (its kernel is not traceable).
    """
    import json
    from pathlib import Path

    with open(Path(checkpoint_dir) / "params.json") as f:
        params = json.load(f)
    model_type = params.get("model_type", "mosaics")
    model_config = config_from_dict(model_type, params.get("model", {}))

    tokenizer = Tokenizer(params.get("data", {}).get("tokenizer_path"))

    dtype = TORCH_DTYPE_MAP[model_dtype]
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    model = build_model(model_type, model_config).to(device="cuda", dtype=dtype)
    model.init_weights()        # after the cast: it rebuilds the (complex) RoPE tables
    if compile:
        _apply_blockwise_compile(model, fullgraph=compile_fullgraph)
    load_checkpoint(checkpoint_dir, model)
    model.eval()
    return model, tokenizer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "plain"],
                        help="chat: the SFT conversation format (tokenizer.chat_template); "
                             "plain: continue the text typed so far")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, compile=True, compile_fullgraph=True)

    generator = Generator(
        model=model,
        model_dtype=next(model.parameters()).dtype,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
    )

    generator.generate([[tokenizer.bos_id]], max_gen_len=8)
    os.system("clear")

    print("### Memory Mosaics ###")
    print("Type your message. Commands: <exit>, <restart>")
    print("\033[1m[User:] \033[0m")

    messages: list[dict] = []      # chat mode: the conversation so far
    text = ""                      # plain mode: the text so far
    line = sys.stdin.readline().strip()

    while True:
        if line == "<exit>":
            print("Bye!")
            break
        elif line == "<restart>":
            messages, text = [], ""
            print("Restarted. Type your message.")
            print("\033[1m[User:] \033[0m")
            line = sys.stdin.readline().strip()
            continue

        if args.mode == "chat":
            messages.append({"role": "user", "content": line})
            prompt = build_generation_prompt(messages, tokenizer)
        else:
            text += line
            prompt = tokenizer.encode(text, bos=True)

        print("\033[1m[MemoryMosaics:] \033[0m")
        generated: list[int] = []
        shown = ""
        for tok in generator.generate_stream(
            prompt_tokens=[prompt],
            max_prompt_len=len(prompt),
            max_gen_len=args.max_gen_len,
            use_sampling=True,
            temperature=args.temperature,
            top_k=args.top_k,
            attn_bias_type=model.attn_bias_type,
        ):
            if tok.item() == tokenizer.eos_id:      # the end of the answer
                break
            generated.append(tok.item())
            # Decode the whole answer, not token by token: SentencePiece pieces carry
            # their leading space, which a single-token decode drops.
            full = tokenizer.decode(generated)
            print(f"\033[32m{full[len(shown):]}\033[0m", end="", flush=True)
            shown = full

        print(f"\n\n[generated {len(generated)} tokens]\n")
        print("\033[1m[User:] \033[0m")

        if args.mode == "chat":
            messages.append({"role": "assistant", "content": shown})
        else:
            text += shown
        line = sys.stdin.readline().strip()


if __name__ == "__main__":
    main()
