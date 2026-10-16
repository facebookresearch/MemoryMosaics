# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""MetaMathQA as SFT data: one {"messages": [user, assistant]} conversation per line.

Downloads meta-math/MetaMathQA (~395k math word problems with worked answers ending in
"The answer is: X") and writes <output>/train.jsonl, optionally a seeded random subset.

Usage:
  # The data of configs/sft/metamath.yaml: 262,144 examples, seed 42
  python -m tools.prepare_metamath_sft_data --num_example 262144 --output data_out/metamath_sft_262k/train

  # All examples
  python -m tools.prepare_metamath_sft_data --output data_out/metamath_sft/train
"""

import argparse
import json
import random
from pathlib import Path


def _to_example(item: dict) -> dict | None:
    """Row -> SFT messages dict, or None if the question or the answer is empty."""
    query, response = item["query"].strip(), item["response"].strip()
    if not query or not response:
        return None
    return {"messages": [{"role": "user", "content": query},
                         {"role": "assistant", "content": response}]}


def prepare(output_path: str, num_example: int = 0, seed: int = 42) -> int:
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    examples = [ex for ex in map(_to_example, ds) if ex is not None]
    if 0 < num_example < len(examples):
        examples = random.Random(seed).sample(examples, num_example)
    elif num_example > len(examples):
        print(f"Warning: --num_example {num_example} but only {len(examples)} examples; writing all.")
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Output directory (train.jsonl is written there)")
    parser.add_argument("--num_example", type=int, default=0,
                        help="Keep a random subset of this many examples; 0 = all")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the random subset")
    args = parser.parse_args()
    count = prepare(args.output, num_example=args.num_example, seed=args.seed)
    print(f"Wrote {count} examples -> {args.output}/train.jsonl")


if __name__ == "__main__":
    main()
