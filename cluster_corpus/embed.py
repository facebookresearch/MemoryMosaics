# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Stage 2 — embed each normalized shard into a matching .npy.

Reads `normalized/<source>/shard_NNNNN.jsonl` (stage 1) and writes
`embeddings/<source>/shard_NNNNN.npy` -- one .npy per .jsonl, same name, row i of the
.npy is line i of the .jsonl.

    python -m cluster_corpus.embed --jsonl_dir normalized/cosmopedia \\
        --output_dir embeddings/cosmopedia --prompt "clustering: "

That embeds every file in one process. This is the expensive stage and it is
embarrassingly parallel: start one process per GPU with --worker_id g --n_workers G and
worker g takes files g, g+G, g+2G, ... loading the model once. Which worker embeds a
file does not change its output, so the worker count can differ between runs; a file
whose .npy exists is skipped, so an interrupted run just resumes.

DOCUMENT ORDER IS A CONTRACT. The global stream is every *.jsonl in sorted-filename
order, then line order within each file (blank / unparseable lines skipped, as stage 4
does). Because .npy names mirror .jsonl names, sorted .npy files concatenate to exactly
that order: an embedding's row index IS its document index. Re-sharding stage 1 after
embedding is not safe -- re-embed.

Defaults: bf16, max_seq_length 2048. `embed_config.json` records the settings (model,
prompt, dtype, length) and every later run in the same --output_dir must match them, so
two embedding spaces can never be mixed in one directory.
"""

import argparse
import glob
import json
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_CONFIG = "embed_config.json"   # settings every shard in the directory shares
DTYPES = ("bf16", "fp16", "fp32")


def read_jsonl(path: str, content_field: str) -> list[str]:
    """One stage-1 shard's documents, in line order.

    Blank and unparseable lines are skipped -- exactly how stage 4 enumerates them, so
    row i here is the i-th document stage 4 will pair with label i.
    """
    texts = []
    with open(path) as fh:
        for line in fh:
            if not (line := line.strip()):
                continue
            try:
                # parse_int=str dodges Python's int<->str digit limit.
                texts.append(json.loads(line, parse_int=str).get(content_field, ""))
            except (json.JSONDecodeError, ValueError):
                continue
    return texts


def load_model(model_name: str, max_seq_length: int, dtype: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"loading {model_name} ({dtype}, max_seq_length={max_seq_length}) ...")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.to({"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype])
    model.max_seq_length = max_seq_length
    return model


class Encoder:
    """Encode with batch-size backoff on CUDA OOM. A halved batch size sticks for the
    rest of the process, so later files don't re-hit the same OOM."""

    def __init__(self, model, batch_size: int, prompt: str | None):
        self.model, self.batch_size = model, batch_size
        self.extra = {"prompt": prompt} if prompt is not None else {}
        self.dim = model.get_sentence_embedding_dimension()

    def __call__(self, texts: list[str]) -> np.ndarray:
        import torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        while self.batch_size >= 1:
            try:
                emb = self.model.encode(texts, batch_size=self.batch_size,
                                        show_progress_bar=True, convert_to_numpy=True,
                                        normalize_embeddings=True, **self.extra)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.batch_size //= 2
                logger.warning(f"OOM; retrying with batch_size={self.batch_size}")
        else:
            raise RuntimeError("OOM even at batch_size=1")

        # The model normalized in its own dtype; bf16 keeps ~3 significant digits, so
        # norms are off 1 by ~1e-3. Renormalize in fp32 -- k-means and the inner-product
        # index assume unit vectors. Zero rows stay zero and NaN rows stay NaN.
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, norms, out=emb, where=norms > 0)
        return emb


def check_embed_config(output_dir: str, cfg: dict) -> None:
    """Every shard in a directory must come from the same settings.

    Shards are skipped once they exist, so without this a change of model, dtype or
    length would splice two embedding spaces together.
    """
    path = os.path.join(output_dir, EMBED_CONFIG)
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        if (diff := sorted(k for k in cfg.keys() | old.keys() if cfg.get(k) != old.get(k))):
            raise SystemExit(f"{path} has {({k: old.get(k) for k in diff})}, this run "
                             f"{({k: cfg.get(k) for k in diff})}. Use another --output_dir, "
                             f"or delete its shard_*.npy and {EMBED_CONFIG} to re-embed.")
        return
    if glob.glob(os.path.join(output_dir, "shard_*.npy")):
        raise SystemExit(f"{output_dir} has shards but no {EMBED_CONFIG}, so their settings "
                         f"are unknown. Delete them (or use another --output_dir).")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)          # concurrent workers all write the same content


def npy_name(jsonl_path: str) -> str:
    return os.path.basename(jsonl_path)[:-len(".jsonl")] + ".npy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl_dir", required=True, help="normalized/<source>")
    ap.add_argument("--output_dir", required=True, help="embeddings/<source>")
    ap.add_argument("--worker_id", type=int, default=0,
                    help="This process takes files worker_id, worker_id+n_workers, ...")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="Processes splitting the files (e.g. one per GPU). Does not "
                         "affect the output.")
    ap.add_argument("--model_name", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--prompt", default=None,
                    help="Prefix prepended to every document. nomic-embed conditions on "
                         "a task prefix -- pass 'clustering: ' for this pipeline.")
    ap.add_argument("--batch_size", type=int, default=256, help="Halved on OOM.")
    ap.add_argument("--max_seq_length", type=int, default=2048,
                    help="Tokens kept per document (nomic-embed's training length). "
                         "Cost grows fast with length; a document's topic is set early.")
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--content_field", default="text")
    args = ap.parse_args()
    if not 0 <= args.worker_id < args.n_workers:
        raise SystemExit(f"--worker_id must be in [0, {args.n_workers})")

    files = sorted(glob.glob(os.path.join(args.jsonl_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl found in {args.jsonl_dir}"
                         + (" (only .tmp files: stage 1 has not finished -- re-run it)"
                            if glob.glob(os.path.join(args.jsonl_dir, "*.jsonl.tmp")) else ""))
    os.makedirs(args.output_dir, exist_ok=True)
    check_embed_config(args.output_dir, {
        "model_name": args.model_name, "prompt": args.prompt, "dtype": args.dtype,
        "max_seq_length": args.max_seq_length, "content_field": args.content_field})

    mine = files[args.worker_id::args.n_workers]
    todo = [f for f in mine if not os.path.exists(os.path.join(args.output_dir, npy_name(f)))]
    logger.info(f"worker {args.worker_id}/{args.n_workers}: {len(mine)} of {len(files)} "
                f"file(s), {len(mine) - len(todo)} already done, {len(todo)} to embed")
    if not todo:
        return

    encode = Encoder(load_model(args.model_name, args.max_seq_length, args.dtype, args.device),
                     args.batch_size, args.prompt)
    t0, n_docs = time.time(), 0
    for i, path in enumerate(todo, 1):
        out = os.path.join(args.output_dir, npy_name(path))
        if os.path.exists(out):                  # another worker finished it meanwhile
            continue
        texts = read_jsonl(path, args.content_field)
        emb = encode(texts)
        # Non-finite rows mean the model failed numerically on some document. Say so
        # here, not two stages later when k-means produces nonsense.
        if (n_bad := int((~np.isfinite(emb).all(axis=1)).sum())):
            logger.warning(f"{os.path.basename(path)}: {n_bad}/{len(emb)} embeddings are NaN/Inf")
        # Atomic: a crash never leaves a truncated .npy that would be taken as done.
        tmp = f"{out}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.save(fh, emb)
        os.replace(tmp, out)
        n_docs += len(texts)
        el = time.time() - t0
        logger.info(f"[{i}/{len(todo)}] {os.path.basename(path)} -> {npy_name(path)} "
                    f"{emb.shape} | {n_docs:,} docs in {el:.0f}s ({n_docs / el:,.0f} docs/s)")


if __name__ == "__main__":
    main()
