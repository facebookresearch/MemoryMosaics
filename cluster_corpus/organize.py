# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Stage 4 — write per-cluster JSONL, then hold out test clusters.

    python -m cluster_corpus.organize write --cluster_dir data_out/clustered_k1200 \\
        --jsonl_root data_out/normalized --output_dir data_out/clustered_k1200
    python -m cluster_corpus.organize split --data_dir data_out/clustered_k1200 --n_test 50

`write` walks every source in the frozen `source_order` from pool_info.json, pairs each
document with its label from stage 3, and appends it to `cluster_XXXX/chunk.NNNN.jsonl`.
Records are {"text", "eidx", "source"}; the training loader reads only "text".

**No --frac here.** The mixture was applied once in stage 1 (`normalize --frac`), so
everything in the normalized pool is written. The only documents dropped are those with
label -1, i.e. whose embedding was NaN/Inf and could not be assigned.

Sources are INTERLEAVED, not concatenated: the emission order is a shuffled sequence of
source indices, so consecutive documents inside one on-disk chunk come from different
corpora. Without this the loader would stream long single-source runs.

`split` partitions the clusters into train/test metadata. The default is the LAST
n_test cluster ids (deterministic, no seed); --shuffle takes a seeded random subset.
"""

import argparse
import glob
import json
import logging
import os
import resource
from collections import OrderedDict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── writing ───────────────────────────────────────────────────────────────────

class ClusterWriter:
    """Append {text, eidx, source} to cluster_XXXX/chunk.NNNN.jsonl.

    A cluster's documents are round-robined across `num_chunks` files by arrival order,
    so eidx = i*num_chunks + chunk recovers the position. K*num_chunks handles stay open
    where the file-descriptor limit allows (it is raised toward the hard limit first);
    otherwise an LRU pool reopens in append mode, which is correct but slower.
    """

    def __init__(self, output_dir: str, k: int, num_chunks: int):
        self.output_dir, self.k, self.num_chunks = output_dir, k, num_chunks
        self.counters = np.zeros(k, dtype=np.int64)
        need = k * num_chunks + 64
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < (target := min(hard, need)):
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
                soft = target
            except (ValueError, OSError):
                pass
        self.max_open = max(64, soft - 64)
        self._handles: OrderedDict = OrderedDict()
        self._seen: set = set()          # (cid, chunk) opened before -> reopen appends
        self._dirs: set = set()
        if need > self.max_open:
            logger.warning(f"{need} files needed, {self.max_open} handles available; using an "
                           f"LRU pool (reopen-append). Raise `ulimit -n` for speed.")

    def _handle(self, cid: int, chunk: int):
        key = (cid, chunk)
        if (fh := self._handles.get(key)) is not None:
            self._handles.move_to_end(key)
            return fh
        if len(self._handles) >= self.max_open:
            self._handles.popitem(last=False)[1].close()
        cdir = os.path.join(self.output_dir, f"cluster_{cid:04d}")
        if cdir not in self._dirs:
            os.makedirs(cdir, exist_ok=True)
            self._dirs.add(cdir)
        fh = open(os.path.join(cdir, f"chunk.{chunk:04d}.jsonl"), "a" if key in self._seen else "w")
        self._seen.add(key)
        self._handles[key] = fh
        return fh

    def write(self, cid: int, text: str, source: str | None = None) -> None:
        idx = int(self.counters[cid])
        rec = {"text": text, "eidx": idx}
        if source is not None:
            rec["source"] = source
        self._handle(cid, idx % self.num_chunks).write(json.dumps(rec) + "\n")
        self.counters[cid] += 1

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


def iter_source_records(jsonl_dir: str, n_docs: int, file_rows: dict | None = None):
    """(index, record) for one source, in stage-2 embedding order.

    Sorted shard filename, then line order, skipping blank/unparseable lines, with the
    index advancing only on kept lines -- exactly how embed.py enumerated them, so index
    i here is embedding row i. `file_rows` ({jsonl name: rows in its .npy}) is checked
    as each file ends, so a .jsonl edited after embedding fails here instead of
    shifting every later label by a few documents.
    """
    files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl found in {jsonl_dir}")
    i = 0
    for fp in files:
        start = i
        with open(fp) as fh:
            for line in fh:
                if not (line := line.strip()):
                    continue
                try:
                    obj = json.loads(line, parse_int=str)
                except (json.JSONDecodeError, ValueError):
                    continue
                if i >= n_docs:
                    if file_rows is not None:
                        raise SystemExit(f"{fp}: more documents than the {n_docs} embedded "
                                         f"-- the .jsonl changed after stage 2.")
                    return
                yield i, obj
                i += 1
        name = os.path.basename(fp)
        if file_rows is not None and i - start != file_rows[name]:
            raise SystemExit(f"{fp}: {i - start} documents but its embedding has "
                             f"{file_rows[name]} rows -- the .jsonl changed after stage 2.")


def embedding_rows(jsonl_dir: str, emb_dir: str) -> dict:
    """{jsonl name: rows in the matching .npy}, requiring a 1:1 pairing.

    Stage 2 writes shard_X.npy for every shard_X.jsonl. A missing one (a failed or
    unfinished embedding task) would make every later document take the label of the
    document after it, so refuse to start instead.
    """
    jsonl = {os.path.basename(p)[:-len(".jsonl")]
             for p in glob.glob(os.path.join(jsonl_dir, "*.jsonl"))}
    npy = {os.path.basename(p)[:-len(".npy")]
           for p in glob.glob(os.path.join(emb_dir, "shard_*.npy"))}
    if jsonl != npy:
        raise SystemExit(f"{jsonl_dir} and {emb_dir} do not pair up 1:1 -- "
                         f"no embedding for {sorted(jsonl - npy)[:5]}, "
                         f"no jsonl for {sorted(npy - jsonl)[:5]}. Finish stage 2 first.")
    return {f"{stem}.jsonl": int(np.load(os.path.join(emb_dir, f"{stem}.npy"),
                                         mmap_mode="r").shape[0]) for stem in jsonl}


def _iter_clustered(jsonl_dir: str, labels: np.ndarray, n_docs: int, content_field: str,
                    file_rows: dict | None = None):
    """(text, cluster) for one source's assignable documents, in embedding order."""
    for i, obj in iter_source_records(jsonl_dir, n_docs, file_rows):
        if labels[i] >= 0:                      # -1 == unassignable embedding
            yield obj.get(content_field, ""), int(labels[i])


def write_metadata(output_dir: str, k: int, num_chunks: int,
                   counters: np.ndarray, extra: dict | None = None) -> None:
    sizes = counters.tolist()
    meta = {"K": k, "num_chunks_per_cluster": num_chunks,
            "total_documents": int(counters.sum()), "cluster_sizes": sizes}
    meta.update(extra or {})
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    a = np.array(sizes)
    logger.info(f"{int(a.sum()):,} docs -> {k} clusters ({int((a > 0).sum())} non-empty). "
                f"sizes: min={a.min()} max={a.max()} mean={a.mean():.0f} median={np.median(a):.0f}")


def cmd_write(args) -> None:
    with open(os.path.join(args.cluster_dir, "pool_info.json")) as f:
        info = json.load(f)
    order, doc_counts, k = info["source_order"], info["doc_counts"], int(info["k_clusters"])
    emb_root = args.embedding_root or info["embedding_root"]
    file_rows = {s: embedding_rows(os.path.join(args.jsonl_root, s), os.path.join(emb_root, s))
                 for s in order}

    labels_by_src, n_keep, n_drop = {}, {}, {}
    for s in order:
        labels = np.load(os.path.join(args.cluster_dir, f"labels_{s}.npy"))
        if len(labels) != (exp := int(doc_counts[s])):
            raise SystemExit(f"{s}: labels_{s}.npy has {len(labels)} != {exp} documents")
        labels_by_src[s] = labels
        n_drop[s] = int((labels < 0).sum())
        n_keep[s] = exp - n_drop[s]
        logger.info(f"  {s}: {exp:,} docs, {n_keep[s]:,} assignable, {n_drop[s]:,} unclustered(-1)")

    if not (total := sum(n_keep.values())):
        raise SystemExit("no assignable documents -- check the labels")

    # Interleave: a shuffled array holding source index si exactly n_keep[si] times.
    # Emitting in this order mixes corpora within each on-disk chunk.
    interleave = np.empty(total, dtype=np.int16)
    pos = 0
    for si, s in enumerate(order):
        interleave[pos:pos + n_keep[s]] = si
        pos += n_keep[s]
    np.random.default_rng(args.seed).shuffle(interleave)

    os.makedirs(args.output_dir, exist_ok=True)
    writer = ClusterWriter(args.output_dir, k, args.num_chunks_per_cluster)
    gens = [_iter_clustered(os.path.join(args.jsonl_root, s), labels_by_src[s],
                            int(doc_counts[s]), args.content_field, file_rows[s]) for s in order]
    try:
        for n, si in enumerate(interleave):
            try:
                text, cid = next(gens[si])
            except StopIteration:
                raise SystemExit(
                    f"source {order[si]!r} ran out of documents early -- its *.jsonl changed "
                    f"since embedding (fewer parseable lines than pool_info doc_counts).")
            writer.write(cid, text, source=order[si])
            if (n + 1) % 1_000_000 == 0:
                logger.info(f"  wrote {n + 1:,}/{total:,}")
        # Run each source to its end: that is when the last file's row count is checked,
        # and anything still yielded means more assignable documents than labels said.
        for s, g in zip(order, gens):
            for _ in g:
                raise SystemExit(f"source {s!r} has more assignable documents than its labels")
    finally:
        writer.close()

    write_metadata(args.output_dir, k, args.num_chunks_per_cluster, writer.counters, extra={
        "sources": order,
        "documents_per_source": n_keep,
        "unclustered_per_source": n_drop,
        "interleave_seed": args.seed,
        "jsonl_root": os.path.abspath(args.jsonl_root),
        "cluster_dir": os.path.abspath(args.cluster_dir),
    })


# ── train / test split ────────────────────────────────────────────────────────

def cmd_split(args) -> None:
    data_dir = Path(args.data_dir)
    with open(data_dir / args.metadata) as f:
        meta = json.load(f)
    k, sizes = meta["K"], meta["cluster_sizes"]
    if len(sizes) != k:
        raise SystemExit(f"metadata inconsistent: K={k} but {len(sizes)} cluster sizes")

    if args.n_test is not None:
        n_test = args.n_test
    elif args.test_ratio is not None:
        n_test = max(1, int(k * args.test_ratio))
    else:
        raise SystemExit("give --n_test or --test_ratio")
    if not 0 < n_test < k:
        raise SystemExit(f"n_test={n_test} must be in (0, K={k})")
    n_train = k - n_test

    if args.shuffle:
        perm = np.random.RandomState(args.seed).permutation(k)
        train_ids, test_ids = sorted(perm[:n_train].tolist()), sorted(perm[n_train:].tolist())
    else:
        train_ids, test_ids = list(range(n_train)), list(range(n_train, k))

    # Key names differ by design: the training loader reads "cluster_ids", the
    # per-cluster eval reads "test_cluster_ids".
    for name, ids, key in (("metadata_train.json", train_ids, "cluster_ids"),
                           ("metadata_test.json", test_ids, "test_cluster_ids")):
        s = [sizes[i] for i in ids]
        with open(data_dir / name, "w") as f:
            json.dump({"K": len(ids), "cluster_sizes": s, key: ids,
                       "total_documents": sum(s), "split_seed": args.seed,
                       "shuffled": bool(args.shuffle)}, f, indent=2)
        logger.info(f"{name}: {len(ids)} clusters, {sum(s):,} documents "
                    f"(ids {ids[0]}..{ids[-1]})")


def cmd_all(args) -> None:
    """write, then split. The split is a metadata partition that always follows the
    write, so the two are only separate sub-commands for re-splitting without
    regenerating the (expensive) per-cluster files."""
    cmd_write(args)
    cmd_split(argparse.Namespace(
        data_dir=args.output_dir, metadata="metadata.json",
        n_test=args.n_test, test_ratio=args.test_ratio,
        seed=args.split_seed, shuffle=args.shuffle))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_write_args(q):
        q.add_argument("--cluster_dir", required=True, help="pool_info.json + labels_<source>.npy")
        q.add_argument("--jsonl_root", required=True, help="parent of <source>/*.jsonl (stage 1)")
        q.add_argument("--embedding_root", default=None,
                       help="parent of <source>/shard_*.npy; default: pool_info's path")
        q.add_argument("--output_dir", required=True)
        q.add_argument("--num_chunks_per_cluster", type=int, default=1)
        q.add_argument("--content_field", default="text")
        q.add_argument("--seed", type=int, default=1, help="source-interleave shuffle seed")

    a_all = sub.add_parser("all", help="write then split (the whole stage)")
    add_write_args(a_all)
    a_all.add_argument("--n_test", type=int, default=None)
    a_all.add_argument("--test_ratio", type=float, default=None)
    # Distinct from --seed above: one shuffles the source interleave, the other picks
    # the test clusters. Same name for both would silently tie two unrelated draws.
    a_all.add_argument("--split_seed", type=int, default=42)
    a_all.add_argument("--shuffle", action="store_true",
                       help="Random train/test split. Default: the LAST n_test ids.")
    a_all.set_defaults(fn=cmd_all)

    w = sub.add_parser("write", help="write per-cluster jsonl only")
    add_write_args(w)
    w.set_defaults(fn=cmd_write)

    s = sub.add_parser("split", help="hold out test clusters")
    s.add_argument("--data_dir", required=True)
    s.add_argument("--metadata", default="metadata.json")
    s.add_argument("--n_test", type=int, default=None)
    s.add_argument("--test_ratio", type=float, default=None)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--shuffle", action="store_true",
                   help="Random split. Default: the LAST n_test cluster ids.")
    s.set_defaults(fn=cmd_split)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
