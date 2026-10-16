# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Stage 3 — fit k-means centroids, then assign every document to one.

Two sub-commands, because fitting is one GPU for minutes and assigning is
embarrassingly parallel over sources:

    python -m cluster_corpus.cluster fit    --embedding_root data_out/embeddings \\
        --k_clusters 1200 --output_dir data_out/clustered_k1200
    python -m cluster_corpus.cluster assign --cluster_dir data_out/clustered_k1200

    # or fan the assignment out, one source per GPU, then combine:
    python -m cluster_corpus.cluster assign --cluster_dir ... --source arxiv
    python -m cluster_corpus.cluster assign --cluster_dir ... --combine

`fit` samples the pool down to FAISS's training cap (k * max_points_per_centroid) and
fits SPHERICAL k-means -- stage 2 writes L2-normalized vectors, so inner product is
cosine. It writes centroids.npy and pool_info.json.

There is **no per-source weighting here.** The mixture is set once in stage 1 by
`normalize --frac`, so this fits on whatever the normalized pool contains. That is
deliberate: when the fit sample and the training set are drawn from different mixtures,
clusters in an over-represented region hold more documents than their centroid budget
and are correspondingly less coherent.

ORDER IS FROZEN by pool_info.json. `source_order` plus each source's sorted-shard, row
order defines the global document index, and stage 4 must walk the sources in exactly
that order for labels to line up with documents.

Documents whose embedding is NaN/Inf (e.g. a near-empty document normalizing to 0/0)
cannot be searched; they get label -1 and stage 4 drops them.
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CENTROIDS = "centroids.npy"
POOL_INFO = "pool_info.json"


# ── shared ────────────────────────────────────────────────────────────────────

def source_shards(embedding_root: str, source: str) -> list[str]:
    """One source's shard_*.npy in sorted (== document) order."""
    d = os.path.join(embedding_root, source)
    shards = sorted(f for f in os.listdir(d)
                    if f.startswith("shard_") and f.endswith(".npy")) if os.path.isdir(d) else []
    if not shards:
        raise SystemExit(f"no shard_*.npy in {d}")
    return [os.path.join(d, f) for f in shards]


def discover_sources(embedding_root: str, sources: list[str] | None) -> dict[str, list[str]]:
    """{source: shard paths}. Default order is alphabetical, and whatever order comes
    out here is FROZEN into pool_info.json."""
    names = sources or sorted(d for d in os.listdir(embedding_root)
                              if os.path.isdir(os.path.join(embedding_root, d)))
    out = {}
    for name in names:
        try:
            out[name] = source_shards(embedding_root, name)
        except SystemExit:
            logger.warning(f"source {name!r}: no shards; skipping")
    if not out:
        raise SystemExit(f"no <source>/shard_*.npy under {embedding_root}")
    return out


def load_info(cluster_dir: str) -> dict:
    with open(os.path.join(cluster_dir, POOL_INFO)) as f:
        return json.load(f)


# ── fit ───────────────────────────────────────────────────────────────────────

def _sample_indices(n_total: int, k: int, rng) -> np.ndarray:
    """k distinct sorted indices uniform on [0, n_total), in O(min(k, n-k)) memory."""
    if k >= n_total:
        return np.arange(n_total)
    pick_drop = k > n_total // 2          # cheaper to choose the complement
    m = n_total - k if pick_drop else k
    chosen: set[int] = set()
    while len(chosen) < m:
        chosen.update(int(v) for v in rng.integers(0, n_total, size=(m - len(chosen)) * 2 + 8))
    sel = np.fromiter(chosen, dtype=np.int64, count=len(chosen))
    if len(sel) > m:
        sel = rng.choice(sel, size=m, replace=False)
    sel.sort()
    if pick_drop:
        mask = np.ones(n_total, dtype=bool)
        mask[sel] = False
        return np.nonzero(mask)[0]
    return sel


def sample_source(paths: list[str], n_req: int, rng) -> np.ndarray:
    """Uniformly sample min(n_req, N) rows across a source's shards, holding one shard
    in RAM at a time."""
    counts = [int(np.load(p, mmap_mode="r").shape[0]) for p in paths]
    n_total = sum(counts)
    if n_req >= n_total:
        return np.concatenate([np.load(p) for p in paths], axis=0)
    idx = _sample_indices(n_total, n_req, rng)
    starts = np.cumsum([0] + counts)
    out = []
    for si, path in enumerate(paths):
        lo, hi = int(starts[si]), int(starts[si + 1])
        a, b = np.searchsorted(idx, lo, "left"), np.searchsorted(idx, hi, "left")
        if b > a:
            out.append(np.load(path)[idx[a:b] - lo])
    return np.concatenate(out, axis=0)


def cmd_fit(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    src_shards = discover_sources(args.embedding_root,
                                  args.sources.split(",") if args.sources else None)
    order = list(src_shards)                                      # FROZEN
    doc_counts = {s: sum(int(np.load(p, mmap_mode="r").shape[0]) for p in src_shards[s])
                  for s in order}
    total = sum(doc_counts.values())
    logger.info(f"sources (frozen order): {order}")
    logger.info(f"pool: {total:,} documents")

    # FAISS trains on at most k * max_points_per_centroid points, so sample down to
    # that here rather than loading embeddings k-means would discard. Each source is
    # scaled by the same factor, so the fit sample keeps the pool's composition.
    cap = args.k_clusters * args.max_points_per_centroid
    scale = min(1.0, cap / total) if total else 1.0
    fit_n = {s: max(1, round(doc_counts[s] * scale)) for s in order}
    if scale < 1.0:
        logger.info(f"pool {total:,} > cap k*mppc={cap:,}; sampling {scale:.4f} of each source")

    rng = np.random.default_rng(args.seed)
    parts = []
    for name in order:
        s = sample_source(src_shards[name], fit_n[name], rng)
        parts.append(np.ascontiguousarray(s, dtype=np.float32))
        logger.info(f"  {name}: {doc_counts[name]:,} docs -> {len(s):,} fit rows")
    sample = np.concatenate(parts, axis=0)
    del parts

    # faiss rejects non-finite input outright.
    finite = np.isfinite(sample).all(axis=1)
    if (n_bad := int((~finite).sum())):
        logger.warning(f"dropping {n_bad:,}/{len(sample):,} fit rows with NaN/Inf")
        sample = np.ascontiguousarray(sample[finite])
    if not len(sample):
        raise SystemExit("every sampled row is non-finite -- check the embeddings")

    dim = sample.shape[1]
    logger.info(f"fitting spherical k-means: {sample.shape}, k={args.k_clusters}, "
                f"niter={args.niter}, seed={args.seed} ...")
    import faiss
    km = faiss.Kmeans(dim, args.k_clusters, niter=args.niter, verbose=True, gpu=not args.cpu,
                      spherical=True, seed=args.seed,
                      max_points_per_centroid=args.max_points_per_centroid)
    km.train(sample)
    centroids = np.ascontiguousarray(km.centroids).reshape(args.k_clusters, dim)
    np.save(os.path.join(args.output_dir, CENTROIDS), centroids)

    with open(os.path.join(args.output_dir, POOL_INFO), "w") as f:
        json.dump({
            "source_order": order,        # stage 4 MUST reuse this order
            "doc_counts": doc_counts,     # embedding rows per source (== documents)
            "fit_counts": {s: int(fit_n[s]) for s in order},
            "fit_total": int(sum(fit_n.values())),
            "total_docs": int(total),
            "k_clusters": args.k_clusters,
            "dim": int(dim),
            "max_points_per_centroid": args.max_points_per_centroid,
            "seed": args.seed,
            "niter": args.niter,
            "spherical": True,
            "embedding_root": os.path.abspath(args.embedding_root),
        }, f, indent=2)
    logger.info(f"centroids {centroids.shape} + {POOL_INFO} -> {args.output_dir}\n"
                f"next: python -m cluster_corpus.cluster assign --cluster_dir {args.output_dir}")


# ── assign ────────────────────────────────────────────────────────────────────

def build_index(centroids: np.ndarray, use_gpu: bool = True):
    """Flat inner-product index. The centroids are unit-norm (spherical k-means), so
    top-1 by dot product is the nearest centroid on the sphere."""
    import faiss
    index = faiss.IndexFlatIP(centroids.shape[1])
    if use_gpu:
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"GPU index unavailable ({e}); using CPU")
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    return index


def assign_source(embedding_root: str, source: str, index, batch: int = 200_000) -> np.ndarray:
    """Labels for one source in sorted-shard, row order. Non-finite rows get -1 so the
    array stays 1:1 with documents."""
    labels, n_bad = [], 0
    for path in source_shards(embedding_root, source):
        x = np.ascontiguousarray(np.load(path), dtype=np.float32)
        for s in range(0, len(x), batch):
            xb = x[s:s + batch]
            lab = np.full(len(xb), -1, dtype=np.int32)
            finite = np.isfinite(xb).all(axis=1)
            n_bad += int((~finite).sum())
            if finite.any():
                _, idx = index.search(np.ascontiguousarray(xb[finite]), 1)
                lab[finite] = idx.ravel().astype(np.int32)
            labels.append(lab)
    if n_bad:
        logger.warning(f"{source}: {n_bad:,} NaN/Inf embeddings -> label -1 (unclustered)")
    return np.concatenate(labels) if labels else np.empty(0, np.int32)


def combine(cluster_dir: str, info: dict) -> np.ndarray:
    """Concatenate labels_<source>.npy in frozen order -> combined_labels.npy."""
    parts = []
    for name in info["source_order"]:
        lp = os.path.join(cluster_dir, f"labels_{name}.npy")
        if not os.path.exists(lp):
            raise SystemExit(f"missing {lp}; assign source {name!r} first")
        lab = np.load(lp)
        if len(lab) != (exp := info["doc_counts"][name]):
            raise SystemExit(f"{name}: {len(lab)} labels != {exp} documents")
        parts.append(lab)
    combined = np.concatenate(parts)
    out = os.path.join(cluster_dir, "combined_labels.npy")
    np.save(out, combined)
    logger.info(f"combined_labels.npy: {len(combined):,} docs, "
                f"{len(np.unique(combined[combined >= 0]))}/{info['k_clusters']} non-empty "
                f"clusters, {int((combined < 0).sum()):,} unclustered -> {out}")
    return combined


def cmd_assign(args) -> None:
    info = load_info(args.cluster_dir)
    order = info["source_order"]
    root = args.embedding_root or info["embedding_root"]

    if args.combine:
        combine(args.cluster_dir, info)
        return

    index = build_index(np.load(os.path.join(args.cluster_dir, CENTROIDS)), use_gpu=not args.cpu)
    for name in ([args.source] if args.source else order):
        if name not in order:
            raise SystemExit(f"source {name!r} not in source_order {order}")
        outp = os.path.join(args.cluster_dir, f"labels_{name}.npy")
        if os.path.exists(outp):
            logger.info(f"{name}: labels exist, skipping")
            continue
        lab = assign_source(root, name, index, batch=args.batch)
        if len(lab) != (exp := info["doc_counts"][name]):
            raise SystemExit(f"{name}: assigned {len(lab)} != {exp} documents")
        np.save(outp, lab)
        logger.info(f"{name}: {len(lab):,} assigned -> {outp}")

    if not args.source:            # whole-set run also writes the frozen concatenation
        combine(args.cluster_dir, info)


def cmd_all(args) -> None:
    """fit, then assign every source. The two sub-commands exist so the assignment can
    be fanned out one source per GPU; when you are not doing that, this is the whole
    stage in one call."""
    cmd_fit(args)
    cmd_assign(argparse.Namespace(
        cluster_dir=args.output_dir, embedding_root=args.embedding_root,
        source=None, combine=False, batch=args.batch, cpu=args.cpu))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_fit_args(q):
        q.add_argument("--embedding_root", required=True, help="parent of <source>/shard_*.npy")
        q.add_argument("--output_dir", required=True)
        q.add_argument("--k_clusters", type=int, required=True)
        q.add_argument("--sources", default=None,
                       help="Comma-separated source order to FREEZE (default: alphabetical).")
        q.add_argument("--max_points_per_centroid", type=int, default=256,
                       help="FAISS training cap: fit points <= k * this. Also bounds RAM.")
        q.add_argument("--niter", type=int, default=25)
        q.add_argument("--seed", type=int, default=1)
        q.add_argument("--cpu", action="store_true")

    a_all = sub.add_parser("all", help="fit then assign (the whole stage)")
    add_fit_args(a_all)
    a_all.add_argument("--batch", type=int, default=200_000)
    a_all.set_defaults(fn=cmd_all)

    f = sub.add_parser("fit", help="fit centroids only (then assign separately)")
    add_fit_args(f)
    f.set_defaults(fn=cmd_fit)

    a = sub.add_parser("assign", help="assign every document to its nearest centroid")
    a.add_argument("--cluster_dir", required=True, help=f"dir with {CENTROIDS} + {POOL_INFO}")
    a.add_argument("--embedding_root", default=None, help="override pool_info's path")
    a.add_argument("--source", default=None, help="assign only this source (array job)")
    a.add_argument("--combine", action="store_true", help="only concatenate existing labels")
    a.add_argument("--batch", type=int, default=200_000)
    a.add_argument("--cpu", action="store_true")
    a.set_defaults(fn=cmd_assign)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
