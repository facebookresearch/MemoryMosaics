# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Stage 1 — normalize every source into one JSONL schema, applying the mixture.

Each source becomes  <out_dir>/<source>/shard_00000.jsonl ...  holding records

    {"id": str, "source": str, "text": str}

**This is the only place the source mixture is set.** `--frac` subsamples here, before
anything is embedded, so every later stage just operates on the whole normalized pool:
k-means fits on it, and the organize stage writes all of it. There is no second fraction
to keep in sync, and no way to cluster one mixture while training on another.

Sources
-------
    cosmopedia            HF HuggingFaceTB/cosmopedia, all 8 subsets (or --subsets)
    nemotron_cc_high_10m  HF OpenSQZ/AutoMathText-V2 [nemotron_cc_high], first --limit
    gutenberg             HF manu/project_gutenberg, deduped, boilerplate stripped
    sharegpt              HF anon8231489123/ShareGPT_Vicuna_unfiltered (fetched on
                          first use), conversation turns flattened
    arxiv                 local proof-pile-2 dump
    open-web-math         local proof-pile-2 dump

Everything streams, so the large sources never load into RAM. Run one source at a time:

    python -m cluster_corpus.normalize --source cosmopedia --workers 16
    python -m cluster_corpus.normalize --source arxiv --frac 0.25 --raw_dir data_out/raw
    python -m cluster_corpus.normalize --source sharegpt          # downloads on first use

Within a source, the four big ones run in parallel, one worker per input file:
cosmopedia and nemotron per upstream parquet file (downloaded, text column only, then
deleted), arxiv and open-web-math per local dump file. Each input file becomes
shard_<file index>.jsonl. The output is the same documents, ids and order as a single
sequential pass -- only the shard boundaries differ, which stage 2 does not care about.
`_plan.json` pins the file list (and HF revision), and finished shards are skipped, so
an interrupted run resumes where it stopped. gutenberg and sharegpt are small and stay
sequential.
"""

import argparse
import glob
import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
# The HF/httpx stack logs one line per request; keep our own INFO output readable.
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "fsspec", "datasets"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SHARD_DOCS = 50_000   # records per output shard
MIN_CHARS = 32        # drop trivially short documents after cleaning

COSMOPEDIA_SUBSETS = ["auto_math_text", "khanacademy", "openstax", "stanford",
                      "stories", "web_samples_v1", "web_samples_v2", "wikihow"]
NEMOTRON_LIMIT = 10_000_000


# ── output ────────────────────────────────────────────────────────────────────

class ShardWriter:
    """Write {id, source, text} into rolling JSONL shards, counting as it goes."""

    def __init__(self, out_dir: str, source: str, min_chars: int = MIN_CHARS):
        self.dir = os.path.join(out_dir, source)
        os.makedirs(self.dir, exist_ok=True)
        self.source, self.min_chars = source, min_chars
        self._fh, self._shard, self._in_shard = None, 0, 0
        self.total = 0

    def write(self, doc_id, text: str) -> None:
        text = (text or "").strip()
        if len(text) < self.min_chars:
            return
        if self._fh is None or self._in_shard >= SHARD_DOCS:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(os.path.join(self.dir, f"shard_{self._shard:05d}.jsonl"), "w")
            self._shard, self._in_shard = self._shard + 1, 0
        self._fh.write(json.dumps({"id": str(doc_id), "source": self.source, "text": text}) + "\n")
        self._in_shard += 1
        self.total += 1
        if self.total % 100_000 == 0:
            logger.info(f"  {self.source}: wrote {self.total:,} docs")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


# ── the mixture ───────────────────────────────────────────────────────────────

def keep_fn(frac, seed: int, source: str):
    """Deterministic per-document keep test for --frac.

    The decision hashes (seed, source, doc_id) rather than using a prefix or a stateful
    RNG. A prefix (as --limit does) inherits whatever ordering the upstream stream has;
    a stateful RNG makes the outcome depend on how the stream was chunked or resumed.
    Hashing makes each document independent, so the same --frac/--frac_seed reproduce
    the same subset regardless of order, sharding or restarts.
    """
    if frac is None or frac >= 1.0:
        return lambda doc_id, i: True
    if not 0 < frac < 1:
        raise SystemExit(f"--frac must be in (0, 1]; got {frac}")
    cutoff = int(frac * (1 << 32))

    def keep(doc_id, i):
        key = f"{seed}:{source}:{doc_id if doc_id else i}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") < cutoff
    return keep


# ── cleaning ──────────────────────────────────────────────────────────────────

# Project Gutenberg wraps each book in a header/footer plus a license. The end marker
# appears both as the modern "*** END OF ... ***" and as a bare "End of the Project
# Gutenberg EBook of <title>" line in older texts.
_PG_START = re.compile(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG EBOOK[\s\S]*?\*\*\*"
                     r"|^[^\n]*End of[^\n]*Project Gutenberg[^\n]*", re.I | re.M)
_PG_CREDIT = re.compile(r"\A\s*Produced by .*?\n", re.I)

SHAREGPT_MARKERS = {"human": "[User:]\n", "gpt": "[Assistant:]\n", "system": "[System:]\n"}


def clean_gutenberg(text: str) -> str:
    """Keep the body between the START/END markers; drop license and credits."""
    if not text:
        return ""
    if (m := _PG_START.search(text)):
        text = text[m.end():]
    if (m := _PG_END.search(text)):
        text = text[:m.start()]
    return _PG_CREDIT.sub("", text).strip()


def sharegpt_to_text(conversations) -> str:
    """Flatten one ShareGPT conversation (list of {from, value}) into a document."""
    parts = []
    for turn in conversations or []:
        if (val := (turn.get("value") or "").strip()):
            parts.append(f"{SHAREGPT_MARKERS.get(turn.get('from'), '[User:]')} {val}")
    return "\n\n".join(parts)


# ── readers: each yields (doc_id, text) ───────────────────────────────────────

def iter_gutenberg(language: str):
    """manu/project_gutenberg keeps each language as a SPLIT of one config. Downloaded
    (not streamed) so re-runs after a cleaning tweak read from the HF cache."""
    from datasets import load_dataset
    logger.info(f"loading manu/project_gutenberg [split={language}] (dedup by id) ...")
    seen, dupes = set(), 0
    for row in load_dataset("manu/project_gutenberg", split=language, streaming=False):
        gid = str(row.get("id"))
        if gid in seen:
            dupes += 1
            continue
        seen.add(gid)
        yield gid, clean_gutenberg(row.get("text", ""))
    logger.info(f"  gutenberg: {len(seen):,} unique ids, {dupes:,} duplicates dropped")


SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"


def ensure_sharegpt(path: str | None, filename: str = SHAREGPT_FILE) -> str:
    """Local path to the ShareGPT json, fetching it on first use.

    Returns the HF cache path rather than copying into --raw_dir: hf_hub_download
    already handles caching, resume and integrity, and moving the file out would just
    make the next call re-download it.

    The dataset is public (no login). NOTE: it is unfiltered user-generated content and
    may contain offensive material.
    """
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"--sharegpt_json {path} does not exist")
        return path
    from huggingface_hub import hf_hub_download
    logger.info(f"fetching {SHAREGPT_REPO} / {filename} (~670 MB, cached after the first run) ...")
    return hf_hub_download(repo_id=SHAREGPT_REPO, filename=filename, repo_type="dataset")


def iter_sharegpt(path: str):
    logger.info(f"loading ShareGPT json {path} ...")
    with open(path) as f:
        # parse_int=str dodges Python's int<->str digit limit on pathological literals.
        data = json.load(f, parse_int=str)
    for rec in data:
        yield rec.get("id", ""), sharegpt_to_text(rec.get("conversations"))


def _open_lines(path: str):
    """Lines of a .jsonl or .jsonl.zst file (zst needs `pip install zstandard`)."""
    if path.endswith(".zst"):
        import zstandard as zstd
        with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
            yield from io.TextIOWrapper(r, encoding="utf-8")
    else:
        with open(path) as fh:
            yield from fh


# ── parallel sources: one worker per input file ───────────────────────────────
# The big sources are split by input file -- an upstream parquet file (cosmopedia,
# nemotron) or a local .jsonl[.zst] dump file (arxiv, open-web-math). Each worker turns
# one input file into shard_<file index>.jsonl, so sorted shards are the input order.
# Every doc's id, its --frac decision and --limit are computed exactly as a single
# sequential pass would, so the output is the same documents in the same order; only
# the shard boundaries differ, which stage 2 does not care about.
#
# `_plan.json` freezes the file list (and, for HF, the upstream revision); a shard that
# exists is finished (written to .tmp, then renamed), so re-running resumes.

PLAN_FILE = "_plan.json"
STATS_DIR = "_stats"          # per-shard {seen, kept, written}, so a resume counts exactly
_HF_URL = re.compile(r"^hf://datasets/(?P<repo>.+?)@(?P<rev>[^/]+)/(?P<path>.+)$")


def _shard_name(index: int) -> str:
    return f"shard_{index:05d}.jsonl"


def _hf_files(repo: str, configs: list[str], split: str) -> list[dict]:
    """Upstream parquet files in the order `load_dataset(streaming=True)` reads them.

    The list comes from `datasets` itself, not a sorted listing: the order is not
    natural (nemotron goes part.0, part.1, part.10, part.100, ...), and a --limit
    prefix is only reproducible if it is exactly this order.
    """
    from datasets import load_dataset_builder
    files = []
    for cfg in configs:
        for url in load_dataset_builder(repo, cfg).config.data_files[split]:
            if not (m := _HF_URL.match(str(url))):
                raise SystemExit(f"unexpected data file url {url!r}")
            files.append({"cfg": cfg, "revision": m["rev"], "path": m["path"]})
    return files


def _hf_plan(repo: str, configs: list[str], split: str, text_field: str,
             limit, workers: int) -> dict:
    """Files with row counts, global offsets and how many rows to take from each.

    Ids are f"{config}-{global row}" and --limit counts raw rows, as in a streaming
    pass, so each file needs the row count of everything before it. Footers are read
    over HTTP range requests (a few KB per file), in order and in batches, stopping
    once --limit is covered -- nemotron needs 50 of 18k files.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    def footer(f):
        with fs.open(f"hf://datasets/{repo}@{f['revision']}/{f['path']}") as fh:
            return pq.read_metadata(fh)

    files = _hf_files(repo, configs, split)
    logger.info(f"planning {repo} [{','.join(configs)}]: {len(files)} upstream file(s)"
                + (f", limit={limit:,}" if limit else ""))
    plan, offset, step = [], 0, max(8, 2 * workers)
    with ThreadPoolExecutor(step) as ex:
        for b in range(0, len(files), step):
            batch = files[b:b + step]
            for fi, (f, md) in enumerate(zip(batch, ex.map(footer, batch)), start=b):
                if fi == 0 and text_field not in md.schema.to_arrow_schema().names:
                    raise SystemExit(f"text field {text_field!r} not in {repo} columns "
                                     f"{md.schema.to_arrow_schema().names}; pass --text_field")
                take = md.num_rows if limit is None else min(md.num_rows, limit - offset)
                if take <= 0:
                    break
                plan.append({**f, "index": fi, "rows": md.num_rows,
                             "offset": offset, "take": take})
                offset += take
            if limit is not None and offset >= limit:
                break
    return {"files": plan, "revisions": sorted({f["revision"] for f in plan}),
            "total_rows": offset}


def _local_files(directory: str) -> list[dict]:
    """Every *.jsonl[.zst] in a directory, in sorted filename order."""
    paths = sorted(glob.glob(os.path.join(directory, "*.jsonl")) +
                   glob.glob(os.path.join(directory, "*.jsonl.zst")))
    if not paths:
        raise SystemExit(f"no .jsonl / .jsonl.zst files in {directory}")
    return [{"index": i, "path": os.path.abspath(p), "bytes": os.path.getsize(p)}
            for i, p in enumerate(paths)]


# Row readers: each yields (doc_id, index for keep_fn, text) for ONE input file.

def _parquet_rows(job: dict):
    """Download one upstream file, read only the text column, delete the download."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = job["file"]
    dl_dir = os.path.join(job["dir"], "_download", str(f["index"]))
    local = hf_hub_download(job["repo"], f["path"], revision=f["revision"],
                            repo_type="dataset", local_dir=dl_dir)
    n = 0
    for batch in pq.ParquetFile(local).iter_batches(batch_size=8192,
                                                    columns=[job["text_field"]]):
        for text in batch.column(0).to_pylist():
            if n >= f["take"]:
                break
            gi = f["offset"] + n
            yield f"{f['cfg']}-{gi}", gi, text
            n += 1
        if n >= f["take"]:
            break
    if n != f["take"]:
        raise RuntimeError(f"{f['path']}: read {n} rows, footer promised {f['take']}")
    shutil.rmtree(dl_dir, ignore_errors=True)


def _local_rows(job: dict):
    """One local dump file. The fallback id f"{file}-{line}" is never empty, so keep_fn
    only ever hashes the id and the per-file line index it gets here is never used --
    each file is independent of the others."""
    path = job["file"]["path"]
    base = os.path.basename(path)
    for i, line in enumerate(_open_lines(path)):
        if not (line := line.strip()):
            continue
        try:
            row = json.loads(line, parse_int=str)
        except (json.JSONDecodeError, ValueError):
            continue
        yield (row.get("id") or f"{base}-{i}"), i, row.get(job["text_field"], "")


_ROW_READERS = {"parquet": _parquet_rows, "local": _local_rows}


def _file_to_shard(job: dict) -> dict:
    """Worker: one input file -> one shard (+ its stats), written atomically."""
    keep = keep_fn(job["frac"], job["frac_seed"], job["source"])
    index = job["file"]["index"]
    out = os.path.join(job["dir"], _shard_name(index))
    stats = {"seen": 0, "kept": 0, "written": 0}
    with open(f"{out}.tmp", "w") as fh:
        for doc_id, i, text in _ROW_READERS[job["kind"]](job):
            stats["seen"] += 1
            if not keep(doc_id, i):
                continue
            stats["kept"] += 1
            text = (text or "").strip()
            if len(text) < job["min_chars"]:
                continue
            fh.write(json.dumps({"id": str(doc_id), "source": job["source"], "text": text}) + "\n")
            stats["written"] += 1
    # Stats first: once the shard exists, the run treats it -- and its stats -- as final.
    with open(os.path.join(job["dir"], STATS_DIR, f"{index:05d}.json"), "w") as fh:
        json.dump(stats, fh)
    os.replace(f"{out}.tmp", out)
    return {"index": index, **stats}


def normalize_parallel(args, kind: str, params: dict, make_plan, extra_job=None) -> dict:
    """Run one source as a pool of per-file workers. Returns totals for the manifest.

    `params` is everything that decides WHICH documents land in the output; resuming
    requires it to match the existing plan exactly.
    """
    out_dir = os.path.join(args.out_dir, args.source)
    os.makedirs(os.path.join(out_dir, STATS_DIR), exist_ok=True)
    plan_path = os.path.join(out_dir, PLAN_FILE)

    if os.path.exists(plan_path):
        with open(plan_path) as fh:
            plan = json.load(fh)
        if (diff := sorted(k for k in params.keys() | plan["params"].keys()
                           if params.get(k) != plan["params"].get(k))):
            raise SystemExit(f"{plan_path} differs from this run in {diff}. "
                             f"Delete {out_dir} to start over.")
        logger.info(f"resuming from {plan_path}")
    else:
        if glob.glob(os.path.join(out_dir, "shard_*.jsonl")):
            raise SystemExit(f"{out_dir} has shards but no {PLAN_FILE} -- stale output from "
                             f"another run would mix into this one. Delete it first.")
        plan = {**make_plan(), "params": params}
        with open(f"{plan_path}.tmp", "w") as fh:
            json.dump(plan, fh, indent=1)
        os.replace(f"{plan_path}.tmp", plan_path)

    files = plan["files"]
    expected = {_shard_name(f["index"]) for f in files}
    if (stray := {os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "shard_*.jsonl"))}
            - expected):
        raise SystemExit(f"{out_dir} has shards not in {PLAN_FILE}: {sorted(stray)[:5]} ...")
    todo = [f for f in files if not os.path.exists(os.path.join(out_dir, _shard_name(f["index"])))]
    logger.info(f"{args.source}: {len(files)} input file(s); {len(files) - len(todo)} already "
                f"done, {len(todo)} to go with {args.workers} worker(s)")

    if todo:
        common = {"kind": kind, "dir": out_dir, "source": args.source,
                  "text_field": args.text_field, "min_chars": args.min_chars,
                  "frac": args.frac, "frac_seed": args.frac_seed, **(extra_job or {})}
        seen = written = 0
        t0 = time.time()
        # spawn, not fork: `datasets`/`huggingface_hub` may have started threads by now.
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_file_to_shard, {**common, "file": f}) for f in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                seen, written = seen + r["seen"], written + r["written"]
                el = max(time.time() - t0, 1e-9)
                logger.info(f"  [{i}/{len(todo)}] {_shard_name(r['index'])}: {r['written']:,} "
                            f"docs | this run {written:,} docs in {el:.0f}s ({seen / el:,.0f} rows/s)")
    shutil.rmtree(os.path.join(out_dir, "_download"), ignore_errors=True)

    total = {"seen": 0, "kept": 0, "written": 0}
    for f in files:
        with open(os.path.join(out_dir, STATS_DIR, f"{f['index']:05d}.json")) as fh:
            for k, v in json.load(fh).items():
                total[k] += v
    if args.frac is not None and args.frac < 1.0:
        logger.info(f"  --frac {args.frac}: kept {total['kept']:,} of {total['seen']:,} "
                    f"({total['kept'] / max(total['seen'], 1):.4f})")
    return {"num_docs": total["written"], "seen_before_frac": total["seen"],
            "input_files": len(files),
            **({"revisions": plan["revisions"]} if "revisions" in plan else {})}


def run_parquet(args, repo: str, configs: list[str]) -> dict:
    params = {"repo": repo, "configs": configs, "split": args.split,
              "text_field": args.text_field, "limit": args.limit, "frac": args.frac,
              "frac_seed": args.frac_seed, "min_chars": args.min_chars}
    return normalize_parallel(
        args, "parquet", params,
        lambda: _hf_plan(repo, configs, args.split, args.text_field, args.limit, args.workers),
        extra_job={"repo": repo})


def run_local(args, directory: str) -> dict:
    files = _local_files(directory)
    # The file list (names and sizes) is part of params: if the dump changes under a
    # half-finished run, resuming would splice two different corpora.
    params = {"files": [[os.path.basename(f["path"]), f["bytes"]] for f in files],
              "text_field": args.text_field, "frac": args.frac,
              "frac_seed": args.frac_seed, "min_chars": args.min_chars}
    return normalize_parallel(args, "local", params, lambda: {"files": files})


# ── source registry ───────────────────────────────────────────────────────────
# One entry per source, so adding a corpus means adding a line here rather than a
# branch in main(). PARALLEL_SOURCES are split per input file and return manifest
# totals; the small ones in SOURCES return a (id, text) stream written sequentially.

PARALLEL_SOURCES = {
    "cosmopedia": lambda a: run_parquet(
        a, "HuggingFaceTB/cosmopedia",
        a.subsets.split(",") if a.subsets else COSMOPEDIA_SUBSETS),
    "nemotron_cc_high_10m": lambda a: run_parquet(
        a, "OpenSQZ/AutoMathText-V2", ["nemotron_cc_high"]),
    "arxiv": lambda a: run_local(a, os.path.join(a.raw_dir, "proof_pile_2_arxiv_train")),
    "open-web-math": lambda a: run_local(
        a, os.path.join(a.raw_dir, "proof_pile_2_open_web_math_train")),
}

SOURCES = {
    # gutenberg dedups by id across the whole split (first occurrence wins), and both
    # are small (38k / 93k docs) -- not worth splitting.
    "gutenberg": lambda a: iter_gutenberg(a.gutenberg_lang),
    "sharegpt": lambda a: iter_sharegpt(ensure_sharegpt(a.sharegpt_json)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted({**SOURCES, **PARALLEL_SOURCES}))
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="Input files processed at once (cosmopedia, nemotron, arxiv, "
                         "open-web-math). Sources themselves run one at a time.")
    ap.add_argument("--out_dir", default="data_out/normalized")
    ap.add_argument("--raw_dir", default="data_out/raw",
                    help="Where the locally staged dumps live (arxiv, open-web-math).")
    ap.add_argument("--frac", type=float, default=None,
                    help="Keep this fraction of the source (deterministic hash of the doc "
                         "id). THE mixture knob -- nothing downstream re-weights.")
    ap.add_argument("--frac_seed", type=int, default=1, help="Seed for --frac.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on documents, as a deterministic PREFIX of the stream. "
                         "Defaults to 10,000,000 for nemotron_cc_high_10m.")
    ap.add_argument("--min_chars", type=int, default=MIN_CHARS,
                    help="Drop documents shorter than this after cleaning.")
    ap.add_argument("--subsets", default=None, help="cosmopedia: comma-separated subsets.")
    ap.add_argument("--gutenberg_lang", default="en",
                    help="Language SPLIT of manu/project_gutenberg (en, de, fr, ...).")
    ap.add_argument("--sharegpt_json", default=None,
                    help="Existing ShareGPT json. Omit to fetch it from HuggingFace.")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    if args.source == "nemotron_cc_high_10m" and args.limit is None:
        args.limit = NEMOTRON_LIMIT

    if args.source in PARALLEL_SOURCES:
        extra = PARALLEL_SOURCES[args.source](args)
        num_docs, n_seen = extra.pop("num_docs"), extra.pop("seen_before_frac")
        shard_docs = "one shard per input file"
    else:
        producer = SOURCES[args.source](args)
        keep = keep_fn(args.frac, args.frac_seed, args.source)
        writer = ShardWriter(args.out_dir, args.source, min_chars=args.min_chars)

        n_seen = n_dropped = 0
        for doc_id, text in producer:
            n_seen += 1
            if keep(doc_id, n_seen - 1):
                writer.write(doc_id, text)
            else:
                n_dropped += 1
        writer.close()

        if n_dropped:
            logger.info(f"  --frac {args.frac}: kept {n_seen - n_dropped:,} of {n_seen:,} "
                        f"({(n_seen - n_dropped) / max(n_seen, 1):.4f})")
        num_docs, shard_docs, extra = writer.total, SHARD_DOCS, {}

    # Provenance next to the data: everything needed to rebuild this source's shard set.
    src_dir = os.path.join(args.out_dir, args.source)
    with open(os.path.join(src_dir, "_manifest.json"), "w") as f:
        json.dump({"source": args.source, "num_docs": num_docs,
                   "shard_docs": shard_docs, "min_chars": args.min_chars,
                   "frac": args.frac, "frac_seed": args.frac_seed, "limit": args.limit,
                   "subsets": args.subsets, "seen_before_frac": n_seen, **extra}, f, indent=2)
    logger.info(f"DONE {args.source}: {num_docs:,} docs -> {src_dir}")

if __name__ == "__main__":
    main()
