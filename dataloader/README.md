# Data loaders

Composable, resumable streaming pipelines. Every loader resumes exactly: a run restarted
from a checkpoint continues the same document stream it would have read uninterrupted.

## Architecture

```
streaming.py: stages, each an iterator with state_dict / load_state_dict
│   JsonlSource (byte-range sharded, endless) → map → shuffle → pack → batch; Sample (weighted mix)
│
├── Pretrain pipeline (pretrain.py)
│   per source: JsonlSource → extract_text → shuffle → tokenize → pack(wrap=True)
│   then Sample across sources (weighted) → batch
│
├── Cluster pipeline (cluster.py)
│   Pre-generated schedule picks the cluster(s) of each sequence; per cluster:
│   JsonlSource → extract_text → shuffle → tokenize, packed into one sequence.
│   LRU-cached pipelines, built in a background process.
│
└── SFT pipeline (sft.py)
    JsonlSource → shuffle → chat template (tokens + loss mask) → pack(wrap=False, pad) → batch
```

## Loaders

### `pretrain`

Weighted multi-source pretraining, documents in random order. Each source is a directory
of JSONL files (`{"text": ...}` per line), e.g. the normalized documents that stage 1 of
`cluster_corpus/prepare.sh` writes. Used by `configs/mosaics/memory_mosaics_v2.yaml`.

```yaml
data:
  loader: pretrain
  root_dir: data_out/normalized
  sources:              # sub-directories of root_dir, with sampling weights
    cosmopedia: 3
    arxiv: 1
```

Documents are distributed across GPUs by byte range (below), so any number of files works with any number of GPUs.

### `cluster`

The loader of the *Open-world AI* configs. Each training sequence is packed from
documents of ONE cluster -- topically related documents, i.e. related tasks -- so that
memories reaching beyond the current document have something to retrieve. Build the
clustered corpus with `cluster_corpus/prepare.sh` (see `cluster_corpus/README.md`).

```yaml
data:
  loader: cluster
  cluster_data_dir: data_out/clustered_simple_k1200
  cluster_metadata: metadata_train.json   # held-out test clusters are excluded
  clusters_per_step: 32
```

**Schedule**: at every step, `clusters_per_step` clusters are drawn (with replacement,
probability proportional to cluster size, seeded by `data.seed`); sequence b of
data-parallel rank r reads cluster `(r * batch_size + b) % clusters_per_step`, so the
batch_size x world-size sequences share the draws evenly. Ranks reading the same cluster
read disjoint documents (each rank reads its own byte range of every file). With `cluster_frac: {n_clusters: n_slots, ...}` some slots pack
documents from several clusters instead (round-robin, then shuffled).

**Within a cluster**: the same pipeline as `pretrain`, over that cluster's JSONL files.
Pipelines are cached (`max_cached_clusters` open at once; an evicted one keeps its
position), prefetched in a background process, and checkpointed with the run, so a
resumed run continues the same document stream.

### `sft`

Chat-format fine-tuning (`{"messages": [{"role": ..., "content": ...}, ...]}` per line,
formatted by `tokenizer/chat_template.py`). Unlike pretraining it packs with `wrap=False`
(a conversation never spans two sequences) and a per-token loss mask keeps the loss on
assistant turns only.

```yaml
data:
  loader: sft
  sft_data_path: /path/to/train      # directory of JSONL files
```

## Key Concepts

**Packing: `.pack(seq_len, wrap, pad)`**

- `wrap=True` (pretrain): documents flow across sequence boundaries. Maximizes token utilization.
- `wrap=False` (SFT): if a conversation doesn't fit in remaining space, pad and start fresh. No mid-conversation splits.

**Byte-range sharding**

`JsonlSource(path, dp_rank, dp_world_size)` gives rank r of W the lines that START in bytes
[r S / W, (r+1) S / W) of each file of size S. Every line goes to exactly one rank, any
number of files works with any number of GPUs (8 shard files across 256 GPUs), and each
rank reads only about 1/W of the data. A rank whose share holds no line at all (a source
with fewer lines than ranks) reads the whole source instead, with a warning.

**Resuming**

Each stage's `state_dict()` covers the stages before it: file and byte offset, shuffle
buffer and RNG, the partly packed document, the sampling RNG. The state differs per rank,
so it is saved beside the checkpoint, one file per rank (`data_state_dp<r>.pkl` for
pretrain / SFT, `pipeline_states_dp<r>.pkl` for the cluster loader). Resuming with a
different number of GPUs is not supported: the shares would differ.

**Batch format**

All loaders output `TrainBatch(x, y, mask, src_names)`:
- `x`: `(B, seq_len)` int64 numpy — input tokens
- `y`: `(B, seq_len)` int64 numpy — target tokens
- `mask`: `(B, seq_len)` float32 numpy (0/1) or None — per-token loss mask (None = all tokens)
- `src_names`: list of source identifiers (for logging)
