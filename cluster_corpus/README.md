# Cluster corpus

Builds the clustered corpus the *Open-world AI* models train on: documents grouped into
topically coherent clusters, so that a packed training sequence is a set of *related
tasks*. Its first stage, the normalized documents, is also the plain pretraining data of
Memory Mosaics v2.

## Pipeline

```
normalize corpus ── embedding per document --  clustering  -- organize documents
```

| | module | in → out (under `data_out/`) |
|---|---|---|
| 1 | `cluster_corpus.normalize` | each source → `normalized/<source>/shard_*.jsonl` |
| 2 | `cluster_corpus.embed` | documents → `embeddings/<source>/shard_*.npy` (nomic-embed-text-v1.5, 768-d) |
| 3 | `cluster_corpus.cluster all` | embeddings → `centroids.npy`, `labels_*.npy` (FAISS spherical k-means) |
| 4 | `cluster_corpus.organize all` | labels → `cluster_XXXX/*.jsonl`, `metadata{,_train,_test}.json` |

`prepare.sh` runs the four stages; every stage is seeded and deterministic.

## Usage

```bash
conda install -c pytorch -c nvidia "faiss-gpu>=1.13"   # first, and >=1.13 (built for numpy 2); see ../README.md#setup
                                                       # (or, CPU only: pip install faiss-cpu)
pip install -r requirements.txt
bash cluster_corpus/prepare.sh tiny --go       # minutes: check that the code runs
bash cluster_corpus/prepare.sh simple          # print the commands (dry run)
bash cluster_corpus/prepare.sh simple --go     # run them
```

Output goes to `data_out/` (`--root_dir` or `$DATA_ROOT` to change it).

|  | `tiny` | `simple` | `exact` |
|---|---|---|---|
| sources | first 50,000 Cosmopedia docs | Cosmopedia | 6-source mixture |
| documents | 50 k | ~31 M | ~49 M |
| clusters | 20 | 1200 | 1200 |
| test clusters held out | 5 | 50 | 50 |

`tiny` is a smoke test of the code, not training data. Embedding (stage 2) is the
bottleneck; `prepare.sh` prints how to split it across GPUs.

`exact` reads two sources from local dumps: stage `EleutherAI/proof-pile-2` (configs
`arxiv`, `open-web-math`) under `PROOF_PILE_ROOT` (default `data_out/raw`) as
`proof_pile_2_arxiv_train/*.jsonl[.zst]` and `proof_pile_2_open_web_math_train/*.jsonl[.zst]`.
The other sources download from HuggingFace. ShareGPT is unfiltered user-generated
content and may contain offensive material.

## Using the result

The clustered corpus, with the `cluster` loader (training excludes the held-out test
clusters):

```yaml
data:
  loader: cluster
  cluster_data_dir: data_out/clustered_simple_k1200     # or clustered_exact_k1200
  cluster_metadata: metadata_train.json
```

The normalized documents of stage 1, with the `pretrain` loader (plain pretraining, as in
`memory_mosaics_v2.yaml`; these include the documents of the held-out clusters):

```yaml
data:
  loader: pretrain
  root_dir: data_out/normalized
  sources:
    cosmopedia: 1          # exact profile: one entry per source
```
