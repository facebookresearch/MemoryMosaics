#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.
# Build the clustered training corpus, end to end.
#
#   bash cluster_corpus/prepare.sh simple                              # DRY RUN: print the commands
#   bash cluster_corpus/prepare.sh simple --go                         # run them
#   bash cluster_corpus/prepare.sh exact  --go --root_dir /scratch/mm  # write somewhere else
#   bash cluster_corpus/prepare.sh tiny   --go --root_dir /tmp/mm      # minutes: test the code
#
# Three profiles, ONE pipeline. They differ only in which sources step 1 pulls and how
# many documents it keeps (plus k and the test split); steps 2-4 run the same code, so
# a `simple` reproduction exercises the same code as the paper.
#
#   tiny     First 50,000 Cosmopedia documents (khanacademy, then openstax: 2 shards),
#            k=20, 5 test clusters. A smoke test of the whole pipeline (two small
#            downloads, minutes on one GPU), not data to train on. EMBED_DEVICE=cpu EMBED_DTYPE=fp32 runs without a GPU, but takes
#            hours at this size.
#   simple   FULL Cosmopedia (~31M documents), k=1200 -- the paper's setup with a single
#            source. Still a cluster-scale job (~60% of `exact`'s embedding cost), but it
#            is what backs the paper's claim that Cosmopedia alone gives the same
#            conclusions. Numbers will not match the 6-source tables.
#   exact    The 6-source mixture the paper used, ~49M documents.
#            Embedding is the bottleneck: ~256 GPU-hours.
#
# OUTPUT ROOT. Everything below is written under one directory, in precedence order:
#   --root_dir <path>   this flag
#   $DATA_ROOT          environment variable
#   <repo>/data_out     default (resolved from this script, NOT from your cwd)
# The default is deliberately NOT cluster_corpus/ -- that is this pipeline's source code, and the
# corpus (hundreds of GB to several TB) does not belong in a git checkout at all. Point
# --root_dir at scratch or other bulk storage.
#
# Pipeline (identical for every profile), all of it under the output root:
#   1 cluster_corpus.normalize          sources     -> normalized/<source>/shard_*.jsonl   (+ --frac)
#   2 cluster_corpus.embed              each shard  -> embeddings/<source>/shard_*.npy  (1:1)
#   3 cluster_corpus.cluster all          embeddings  -> centroids.npy, pool_info.json, labels_*.npy
#   4 cluster_corpus.organize all          labels      -> cluster_XXXX/*.jsonl, metadata{,_train,_test}.json
#
# Stages 3 and 4 each split into sub-commands (fit/assign, write/split) when you want to
# fan the assignment out per GPU, or re-split without regenerating the cluster files.
#
# PROOF_PILE_ROOT (exact profile only) defaults to <root>/raw; export it to read the
# locally staged proof-pile-2 dumps from elsewhere. WORKERS (default 16) sets stage 1's
# per-source parallelism; an interrupted stage 1 resumes on re-run. EMBED_DEVICE
# (default cuda) and EMBED_DTYPE (default bf16) override stage 2's device and precision.

set -euo pipefail

# ── output style ───────────────────────────────────────────────────────────────
# This script's own lines stand out from the tools' logs (HF downloads, tqdm, ...):
# stage headers bold cyan, everything else it prints (notes, commands) bold, errors
# bold red. Plain when the stream is not a terminal -- e.g. piped into a log file --
# or NO_COLOR is set. Decided once here: `[ -t 1 ]` inside $(...) is always false.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    S_HEAD=$'\033[1;36m'; S_MSG=$'\033[1m'; S_OFF=$'\033[0m'
else
    S_HEAD=""; S_MSG=""; S_OFF=""
fi
if [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then S_ERR=$'\033[1;31m'; S_EOFF=$'\033[0m'; else S_ERR=""; S_EOFF=""; fi
header() { printf '\n%s%s%s\n' "${S_HEAD}" "$*" "${S_OFF}"; }
say()    { printf '%s%s%s\n' "${S_MSG}" "$*" "${S_OFF}"; }
err()    { printf '%sERROR: %s%s\n' "${S_ERR}" "$*" "${S_EOFF}" >&2; }

usage() { sed -n '2,45p' "$0"; exit "${1:-1}"; }

PROFILE=""
GO=0
ROOT_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --go)        GO=1; shift;;
        --root_dir)  [ $# -ge 2 ] || { err "--root_dir needs a path"; exit 1; }
                     ROOT_FLAG="$2"; shift 2;;
        --root_dir=*) ROOT_FLAG="${1#*=}"; shift;;
        -h|--help)   usage 0;;
        tiny|simple|exact) [ -z "${PROFILE}" ] || { err "profile given twice ('${PROFILE}', '$1')"; exit 1; }
                     PROFILE="$1"; shift;;
        *)           err "unknown argument '$1'"; usage;;
    esac
done
[ -n "${PROFILE}" ] || usage

# Default root is resolved from THIS SCRIPT's location, so the output lands in the
# same place no matter which directory you invoke it from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT_FLAG:-${DATA_ROOT:-${REPO_ROOT}/data_out}}"
case "${ROOT}" in /*) ;; *) ROOT="$(pwd)/${ROOT}";; esac   # relative -> absolute
# The stages run as `python -m cluster_corpus.<stage>`, resolved from the cwd.
cd "${REPO_ROOT}"
NORM="${ROOT}/normalized"
EMB="${ROOT}/embeddings"

# ── profile settings ───────────────────────────────────────────────────────────
# K_CLUSTERS / SEED / NITER / MAX_PTS and the per-source sample_frac below are the
# values recorded in the paper run's pool_info.json.
CLUSTER_FLAGS=()
if [ "${PROFILE}" = "tiny" ]; then
    # A code test, not a dataset: same stages, a few minutes end to end on one GPU.
    # Stage 1 writes one shard per upstream file, and --limit takes a prefix, so the
    # subset order decides how many shards there are. Every file holds 63k+ rows except
    # khanacademy's single 24,123-row file; khanacademy then openstax makes 50,000 docs
    # span 2 shards (24,123 + 25,877), so the multi-shard paths get exercised.
    SOURCES=(cosmopedia)
    COSMO_SUBSETS="khanacademy,openstax"
    LIMIT_cosmopedia=50000
    K_CLUSTERS=20
    MAX_PTS=512
    N_TEST=5
    CLUSTERED="${ROOT}/clustered_tiny_k${K_CLUSTERS}"
    CLUSTER_FLAGS=(--cpu)                 # 50k points: no faiss-gpu needed

elif [ "${PROFILE}" = "simple" ]; then
    # The paper's setup with a single source: FULL Cosmopedia, no --limit, same k and
    # seeds as `exact`. One source means both fractions are vacuous -- there is nothing
    # to re-weight against -- so neither --sample_frac nor --frac is passed.
    SOURCES=(cosmopedia)
    COSMO_SUBSETS=""                      # empty -> all 8 subsets, ~31M docs
    K_CLUSTERS=1200
    MAX_PTS=512
    N_TEST=50
    CLUSTERED="${ROOT}/clustered_simple_k${K_CLUSTERS}"

else
    SOURCES=(arxiv cosmopedia gutenberg nemotron_cc_high_10m open-web-math sharegpt)
    COSMO_SUBSETS=""                      # all 8 subsets
    LIMIT_nemotron_cc_high_10m=10000000   # ONLY this source is capped; see README
    K_CLUSTERS=1200
    MAX_PTS=512
    # Source mixture, applied ONCE at step 1. Everything downstream then operates on the
    # whole normalized pool, so the clustering and the training set cannot be built from
    # different mixtures. Sources not listed default to 1.0.
    FRAC_arxiv=0.25
    FRAC_gutenberg=0.75
    FRAC_open_web_math=0.333
    N_TEST=50                            
    CLUSTERED="${ROOT}/clustered_exact_k${K_CLUSTERS}"
    # arxiv / open-web-math read LOCAL proof-pile-2 dumps under PROOF_PILE_ROOT:
    #   proof_pile_2_{arxiv,open_web_math}_train/*.jsonl[.zst]
    PROOF_PILE_ROOT="${PROOF_PILE_ROOT:-${ROOT}/raw}"
fi
SEED=1            # cluster_pool --seed  (pool_info.json: seed)
NITER=25          # cluster_pool --niter (pool_info.json: niter)
FRAC_SEED=1       # normalize_sources --frac_seed (recorded in each _manifest.json)
SPLIT_SEED=42     # split_train_test --seed
# Stage 1 parallelism WITHIN a source (sources still run one after another): input
# files processed at once for cosmopedia, nemotron, arxiv, open-web-math. For the HF
# ones each worker holds one ~250 MB download, so extra disk is ~WORKERS x 250 MB.
WORKERS="${WORKERS:-16}"
EMBED_MODEL="nomic-ai/nomic-embed-text-v1.5"     # 768-d, matches pool_info.json dim
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
EMBED_DTYPE="${EMBED_DTYPE:-bf16}"

run() {
    # Shell-quoted, so a printed (dry-run) command can be pasted as is -- e.g. the
    # prompt keeps its trailing space.
    printf '%s ' "${S_MSG}"; printf ' %q' "$@"; printf '%s\n' "${S_OFF}"
    [ "${GO}" -eq 1 ] && "$@"
    return 0
}

header "=== profile: ${PROFILE}   root: ${ROOT}   mode: $([ "${GO}" -eq 1 ] && echo RUN || echo 'DRY RUN (add --go)') ==="

# ── 1. normalize ───────────────────────────────────────────────────────────────
header "--- 1. normalize sources -> ${NORM} ---"
for src in "${SOURCES[@]}"; do
    args=(python -m cluster_corpus.normalize --source "${src}" --out_dir "${NORM}" --workers "${WORKERS}")
    [ "${src}" = "cosmopedia" ] && [ -n "${COSMO_SUBSETS}" ] && args+=(--subsets "${COSMO_SUBSETS}")
    # arxiv / open-web-math read locally staged proof-pile-2 dumps; sharegpt fetches
    # itself from HuggingFace on first use.
    case "${src}" in
        arxiv|open-web-math) args+=(--raw_dir "${PROOF_PILE_ROOT}");;
    esac
    lim_var="LIMIT_${src//-/_}"
    [ -n "${!lim_var:-}" ] && args+=(--limit "${!lim_var}")
    frac_var="FRAC_${src//-/_}"
    [ -n "${!frac_var:-}" ] && args+=(--frac "${!frac_var}" --frac_seed "${FRAC_SEED}")
    run "${args[@]}"
done

# ── 2. embed ───────────────────────────────────────────────────────────────────
# One .npy per stage-1 shard (shard_X.jsonl -> shard_X.npy), model loaded once per
# process. A finished file is skipped on re-run, so an interrupted stage resumes. Here
# every source runs in ONE process, shard after shard; with several GPUs, run the same
# command once per GPU with --worker_id g --n_workers G (or a SLURM array) instead --
# the split does not affect the output.
embed_args() {          # embed_args <source>  ->  EMBED_ARGS
    EMBED_ARGS=(python -m cluster_corpus.embed --jsonl_dir "${NORM}/$1" --output_dir "${EMB}/$1"
                --model_name "${EMBED_MODEL}" --prompt "clustering: "
                --dtype "${EMBED_DTYPE}" --device "${EMBED_DEVICE}" --max_seq_length 2048)
}
shard_range() {         # "336 shards (shard_00000.jsonl ~ shard_00335.jsonl)", or why not
    local files=("$1"/shard_*.jsonl)
    if [ -e "${files[0]}" ]; then
        echo "${#files[@]} shards ($(basename "${files[0]}") ~ $(basename "${files[${#files[@]}-1]}"))"
    else
        echo "the shards of $1 (not there yet -- stage 1 writes them)"
    fi
}

header "--- 2. embed -> ${EMB} (one .npy per shard, ${EMBED_DTYPE} on ${EMBED_DEVICE}, 2048 tokens) ---"
embed_args "${SOURCES[0]}"
EX="$(printf ' %q' "${EMBED_ARGS[@]}")"
printf '%s' "${S_MSG}"
cat <<EOF
  Runs sequentially below: one process per source, shard after shard. To use G GPUs,
  run the same command once per GPU with --worker_id g --n_workers G instead (a shard
  goes to the same .npy whichever worker embeds it; finished shards are skipped).
  E.g. for ${SOURCES[0]} on one 8-GPU node:
    for g in \$(seq 0 7); do CUDA_VISIBLE_DEVICES=\$g${EX} --worker_id \$g --n_workers 8 & done; wait
  or in a SLURM array script (#SBATCH --array=0-7, one GPU per task):
   ${EX} --worker_id \$SLURM_ARRAY_TASK_ID --n_workers 8
EOF
printf '%s' "${S_OFF}"
for src in "${SOURCES[@]}"; do
    echo; say "  ${src}: computing document embeddings for $(shard_range "${NORM}/${src}") sequentially:"
    embed_args "${src}"
    run "${EMBED_ARGS[@]}"
done

# ── 3. cluster ─────────────────────────────────────────────────────────────────
header "--- 3. cluster: fit k-means (k=${K_CLUSTERS}, seed=${SEED}) + assign every document ---"
run python -m cluster_corpus.cluster all \
    --embedding_root "${EMB}" --k_clusters "${K_CLUSTERS}" \
    --max_points_per_centroid "${MAX_PTS}" --seed "${SEED}" --niter "${NITER}" \
    --output_dir "${CLUSTERED}" ${CLUSTER_FLAGS[@]+"${CLUSTER_FLAGS[@]}"}

# ── 4. organize ────────────────────────────────────────────────────────────────
# No --frac here: the mixture was already applied at step 1, so everything in the
# normalized pool gets written.
header "--- 4. organize: write per-cluster jsonl + hold out ${N_TEST} test clusters ---"
run python -m cluster_corpus.organize all \
    --cluster_dir "${CLUSTERED}" --jsonl_root "${NORM}" \
    --output_dir "${CLUSTERED}" --num_chunks_per_cluster 1 \
    --seed "${FRAC_SEED}" --n_test "${N_TEST}" --split_seed "${SPLIT_SEED}"

header "Done. Point a training config at:"
say "    cluster_data_dir: ${CLUSTERED}"
say "    cluster_metadata: metadata_train.json"
if [ "${GO}" -eq 0 ]; then echo; say "(dry run -- nothing executed; re-run with --go)"; fi
