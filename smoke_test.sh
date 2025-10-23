#!/usr/bin/env bash
set -euo pipefail

# Minimal-cost sanity run for tinychat:
# - trains a tiny tokenizer on ~0.2M chars
# - base-trains a TINY model for a handful of steps
# - mid-trains & SFT for a handful of steps
# - runs a tiny accuracy eval
#
# Expected wall time: ~10–20 minutes on a single recent GPU (or CPU/mps, slower).
# Cost: negligible. Good for CI/smoke tests.

# ---------- knobs you can tweak ----------
export WANDB_RUN="${WANDB_RUN:-dummy}"          # keep "dummy" to skip wandb network calls
export DEVICE_TYPE="${DEVICE_TYPE:-}"            # "", "cuda", "mps", or "cpu"
export HF_TOKEN_OK="${HF_TOKEN_OK:-0}"           # set 1 if HUGGINGFACE_TOKEN is already exported
export USE_HF_BASE="${USE_HF_BASE:-0}"           # 1 to stream tiny HF base mixture
export USE_HF_MID="${USE_HF_MID:-0}"             # 1 to stream tiny HF mid mixture
export USE_HF_SFT="${USE_HF_SFT:-0}"             # 1 to stream tiny HF SFT dataset
export SFT_REPO_ID="${SFT_REPO_ID:-aveekmukherjee/travel-sft-eu-india}"

# micro configs (keep tiny!)
SEQ_LEN=256
DEV_BS=1
TOT_BS=$((DEV_BS * SEQ_LEN * 8))                 # ~2K tokens/batch; 8 micro-batches world=1
BASE_STEPS=10
MID_STEPS=20
SFT_STEPS=20
EVAL_TOKENS=$((SEQ_LEN * DEV_BS * 32))          # ~8k tokens for quick bpb/eval if needed

# ---------- environment & deps ----------
export OMP_NUM_THREADS=1
export TINYCHAT_BASE_DIR="$HOME/.cache/tinychat"
export NANOCHAT_BASE_DIR="$TINYCHAT_BASE_DIR"   # nanochat reads this one
mkdir -p "$NANOCHAT_BASE_DIR"

# venv via uv (fast)
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# Optional HF auth (only if you set HF_TOKEN_OK=1 and exported the token)
if [[ "$HF_TOKEN_OK" == "1" ]]; then
  echo "[info] Using HUGGINGFACE_TOKEN from environment"
else
  echo "[info] Skipping HF auth (private datasets will be unavailable)."
fi

# write report header (purely cosmetic)
python -m nanochat.report reset

# ---------- tokenizer (tiny) ----------
# Rust BPE build (noop if already built)
# Ensure maturin is available
if ! command -v maturin &>/dev/null; then uv tool install maturin || true; fi
# Build the rustbpe Tokenizer (only if no build artifacts exist)
if compgen -G "rustbpe/target/release/*rustbpe*" > /dev/null; then
  echo "[info] rustbpe already built."
else
  # fall back to uvx if maturin isn't on PATH
  if command -v maturin &>/dev/null; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
  else
    uvx maturin develop --release --manifest-path rustbpe/Cargo.toml
  fi
fi

# download only 1 shard and train a tiny tokenizer on ~200k chars
python -m nanochat.dataset -n 1

# >>> IMPORTANT: stage the single shard into train/ and val/ BEFORE tok_train/eval
DATA_ROOT="${NANOCHAT_BASE_DIR}/base_data"
mkdir -p "${DATA_ROOT}/train" "${DATA_ROOT}/val"
# If shard(s) are at base_data/*.parquet, mirror one into both splits
if ls "${DATA_ROOT}"/*.parquet >/dev/null 2>&1; then
  for f in "${DATA_ROOT}"/*.parquet; do
    [ -f "${DATA_ROOT}/train/$(basename "$f")" ] || cp "$f" "${DATA_ROOT}/train/"
    [ -f "${DATA_ROOT}/val/$(basename "$f")" ]   || cp "$f" "${DATA_ROOT}/val/"
  done
fi

# Quick sanity: ensure at least one file is visible to the split readers
echo "[debug] train shards:"; ls -lh "${DATA_ROOT}/train" || true
echo "[debug] val shards:";   ls -lh "${DATA_ROOT}/val"   || true


python -m scripts.tok_train --max_chars=200000
if ls "${DATA_ROOT}/train"/*.parquet >/dev/null 2>&1; then
  python -m scripts.tok_eval
else
  echo "[warn] No train shards visible to tok_eval; skipping tokenizer eval."
fi


# ---------- base (tiny) ----------
# eval bundle (for core metric if enabled; we’ll keep core metric off to save time)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$TINYCHAT_BASE_DIR/eval_bundle" ]; then
  curl -L -o eval_bundle.zip "$EVAL_BUNDLE_URL"
  unzip -q eval_bundle.zip && rm -f eval_bundle.zip
  mv eval_bundle "$TINYCHAT_BASE_DIR"
fi

# Optional HF mixture file (if using HF for base)
if [[ "${USE_HF_BASE}" == "1" ]]; then
  cat >"$TINYCHAT_BASE_DIR/base_mixture_smoke.yaml" <<'YAML'
mixture:
  - {name: "aveekmukherjee/wikivoyage-eu-india-sections", weight: 0.5, text_key: text, preproc: section_header}
  - {name: "aveekmukherjee/wikipedia-travel-eu-india",    weight: 0.5, text_key: text, preproc: section_header}
YAML
  BASE_FLAGS="--use_hf --hf_mixture $TINYCHAT_BASE_DIR/base_mixture_smoke.yaml"
else
  BASE_FLAGS=""
fi

# train a *tiny* model: depth=4, short seq, min steps, tiny batches
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --depth=4 \
  --max_seq_len="$SEQ_LEN" \
  --device_batch_size="$DEV_BS" \
  --total_batch_size="$TOT_BS" \
  --num_iterations="$BASE_STEPS" \
  --eval_tokens="$EVAL_TOKENS" \
  --core_metric_every=-1 \
  --sample_every=1000000 \
  --run="$WANDB_RUN" \
  ${BASE_FLAGS}

# quick loss eval (uses same tiny settings)
torchrun --standalone --nproc_per_node=1 -m scripts.base_loss -- \
  --max_seq_len="$SEQ_LEN" \
  --device_batch_size="$DEV_BS" \
  --eval_tokens="$EVAL_TOKENS"

# ---------- mid (tiny) ----------
# If not using HF, the stock script wants identity_conversations.jsonl
if [[ "${USE_HF_MID}" != "1" ]]; then
  curl -L -o "$TINYCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
  MID_FLAGS=""
else
  cat >"$TINYCHAT_BASE_DIR/mid_mixture_smoke.yaml" <<'YAML'
mixture:
  - {name: "aveekmukherjee/wikivoyage-eu-india-sections", weight: 0.7, text_key: text, preproc: section_header}
  - {name: "aveekmukherjee/wikipedia-travel-eu-india",    weight: 0.3, text_key: text, preproc: section_header}
YAML
  MID_FLAGS="--use_hf --hf_mixture $TINYCHAT_BASE_DIR/mid_mixture_smoke.yaml"
fi

torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- \
  --device_batch_size="$DEV_BS" \
  --total_batch_size="$TOT_BS" \
  --max_seq_len="$SEQ_LEN" \
  --num_iterations="$MID_STEPS" \
  --eval_every=50 \
  --eval_tokens="$EVAL_TOKENS" \
  --run="$WANDB_RUN" \
  ${MID_FLAGS}

# tiny chat eval (one task, few problems)
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- \
  -i mid -a ARC-Easy -x 16 -b 4 --device-type "${DEVICE_TYPE}"

# ---------- SFT (tiny) ----------
if [[ "${USE_HF_SFT}" == "1" ]]; then
  SFT_FLAGS="--use_hf_sft --sft_repo_id $SFT_REPO_ID --sft_split train"
else
  SFT_FLAGS=""   # uses the local TaskMixture fallback in chat_sft.py
fi

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  --device_batch_size="$DEV_BS" \
  --total_batch_size="$TOT_BS" \
  --max_seq_len="$SEQ_LEN" \
  --num_iterations="$SFT_STEPS" \
  --eval_every=50 \
  --run="$WANDB_RUN" \
  ${SFT_FLAGS}

# tiny eval again
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- \
  -i sft -a ARC-Easy -x 16 -b 4 --device-type "${DEVICE_TYPE}"

echo "[OK] smoke test completed."
