#!/usr/bin/env bash
set -euo pipefail

# Minimal-cost sanity run for nanochat on a single GPU:
# - trains a tiny tokenizer on ~0.2M chars
# - base-trains a TINY model a few steps
# - mid-trains from HF streaming mixture
# - SFT on your HF SFT dataset
# - quick ARC-Easy evals after mid & SFT
#
# Expected wall time: ~10–20 minutes on a recent GPU. Cost: negligible.

# ---------- knobs you can tweak ----------
export WANDB_RUN="${WANDB_RUN:-dummy}"            # keep "dummy" to skip wandb network calls
export DEVICE_TYPE="${DEVICE_TYPE:-}"              # "", "cuda", "mps", or "cpu" (empty => autodetect)
export HF_TOKEN_OK="${HF_TOKEN_OK:-0}"             # set 1 if HUGGINGFACE_TOKEN / HF_TOKEN is already exported
export USE_HF_BASE="${USE_HF_BASE:-0}"             # (unused in this smoke; base uses local parquet)
export USE_HF_MID="${USE_HF_MID:-1}"               # 1 to stream tiny HF mid mixture (recommended)
export USE_HF_SFT="${USE_HF_SFT:-1}"               # 1 to stream tiny HF SFT dataset (recommended)
export SFT_REPO_ID="${SFT_REPO_ID:-aveekmukherjee/travel-sft-eu-india}"

# micro configs (keep tiny!)
SEQ_LEN=256
DEV_BS=1
TOT_BS=$((SEQ_LEN * DEV_BS * 8))                   # ~2K tokens/batch; 8 micro-batches world=1
BASE_STEPS=10
MID_STEPS=20
SFT_STEPS=20
EVAL_TOKENS=$((SEQ_LEN * DEV_BS * 32))            # ~8k tokens for quick bpb/eval

# ---------- environment & deps ----------
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# venv via uv (fast)
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# Optional: HF auth
if [[ "$HF_TOKEN_OK" == "1" ]] && [[ -n "${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}" ]]; then
  python - <<'PY'
import os
tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if tok:
  try:
    from huggingface_hub import HfFolder
    HfFolder.save_token(tok)
    print("[info] Saved HF token to cache.")
  except Exception as e:
    print("[warn] Could not save HF token:", e)
else:
  print("[warn] HF_TOKEN_OK=1 but no token env var found.")
PY
  echo "[info] Using HUGGINGFACE_TOKEN/HF_TOKEN from environment"
else
  echo "[info] Skipping HF auth (private datasets may be unavailable)."
fi

# write report header (purely cosmetic)
python -m nanochat.report reset

# ---------- tokenizer (tiny) ----------
# Build rustbpe only if needed
if ! command -v maturin &>/dev/null; then uv tool install maturin; fi
if compgen -G "rustbpe/target/release/*rustbpe*" > /dev/null; then
  echo "[info] rustbpe already built."
else
  uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Grab 1 shard of base data (has a 'text' column)
python -m nanochat.dataset -n 1
DATA_ROOT="${NANOCHAT_BASE_DIR}/base_data"

echo "[debug] DATA_ROOT=${DATA_ROOT}"
echo "[debug] train files:"; ls -lh "${DATA_ROOT}/train" || true
echo "[debug] val files:";   ls -lh "${DATA_ROOT}/val"   || true

# Train a tiny tokenizer on ~200k chars
python -m scripts.tok_train --max_chars=200000
# Quick tokenizer eval (needs at least one parquet in train/)
if ls "${DATA_ROOT}/train"/*.parquet >/dev/null 2>&1; then
  python -m scripts.tok_eval || echo "[warn] tok_eval failed; continuing"
else
  echo "[warn] No train shards visible to tok_eval; skipping tokenizer eval."
fi

# ---------- base (tiny) ----------
# eval bundle for possible later use (cached)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
  curl -L -o eval_bundle.zip "$EVAL_BUNDLE_URL"
  unzip -q eval_bundle.zip && rm -f eval_bundle.zip
  mv eval_bundle "$NANOCHAT_BASE_DIR"
fi

# Train tiny base using base_train2.py (the one that worked)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train2 -- \
  --depth 4 \
  --max_seq_len $SEQ_LEN \
  --device_batch_size $DEV_BS \
  --total_batch_size $TOT_BS \
  --num_iterations $BASE_STEPS \
  --eval_tokens $EVAL_TOKENS \
  --core_metric_every -1 \
  --sample_every 1000000 \
  --run $WANDB_RUN

# Quick loss eval on the trained base checkpoint
torchrun --standalone --nproc_per_node=1 -m scripts.base_loss -- \
  --device_batch_size $DEV_BS \
  --eval_tokens $EVAL_TOKENS

# ---------- mid (tiny, HF stream) ----------
# Build tiny mid mixture YAML if using HF
MID_FLAGS=""
if [[ "${USE_HF_MID}" == "1" ]]; then
  cat >"$NANOCHAT_BASE_DIR/mid_mixture.yaml" <<'YAML'
YAML
  MID_FLAGS="--use_hf=1 --hf_mixture $NANOCHAT_BASE_DIR/mid_mixture.yaml"
fi

torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- \
  --device_batch_size $DEV_BS \
  --total_batch_size $TOT_BS \
  --max_seq_len $SEQ_LEN \
  --num_iterations $MID_STEPS \
  --eval_every 50 \
  --eval_tokens $EVAL_TOKENS \
  --run $WANDB_RUN \
  $MID_FLAGS

# Tiny chat eval after mid (explicit device type is required by chat_eval)
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- \
  -i mid -a ARC-Easy -x 16 -b 4 --device-type "${DEVICE_TYPE:-cuda}"

# ---------- SFT (tiny, HF stream) ----------
SFT_FLAGS=""
if [[ "${USE_HF_SFT}" == "1" ]]; then
  SFT_FLAGS="--use_hf_sft=1 --sft_repo_id $SFT_REPO_ID --sft_split train"
fi

# (Temporary guard) Older chat_sft.py expected a generator and not a factory in one spot.
# Patch line once if needed (idempotent). Safe no-op if already patched or code differs.
set +e
grep -q 'train_iter = iter(train_loader()' scripts/chat_sft.py || \
  sed -i 's/train_iter = iter(train_loader)/train_iter = iter(train_loader() if callable(train_loader) else train_loader)/' scripts/chat_sft.py
set -e

# Some versions also use max_seq_len implicitly inside validation helpers. Export here if needed.
export max_seq_len="$SEQ_LEN"

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  --device_batch_size $DEV_BS \
  --num_iterations $SFT_STEPS \
  --eval_every 50 \
  --run $WANDB_RUN \
  $SFT_FLAGS

# Tiny chat eval after SFT
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- \
  -i sft -a ARC-Easy -x 16 -b 4 --device-type "${DEVICE_TYPE:-cuda}"

echo "[OK] test completed."