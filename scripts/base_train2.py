#!/usr/bin/env python3
"""
Train model. Run as:

python scripts/base_train2.py

Distributed:
torchrun --nproc_per_node=8 scripts/base_train.py

CPU/Mac quick tiny run example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

New flags in this edition:
  --use_hf                # stream data from Hugging Face mixtures (see --hf_mixture)
  --hf_mixture MIX        # YAML string OR path describing HF datasets+weights
"""
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import random
import yaml
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterator, Dict, List, Optional

import wandb
import torch

# optional dependency only used when --use_hf
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings (original flags preserved)
run = "dummy"
device_type = ""  # cuda|cpu|mps (empty => autodetect)
depth = 20
max_seq_len = 2048
num_iterations = -1
target_flops = -1.0
target_param_data_ratio = 20
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
eval_every = 250
eval_tokens = 20*524288
core_metric_every = 2000
core_metric_max_per_task = 500
sample_every = 2000
model_tag = ""

# ---- New flags for HF streaming ----
use_hf = False
hf_mixture = "base_mixture.yaml"  # YAML string or path
hf_seed = 42     # controls mixture sampling order

# allow CLI to override
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer (keep nanochat’s tokenizer so eval & sampling work unchanged)
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs derived from depth
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# tokens per step math
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model
use_compile = (device_type == "cuda") and os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1"
if use_compile: # we run this on cuda only
    try:
        model = torch.compile(model, dynamic=False)
    except Exception as e:
        print0(f"[warn] torch.compile disabled (reason: {e})")
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# horizon
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Optimizers
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# -----------------------------------------------------------------------------
# HF streaming mixture (optional)
@dataclass
class MixSpec:
    name: str
    weight: float
    split: str = "train"
    text_key: str = "text"     # or "auto"
    preproc: Optional[str] = None

def _parse_hf_mixture(arg: str) -> List[MixSpec]:
    if not arg:
        return []
    if os.path.exists(arg):
        obj = yaml.safe_load(open(arg))
        items = obj.get("mixture", obj)
    else:
        obj = yaml.safe_load(arg)
        items = obj.get("mixture", obj)
    out = []
    for it in items:
        out.append(MixSpec(
            name=it["name"],
            weight=float(it.get("weight", 1.0)),
            split=it.get("split", "train"),
            text_key=it.get("text_key", "text"),
            preproc=it.get("preproc")
        ))
    return out

def _section_header(ex):
    h = ex.get("section_title")
    t = ex.get("text", "")
    if h and isinstance(h, str) and h.strip():
        return f"== {h} ==\n{t}"
    return t

def _mid_auto_text(ex):
    task = ex.get("task")
    if task == "bulletize":
        tgt = ex.get("target") or []
        if isinstance(tgt, list):
            bullets = "• " + "\n• ".join([str(x) for x in tgt[:4]])
        else:
            bullets = str(tgt)
        src = ex.get("source") or {}
        title = src.get("title", "")
        url = src.get("url", "")
        hdr = (title + "\n") if title else ""
        cite = f"\n[{url}]" if url else ""
        return hdr + bullets + cite
    if task == "qa_grounded":
        ans = ex.get("answer", "")
        cites = ex.get("citations") or []
        cite = f" [{cites[0]}]" if cites else ""
        return str(ans) + cite
    if task in ("tz_math", "date_range", "distance_km"):
        return f"{task} | {ex.get('input')} -> {ex.get('target')}"
    return ex.get("text")

def _encode_with_nanochat(tok, text: str) -> List[int]:
    """
    nanochat tokenizer is callable; fall back to .encode if present.
    returns python list of token ids
    """
    try:
        ids = tok(text)  # many nanochat tokenizers implement __call__
    except TypeError:
        ids = tok.encode(text)  # fallback
    # ensure list[int]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return list(ids)

def text_stream_from_mixture(specs: List[MixSpec], seed: int = 1234) -> Iterator[str]:
    assert load_dataset is not None, "pip install datasets"
    rnd = random.Random(seed + ddp_rank)  # shard sampling order per-rank
    buckets = []
    for s in specs:
        ds = load_dataset(s.name, split=s.split, streaming=True)
        buckets.append((s, iter(ds)))
    weights = [s.weight for s, _ in buckets]
    probs = [w/sum(weights) for w in weights]

    while True:
        i = rnd.choices(range(len(buckets)), weights=probs, k=1)[0]
        spec, it = buckets[i]
        ex = next(it, None)
        if ex is None:
            # refresh iterator (some remotes yield finite streams)
            buckets[i] = (spec, iter(load_dataset(spec.name, split=spec.split, streaming=True)))
            continue
        if spec.text_key == "auto":
            s = _mid_auto_text(ex)
        else:
            s = ex.get(spec.text_key)
        if not s:
            continue
        if spec.preproc == "section_header":
            ex2 = dict(ex)
            ex2["text"] = s
            s = _section_header(ex2)
        yield str(s).strip()

def pack_tokens(text_iter: Iterator[str], seq_len: int) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Concatenate-then-chunk; yields (x, y) with x.shape=(B, T), y same, on CPU.
    Caller will move to device.
    """
    buf: List[int] = []
    for txt in text_iter:
        ids = _encode_with_nanochat(tokenizer, txt)
        if hasattr(tokenizer, "eos_id"):
            # prefer <|eot|> if your tokenizer has it, else fall back to eos_id
            if "<|eot|>" in getattr(tokenizer, "special_tokens", []) or "<|eot|>" in str(tokenizer):
                eot = _encode_with_nanochat(tokenizer, "<|eot|>")[-1]
                buf.extend(ids + [eot])
            else:
                buf.extend(ids + [tokenizer.eos_id])
        else:
            buf.extend(ids)
        while len(buf) >= seq_len + 1:
            x = torch.tensor(buf[:seq_len], dtype=torch.long)
            y = torch.tensor(buf[1:seq_len+1], dtype=torch.long)
            yield {"x": x, "y": y}
            buf = buf[seq_len:]

def build_hf_loader(device_batch_size: int, seq_len: int, specs: List[MixSpec]) -> Iterator[torch.Tensor]:
    """
    Returns an iterator that yields (x, y) already stacked to (B, T).
    """
    text_iter = text_stream_from_mixture(specs, seed=hf_seed)
    pack_iter = pack_tokens(text_iter, seq_len=seq_len)
    batch_x, batch_y = [], []
    while True:
        ex = next(pack_iter)
        batch_x.append(ex["x"])
        batch_y.append(ex["y"])
        if len(batch_x) == device_batch_size:
            x = torch.stack(batch_x, dim=0)
            y = torch.stack(batch_y, dim=0)
            batch_x, batch_y = []
            yield x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Initialize DataLoaders
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")

if use_hf:
    specs = _parse_hf_mixture(hf_mixture)
    assert specs, "--use_hf set but --hf_mixture is empty."
    print0("[data] Using Hugging Face streaming mixture:")
    for s in specs:
        print0(f"  - {s.name} (split={s.split}, weight={s.weight}, text_key={s.text_key}, preproc={s.preproc})")

    # train stream
    train_loader = build_hf_loader(device_batch_size, max_seq_len, specs)

    # validation stream (reuse same mix but independent iterator)
    def build_val_loader():
        return build_hf_loader(device_batch_size, max_seq_len, specs)
    x, y = next(train_loader)
else:
    train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
    build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
    x, y = next(train_loader)  # prime

# -----------------------------------------------------------------------------
# Schedulers (unchanged)
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / max(1, warmup_iters)
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / max(1, warmdown_iters)
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop (unchanged)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break

    # --------- one training step ----------
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)  # prefetch next
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    dt = time.time() - t0
    # --------------------------------------

    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# wrap-up
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
    },
    {
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

wandb_run.finish()
compute_cleanup()
