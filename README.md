# MLX Tiny Transformer

A decoder-only transformer built from scratch in [Apple MLX](https://github.com/ml-explore/mlx), trained on FineWeb-Edu on Apple Silicon. The goal is not a working language model for its own sake — it is understanding *where the hardware time goes*: FLOPs, memory bandwidth, model-FLOPs utilization (MFU), and the Chinchilla scaling laws.

> Part of [`scaling-from-scratch`](.) — one of three from-scratch ML-systems projects following the [frontier-lab learning path](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html). Technical framing from the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/).

## Architecture

Modern LLaMA-style recipe, every component explicit (no fused QKV, no black boxes):

- **Pre-norm** residual blocks with **RMSNorm**
- **RoPE** rotary positional embeddings
- **SwiGLU** feed-forward (2/3 rule to match a 4·d GELU block's parameter count)
- **Grouped-query attention** capable (`n_kv_heads < n_heads`), MHA by default
- **Flash-style** attention via `mx.fast.scaled_dot_product_attention`
- **Tied** embedding / LM-head weights
- **KV cache** for O(1)-per-token autoregressive decode

## Layout

```
config.py        ModelConfig/TrainConfig — params, FLOPs, scaling presets
model/
  rope.py        rotary embeddings
  ffn.py         SwiGLU / GELU feed-forward
  attention.py   MHA + GQA, flash-style SDPA, KV-cache hooks
  cache.py       KV cache for decoding
  transformer.py full model: embed -> N blocks -> norm -> LM head
data/
  tokenizer.py   tiktoken GPT-2 BPE (50257 -> padded 50304)
  download.py    stream + tokenize FineWeb-Edu to a uint16 .bin
  dataset.py     memory-mapped next-token batching
train.py         training loop with loss / step-time / tok-s / MFU logging
generate.py      prefill + incremental decode (greedy / top-p)
profile.py       roofline: achieved FLOP/s, bandwidth, MFU vs ridge point
scaling.py       Chinchilla fit: L(N,D) = E + A/N^a + B/D^b
```

## Setup

```bash
conda create -n mlx-transformer python=3.11 -y
conda activate mlx-transformer
pip install -r requirements.txt
```

## Usage

```bash
# 1. download a slice of FineWeb-Edu (uint16 .bin, mmapped at train time)
python -m data.download --tokens 100_000_000

# 2. train (logs loss, step_time_ms, tokens/sec, MFU every step)
python train.py --preset 1M --steps 5000

# 3. generate
python generate.py --prompt "The history of" --max-tokens 100 --strategy top-p

# 4. roofline profile across batch/seq configs
python profile.py --preset 10M

# 5. scaling-law sweep + fit
python scaling.py --sweep --presets 1M 3M 10M --budgets 50_000_000 150_000_000 400_000_000
python scaling.py --fit
```

## The metrics are the point

**MFU** (model FLOPs utilization) = achieved FLOP/s ÷ chip peak FLOP/s — the headline training-efficiency number. On an M3 Pro (~6.4 TFLOP/s fp16, ~150 GB/s, ridge ≈ 43 FLOP/byte), a healthy small model lands ~20–40%; it's **memory-bound** at small scale because the embedding table dominates (84% of params on the 1M preset, 25% on the 100M preset — `python config.py` prints the breakdown).

**Chinchilla scaling**: fit `L(N, D) = E + A/N^α + B/D^β` over a sweep of non-embedding param counts N and token budgets D. The target result — compute-optimal allocation scales as `N* ∝ C^0.5`, `D* ∝ C^0.5` (grow model and data together).

## Hardware

Developed on Apple M3 Pro (18 GB unified memory). MLX uses unified memory — no `.to(device)`, weights and activations share the chip. Peak specs are configurable in `config.py` for other M-series chips.
