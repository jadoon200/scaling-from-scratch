"""Quality vs compression Pareto sweep.

Sweeps precision (INT8/INT4) x group size (32/64/128) on the trained model and
plots validation perplexity against model storage. This is the decision-relevant
view: for a given memory budget, which (bits, group) gives the best quality?

Needs trained weights (model.safetensors) + val data; train.py produces them.

    python pareto.py
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ModelConfig, QuantConfig
from model import Transformer, quantize_model
from bench import perplexity

CKPT = "model.safetensors"
DATA = "data/fineweb/val.bin"


def storage_mb(cfg: ModelConfig, bits, group_size) -> float:
    base = Transformer(cfg); mx.eval(base.parameters())
    n_total = sum(p.size for _, p in tree_flatten(base.parameters()))
    n_emb = cfg.vocab_size * cfg.d_model
    if bits is None:
        return n_total * 16 / 8 / 1e6  # fp16
    eff = QuantConfig(bits=bits, group_size=group_size).effective_bits()
    return (n_emb * 16 + (n_total - n_emb) * eff) / 8 / 1e6


def run():
    cfg = ModelConfig(d_model=512, n_layers=8, seq_len=256)
    if not (os.path.exists(CKPT) and os.path.exists(DATA)):
        print(f"need {CKPT} + {DATA} (run train.py first)")
        return

    configs = [(None, None)] + [(b, g) for b in (8, 4) for g in (32, 64, 128)]
    rows = []
    print(f"{'config':>12} {'MB':>7} {'ppl':>8}")
    for bits, g in configs:
        m = Transformer(cfg); m.load_weights(CKPT); mx.eval(m.parameters())
        if bits is not None:
            quantize_model(m, bits=bits, group_size=g); mx.eval(m.parameters())
        ppl = perplexity(m, DATA, cfg)
        mb = storage_mb(cfg, bits, g)
        tag = "fp16" if bits is None else f"INT{bits} g{g}"
        rows.append({"tag": tag, "bits": bits, "group": g, "mb": mb, "ppl": ppl})
        print(f"{tag:>12} {mb:>7.1f} {ppl:>8.2f}")

    _plot(rows)
    return rows


def _plot(rows):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        if r["bits"] is None:
            color, marker = "#888888", "*"
        else:
            color = "#4f81bd" if r["bits"] == 8 else "#c0504d"
            marker = "o"
        ax.scatter(r["mb"], r["ppl"], c=color, marker=marker, s=110, zorder=3)
        ax.annotate(r["tag"], (r["mb"], r["ppl"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("model storage (MB)  — smaller is better →" )
    ax.invert_xaxis()
    ax.set_ylabel("validation perplexity  — lower is better")
    ax.set_title("Quality vs compression Pareto (INT8 blue, INT4 red, fp16 ★)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    os.makedirs("report/figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig("report/figures/pareto.png", bbox_inches="tight")
    plt.close(fig)
    print("saved report/figures/pareto.png")


if __name__ == "__main__":
    run()
