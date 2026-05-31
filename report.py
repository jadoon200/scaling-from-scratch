"""Generate presentation-ready figures and a markdown report.

Produces, under report/:
  figures/param_breakdown.png   embedding vs non-embedding params per preset
  figures/scaling_compute.png   val loss vs training compute C=6ND (+ Chinchilla fit)
  figures/scaling_NvsD.png      loss surface over (N, D)
  figures/mfu.png               achieved MFU per run
  figures/roofline.png          achieved FLOP/s vs arithmetic intensity (+ roofs)
  RESULTS.md                    the writeup, embedding the figures

Run after a sweep:
    python scaling.py --sweep ...        # writes runs/results.json
    python report.py                     # builds figures + RESULTS.md

Figures that need no training (param_breakdown, roofline) are generated even if
runs/results.json is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (SCALING_PRESETS, PEAK_FLOPS_FP16, PEAK_BW_BYTES, RIDGE_POINT)

FIG_DIR = "report/figures"
RESULTS_JSON = "runs/results.json"
ROOFLINE_JSON = "runs/roofline.json"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------- param figure
def fig_param_breakdown():
    names = list(SCALING_PRESETS)
    ne = np.array([SCALING_PRESETS[n].n_params_non_embedding for n in names]) / 1e6
    emb = np.array([SCALING_PRESETS[n].n_params_embedding for n in names]) / 1e6

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    ax.bar(x, emb, label="embedding", color="#c9b3e0")
    ax.bar(x, ne, bottom=emb, label="non-embedding (transformer)", color="#6a3d9a")
    for i, n in enumerate(names):
        frac = 100 * emb[i] / (emb[i] + ne[i])
        ax.text(i, emb[i] + ne[i], f"{frac:.0f}%\nembed", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("parameters (millions)")
    ax.set_title("Where the parameters live: embedding dominates at small scale")
    ax.legend()
    fig.tight_layout()
    _save(fig, "param_breakdown.png")


# ------------------------------------------------------------- roofline figure
def fig_roofline():
    if not os.path.exists(ROOFLINE_JSON):
        print(f"  (skip roofline figure — no {ROOFLINE_JSON})")
        return
    data = json.load(open(ROOFLINE_JSON))
    pts = data["points"]
    intens = np.array([p["intensity"] for p in pts])
    achieved = np.array([p["tflops"] for p in pts])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # roofline: y = min(peak, slope * x)
    xs = np.logspace(0, 4, 200)
    roof = np.minimum(PEAK_FLOPS_FP16 / 1e12, PEAK_BW_BYTES / 1e12 * xs)
    ax.plot(xs, roof, "k-", lw=1.5, label="roofline (M3 Pro)")
    ax.axvline(RIDGE_POINT, ls=":", color="gray", label=f"ridge {RIDGE_POINT:.0f} FLOP/byte")
    ax.scatter(intens, achieved, c="#e31a1c", s=60, zorder=3, label="measured")
    for p in pts:
        ax.annotate(p["label"], (p["intensity"], p["tflops"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"Roofline: peak {PEAK_FLOPS_FP16/1e12:.1f} TFLOP/s, "
                 f"{PEAK_BW_BYTES/1e9:.0f} GB/s")
    ax.legend()
    fig.tight_layout()
    _save(fig, "roofline.png")


# ---------------------------------------------------------- scaling-law figures
def _load_results():
    if not os.path.exists(RESULTS_JSON):
        return None
    return json.load(open(RESULTS_JSON))


def _fit(results):
    from scipy.optimize import curve_fit
    N = np.array([r["N"] for r in results], float)
    D = np.array([r["D"] for r in results], float)
    L = np.array([r["val_loss"] for r in results], float)

    def chin(ND, E, A, B, a, b):
        n, d = ND
        return E + A / np.power(n, a) + B / np.power(d, b)

    p0 = [min(L) * 0.9, 1.0, 1.0, 0.34, 0.28]
    bounds = ([0, 0, 0, 0.01, 0.01], [min(L), 1e9, 1e9, 1.0, 1.0])
    popt, _ = curve_fit(chin, (N, D), L, p0=p0, bounds=bounds, maxfev=200000)
    pred = chin((N, D), *popt)
    rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
    return popt, rmse, chin


def fig_scaling(results, fit_out):
    N = np.array([r["N"] for r in results], float)
    D = np.array([r["D"] for r in results], float)
    L = np.array([r["val_loss"] for r in results], float)
    C = 6 * N * D

    # (1) loss vs compute
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(C, L, c=np.log10(N), cmap="viridis", s=70, zorder=3)
    if fit_out is not None:
        popt, rmse, chin = fit_out
        order = np.argsort(C)
        ax.plot(C[order], chin((N[order], D[order]), *popt), "k--", alpha=0.6,
                label=f"Chinchilla fit (RMSE {rmse:.3f})")
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("training compute  C = 6ND  (FLOPs)")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("Scaling: loss decreases with compute")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("log10(non-embedding params)")
    fig.tight_layout()
    _save(fig, "scaling_compute.png")

    # (2) loss vs N and vs D
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    s0 = axes[0].scatter(N, L, c=np.log10(D), cmap="plasma", s=60)
    axes[0].set_xscale("log"); axes[0].set_xlabel("non-embedding params N")
    axes[0].set_ylabel("val loss"); axes[0].set_title("Loss vs N (color = log D)")
    fig.colorbar(s0, ax=axes[0]).set_label("log10(tokens D)")
    s1 = axes[1].scatter(D, L, c=np.log10(N), cmap="viridis", s=60)
    axes[1].set_xscale("log"); axes[1].set_xlabel("training tokens D")
    axes[1].set_ylabel("val loss"); axes[1].set_title("Loss vs D (color = log N)")
    fig.colorbar(s1, ax=axes[1]).set_label("log10(params N)")
    fig.tight_layout()
    _save(fig, "scaling_NvsD.png")


def fig_mfu(results):
    labels = [f"{r['preset']}\nD={r['D']/1e6:.0f}M" for r in results]
    mfu = [100 * r["avg_mfu"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(labels)), mfu, color="#1f78b4")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("avg MFU (%)")
    ax.set_title("Model FLOPs Utilization per run (M3 Pro)")
    ax.axhline(np.mean(mfu), ls="--", color="gray",
               label=f"mean {np.mean(mfu):.1f}%")
    ax.legend()
    fig.tight_layout()
    _save(fig, "mfu.png")


# ------------------------------------------------------------------- the report
def write_report(results, fit_out):
    os.makedirs("report", exist_ok=True)
    lines = []
    A = lines.append
    A("# MLX Tiny Transformer — Results\n")
    A("A decoder-only transformer built from scratch in Apple MLX, trained on "
      "FineWeb-Edu on an Apple M3 Pro (18 GB). This writeup focuses on the "
      "*systems* numbers — FLOPs, MFU, and empirical scaling laws — not just "
      "that the model runs.\n")

    A("## 1. Parameter accounting\n")
    A("At small scale the token-embedding table dominates the parameter budget, "
      "which is why such models are memory-bound. Scaling laws therefore track "
      "*non-embedding* parameters.\n")
    A("![param breakdown](figures/param_breakdown.png)\n")
    A("| preset | non-embedding | total | embedding % |")
    A("|---|---|---|---|")
    for name, cfg in SCALING_PRESETS.items():
        ne, tot = cfg.n_params_non_embedding, cfg.n_params
        A(f"| {name} | {ne/1e6:.2f}M | {tot/1e6:.2f}M | "
          f"{100*cfg.n_params_embedding/tot:.0f}% |")
    A("")

    A("## 2. Roofline\n")
    if os.path.exists(ROOFLINE_JSON):
        A("Achieved throughput against the M3 Pro roofline "
          f"(peak {PEAK_FLOPS_FP16/1e12:.1f} TFLOP/s, {PEAK_BW_BYTES/1e9:.0f} GB/s, "
          f"ridge {RIDGE_POINT:.0f} FLOP/byte). Points below the sloped roof are "
          "memory-bound; the model only approaches the compute roof at large "
          "batch and sequence length.\n")
        A("![roofline](figures/roofline.png)\n")
    else:
        A("_(roofline figure not generated — run the roofline capture)_\n")

    A("## 3. Scaling law\n")
    if results:
        A(f"Swept {len(results)} (N, D) points and fit the Chinchilla form\n")
        A("```\nL(N, D) = E + A / N^α + B / D^β\n```\n")
        if fit_out is not None:
            popt, rmse, _ = fit_out
            E, Acoef, Bcoef, alpha, beta = popt
            a = beta / (alpha + beta)
            A("| coefficient | value |")
            A("|---|---|")
            A(f"| E (irreducible loss) | {E:.3f} |")
            A(f"| A | {Acoef:.3g} |")
            A(f"| α | {alpha:.3f} |")
            A(f"| B | {Bcoef:.3g} |")
            A(f"| β | {beta:.3f} |")
            A(f"| fit RMSE | {rmse:.4f} nats |")
            A("")
            A(f"Compute-optimal allocation from this fit: **N\\* ∝ C^{a:.2f}**, "
              f"**D\\* ∝ C^{1-a:.2f}** (Hoffmann et al. found ≈ 0.5 / 0.5; small "
              "under-converged runs on a limited corpus shift the exponents).\n")
        A("![scaling vs compute](figures/scaling_compute.png)\n")
        A("![loss vs N and D](figures/scaling_NvsD.png)\n")
        A("## 4. Training efficiency (MFU)\n")
        A("![mfu](figures/mfu.png)\n")
        mean_mfu = 100 * np.mean([r["avg_mfu"] for r in results])
        A(f"Mean MFU across runs: **{mean_mfu:.1f}%**. For reference GPT-3 175B "
          "trained at ~46% on A100s; small models are memory-bound and sit "
          "lower, climbing with model width and batch size.\n")
    else:
        A("_(no runs/results.json yet — run `python scaling.py --sweep ...`)_\n")

    A("## Reproduce\n")
    A("```bash\nconda activate mlx-transformer\n"
      "python -m data.download --tokens 50_000_000\n"
      "python scaling.py --sweep --presets 1M 3M 10M "
      "--budgets 2_000_000 6_000_000 18_000_000\n"
      "python report.py\n```\n")

    open("report/RESULTS.md", "w").write("\n".join(lines))
    print("wrote report/RESULTS.md")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("generating figures...")
    fig_param_breakdown()
    fig_roofline()
    results = _load_results()
    fit_out = None
    if results:
        try:
            fit_out = _fit(results)
        except Exception as e:
            print(f"  (fit failed: {e})")
        fig_scaling(results, fit_out)
        fig_mfu(results)
    write_report(results, fit_out)


if __name__ == "__main__":
    main()
