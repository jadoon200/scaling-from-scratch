"""Fit the Chinchilla scaling law from a sweep of training runs.

    L(N, D) = E + A / N^alpha + B / D^beta

where N = non-embedding params, D = training tokens, E = irreducible loss.
We track *non-embedding* params (the quantity the Kaplan/Hoffmann papers use):
embedding params don't contribute the same FLOPs and wash out as models grow.

Workflow:
    python scaling.py --sweep        # run the experiment grid, save runs/results.json
    python scaling.py --fit          # fit the law to runs/results.json, plot

The key result to reproduce: at a fixed compute budget C = 6*N*D, optimal
allocation scales as N* ~ C^0.5 and D* ~ C^0.5 — grow model and data together.

NOTE: a clean fit needs runs that are *converged* at each (N, D). Short runs on
a tiny corpus give a noisy fit — the machinery is what's being demonstrated; the
coefficients sharpen as you add longer runs on more tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


RESULTS_PATH = "runs/results.json"


# ---- experiment sweep -------------------------------------------------------

def run_sweep(presets: list[str], token_budgets: list[int], steps_cap: int,
              batch_size: int, data_dir: str):
    """Train each (preset, token-budget) point and record final val loss."""
    from config import SCALING_PRESETS, TrainConfig
    from train import train_model

    os.makedirs("runs", exist_ok=True)
    results = _load() if os.path.exists(RESULTS_PATH) else []

    for name in presets:
        cfg = SCALING_PRESETS[name]
        for D in token_budgets:
            tokens_per_step = batch_size * cfg.seq_len
            want_steps = max(1, D // tokens_per_step)
            steps = min(steps_cap, want_steps)
            tc = TrainConfig(batch_size=batch_size, max_steps=steps,
                             warmup_steps=max(1, steps // 20),
                             data_dir=data_dir, run_name=f"{name}_D{D}",
                             log_every=max(1, steps // 5))
            actual_D = steps * tokens_per_step
            print(f"\n=== {name} preset, target D={D/1e6:.0f}M "
                  f"-> {steps} steps, actual D={actual_D/1e6:.1f}M ===")
            if steps < want_steps:
                print(f"  !! WARNING: steps_cap={steps_cap} clamped this budget "
                      f"(wanted {want_steps} steps). actual_D is pinned to the cap, "
                      f"so different budgets may collapse to the SAME D and the "
                      f"Chinchilla fit will be degenerate. Raise --steps-cap.")
            m = train_model(cfg, tc, verbose=True, save=False)
            results.append({
                "preset": name,
                "N": m["n_params_non_embedding"],
                "N_total": m["n_params"],
                "D": actual_D,
                "val_loss": m["val_loss"],
                "avg_mfu": m["avg_mfu"],
            })
            _save(results)
    print(f"\nsaved {len(results)} points to {RESULTS_PATH}")


# ---- Chinchilla fit ---------------------------------------------------------

def chinchilla(Nth, E, A, B, alpha, beta):
    N, D = Nth
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def fit(plot: bool = True):
    from scipy.optimize import curve_fit

    results = _load()
    if len(results) < 5:
        print(f"warning: only {len(results)} points — Chinchilla has 5 free "
              f"params, so the fit is under-determined. Add more runs.")
    N = np.array([r["N"] for r in results], dtype=float)
    D = np.array([r["D"] for r in results], dtype=float)
    L = np.array([r["val_loss"] for r in results], dtype=float)

    # fit in a sensible range; bounds keep exponents physical (0,1)
    p0 = [min(L) * 0.9, 1.0, 1.0, 0.34, 0.28]
    bounds = ([0, 0, 0, 0.01, 0.01], [min(L), 1e6, 1e6, 1.0, 1.0])
    try:
        popt, _ = curve_fit(chinchilla, (N, D), L, p0=p0, bounds=bounds, maxfev=100000)
        E, A, B, alpha, beta = popt
        pred = chinchilla((N, D), *popt)
        rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
        print("\nFitted Chinchilla law  L(N,D) = E + A/N^alpha + B/D^beta")
        print(f"  E (irreducible) = {E:.3f}")
        print(f"  A = {A:.3g}   alpha = {alpha:.3f}")
        print(f"  B = {B:.3g}   beta  = {beta:.3f}")
        print(f"  fit RMSE = {rmse:.4f} nats")
        # compute-optimal exponent: a = beta/(alpha+beta) => N* ~ C^a
        a = beta / (alpha + beta)
        print(f"\n  compute-optimal: N* ~ C^{a:.2f}, D* ~ C^{1-a:.2f} "
              f"(Chinchilla found ~0.5 / 0.5)")
        if plot:
            _plot(results, popt)
    except Exception as e:
        print(f"fit failed: {e}\nLikely too few/too noisy points — add runs.")


def _plot(results, popt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([r["N"] for r in results], dtype=float)
    D = np.array([r["D"] for r in results], dtype=float)
    L = np.array([r["val_loss"] for r in results], dtype=float)
    C = 6 * N * D  # training FLOPs

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # (1) loss vs compute, the headline scaling plot
    order = np.argsort(C)
    ax[0].scatter(C, L, c=np.log10(N), cmap="viridis", s=60, zorder=3)
    ax[0].plot(C[order], chinchilla((N[order], D[order]), *popt), "k--",
               alpha=0.5, label="fit")
    ax[0].set_xscale("log"); ax[0].set_xlabel("compute C = 6ND (FLOPs)")
    ax[0].set_ylabel("val loss (nats)"); ax[0].set_title("Loss vs compute")
    ax[0].legend()

    # (2) iso-loss surface: loss vs N, colored by D
    ax[1].scatter(N, L, c=np.log10(D), cmap="plasma", s=60)
    ax[1].set_xscale("log"); ax[1].set_xlabel("non-embedding params N")
    ax[1].set_ylabel("val loss (nats)"); ax[1].set_title("Loss vs N (color = log D)")

    fig.tight_layout()
    out = "runs/scaling_fit.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved plot to {out}")


def _load() -> list[dict]:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _save(results: list[dict]):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="run the experiment grid")
    ap.add_argument("--fit", action="store_true", help="fit law to runs/results.json")
    ap.add_argument("--presets", nargs="+", default=["1M", "3M", "10M"])
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[1_000_000, 4_000_000, 16_000_000])
    ap.add_argument("--steps-cap", type=int, default=50000,
                    help="safety ceiling on steps; keep it above D/tokens_per_step "
                         "for every budget or D collapses and the fit degenerates")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--data-dir", type=str, default="data/fineweb")
    ap.add_argument("--no-report", action="store_true",
                    help="skip auto-generating report figures after a sweep")
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args.presets, args.budgets, args.steps_cap,
                  args.batch_size, args.data_dir)
        # graphs are part of the pipeline: a sweep always produces the figures
        # and the RESULTS.md writeup unless explicitly suppressed
        if not args.no_report:
            print("\n=== building report figures + RESULTS.md ===")
            import report
            report.main()
    if args.fit or not args.sweep:
        fit(plot=True)


if __name__ == "__main__":
    main()
