"""
plot_results.py
---------------
Reads multiseed_results.json and produces two publication-ready figures:

  Figure 1 — convergence_plot.png
      Per-round accuracy bands (mean ± std) for all 4 defenses across seeds.

  Figure 2 — barplot.png
      Steady-state accuracy bar chart with error bars and per-seed scatter.

Usage:
    python plot_results.py
    python plot_results.py --input multiseed_results.json
    python plot_results.py --dpi 150
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "none": "#d62728",    # red
    "trimmed_mean": "#1f77b4",  # blue
    "fltrust": "#ff7f0e",       # orange
    "trust_anchored": "#2ca02c", # green
}
LABELS = {
    "none": "No defense (FedAvg)",
    "trimmed_mean": "Trimmed mean",
    "fltrust": "FLTrust",
    "trust_anchored": "Trust-anchored (proposed)",
}
ORDER = ["none", "trimmed_mean", "fltrust", "trust_anchored"]


def load(path):
    with open(path) as f:
        return json.load(f)


def convergence_plot(data, out, dpi):
    rounds = list(range(1, data["config"]["rounds"] + 1))
    ss = data["aggregate"]
    cfg = data["config"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")

    for key in ORDER:
        mean_key = key.replace("trust_anchored", "ta") \
                      .replace("trimmed_mean", "tm") \
                      .replace("fltrust", "ft") \
                      .replace("none", "none")
        # Try both naming conventions in the JSON
        means = (data["per_round_mean"].get(key) or
                 data["per_round_mean"].get(mean_key))
        stds  = (data["per_round_std"].get(key) or
                 data["per_round_std"].get(mean_key))
        if means is None:
            continue

        means = np.array(means)
        stds  = np.array(stds)
        ss_mean = (ss.get(f"{key}_mean") or
                   ss.get(f"{mean_key}_mean", float("nan")))
        ss_std  = (ss.get(f"{key}_std") or
                   ss.get(f"{mean_key}_std", 0.0))

        label = f"{LABELS[key]}  ({ss_mean:.1f}% ±{ss_std:.1f})"
        ax.plot(rounds, means, lw=2, color=COLORS[key], label=label)
        ax.fill_between(rounds, means - stds, means + stds,
                        color=COLORS[key], alpha=0.13)

    attack = cfg.get("attack", "unknown").replace("_", "-")
    n_seeds = cfg.get("seeds", "?")
    alpha   = cfg.get("alpha", "?")

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Global FBS-Detection Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Accuracy per Round  —  attack: {attack}  |  "
        f"{n_seeds} seeds  |  α={alpha}  |  1 of 5 clients poisoned",
        fontsize=10, pad=8,
    )
    ax.set_ylim(0, 105)
    ax.set_xticks(rounds)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    ax.grid(alpha=0.28, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def bar_plot(data, out, dpi):
    ss  = data["aggregate"]
    cfg = data["config"]

    # Collect per-seed steady-state values for scatter
    per_seed = data.get("steady_state_per_seed", {})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(ORDER))
    bar_w = 0.52

    for i, key in enumerate(ORDER):
        mean_key = key.replace("trust_anchored", "ta") \
                      .replace("trimmed_mean", "tm") \
                      .replace("fltrust", "ft") \
                      .replace("none", "none")
        mean = (ss.get(f"{key}_mean") or ss.get(f"{mean_key}_mean", 0))
        std  = (ss.get(f"{key}_std")  or ss.get(f"{mean_key}_std",  0))

        ax.bar(x[i], mean, bar_w,
               color=COLORS[key], alpha=0.82,
               yerr=std, capsize=7,
               error_kw={"elinewidth": 1.8, "ecolor": "black"})
        ax.text(x[i], mean + std + 1.8, f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

        # per-seed scatter dots
        seeds_vals = (per_seed.get(key) or per_seed.get(mean_key))
        if seeds_vals:
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(seeds_vals))
            ax.scatter(x[i] + jitter, seeds_vals,
                       color=COLORS[key], alpha=0.45, s=18, zorder=3)

    attack  = cfg.get("attack", "unknown").replace("_", "-")
    n_seeds = cfg.get("seeds", "?")
    alpha   = cfg.get("alpha", "?")
    k       = ss.get("k", "?")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in ORDER], fontsize=9)
    ax.set_ylabel("Steady-state Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Steady-state Summary  —  attack: {attack}  |  "
        f"last {k} rounds  |  {n_seeds} seeds  |  α={alpha}",
        fontsize=10, pad=8,
    )
    ax.set_ylim(0, 112)
    ax.grid(alpha=0.28, axis="y", linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="multiseed_results.json")
    ap.add_argument("--convergence-out", default="convergence_plot.png")
    ap.add_argument("--bar-out",         default="barplot.png")
    ap.add_argument("--dpi", type=int,   default=300)
    args = ap.parse_args()

    try:
        data = load(args.input)
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.input} not found — run multiseed.py first")

    convergence_plot(data, args.convergence_out, args.dpi)
    bar_plot(data, args.bar_out, args.dpi)


if __name__ == "__main__":
    main()
