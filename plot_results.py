"""
plot_results.py
---------------
Reads results.json produced by launcher.py and saves a poster-ready figure.

Usage:
    venv/bin/python plot_results.py                    # reads results.json
    venv/bin/python plot_results.py --input my.json    # custom input file
    venv/bin/python plot_results.py --out fig.pdf      # PDF instead of PNG
    venv/bin/python plot_results.py --dpi 150          # lower DPI for draft
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── colour palette (WCAG-AA contrast, printer-safe) ──────────────────────────
C_NONE = "#d62728"       # red   — no defense
C_TM   = "#1f77b4"       # blue  — trimmed mean
C_BAND = "#aec7e8"       # light blue — steady-state band


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_acc(d: dict) -> tuple[list[int], list[float]]:
    pairs = sorted((int(k), v) for k, v in d.items())
    rounds = [r for r, _ in pairs]
    accs   = [a for _, a in pairs]
    return rounds, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results.json")
    ap.add_argument("--out",   default="fl_results.png",
                    help="output filename (.png or .pdf)")
    ap.add_argument("--dpi",   type=int, default=300)
    args = ap.parse_args()

    try:
        data = load(args.input)
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.input} not found — run launcher.py first")

    rounds_n, acc_n = parse_acc(data["none"])
    rounds_t, acc_t = parse_acc(data["trimmed_mean"])
    ss = data["steady_state"]

    tail_start = ss["rounds_averaged"][0]
    tail_end   = ss["rounds_averaged"][-1]

    n_mean, n_std   = ss["none_mean"],         ss["none_std"]
    t_mean, t_std   = ss["trimmed_mean_mean"],  ss["trimmed_mean_std"]
    recovery        = ss["recovery_pp"]
    attack          = data.get("attack", "unknown")
    scale           = data.get("scale", "?")
    n_averaged      = len(ss["rounds_averaged"])

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    # steady-state shaded band for trimmed mean
    ax.axvspan(tail_start - 0.5, tail_end + 0.5,
               color=C_BAND, alpha=0.25, zorder=0,
               label=f"Steady-state window (last {n_averaged} rounds)")

    # per-round accuracy lines
    ax.plot(rounds_n, acc_n, color=C_NONE, linewidth=1.8,
            marker="o", markersize=4, zorder=3,
            label=f"No defense  (mean {n_mean:.1f}% ±{n_std:.1f})")
    ax.plot(rounds_t, acc_t, color=C_TM, linewidth=2.2,
            marker="s", markersize=4, zorder=3,
            label=f"Trimmed mean  (mean {t_mean:.1f}% ±{t_std:.1f})")

    # steady-state mean dashed lines
    ax.hlines(n_mean, tail_start - 0.5, tail_end + 0.5,
              colors=C_NONE, linestyles="--", linewidth=1.4, zorder=4)
    ax.hlines(t_mean, tail_start - 0.5, tail_end + 0.5,
              colors=C_TM,  linestyles="--", linewidth=1.4, zorder=4)

    # recovery annotation arrow
    mid_x = (tail_start + tail_end) / 2 + 0.3
    ax.annotate(
        "",
        xy=(mid_x, t_mean), xytext=(mid_x, n_mean),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
        zorder=5,
    )
    ax.text(mid_x + 0.3, (n_mean + t_mean) / 2,
            f"+{recovery:.1f} pp",
            va="center", ha="left", fontsize=9,
            color="black", fontweight="bold")

    # ── axes styling ──────────────────────────────────────────────────────────
    all_rounds = sorted(set(rounds_n) | set(rounds_t))
    ax.set_xlim(all_rounds[0] - 0.5, all_rounds[-1] + 0.8)
    ax.set_ylim(30, 102)
    ax.set_xticks(all_rounds)
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title(
        f"FL Poisoning Attack vs. Byzantine-Robust Aggregation\n"
        f"Attack: {attack.replace('_', ' ')}  |  Scale: {scale}  |  "
        f"1 of 5 clients poisoned  (non-IID data)",
        fontsize=11, pad=10,
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # legend
    extra = mpatches.Patch(color=C_BAND, alpha=0.4,
                           label=f"Steady-state window (last {n_averaged} rounds)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,
              loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}  ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
