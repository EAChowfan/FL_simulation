"""
sweep.py
--------
Step 1 of the experimental plan: find the attack scale where no-defense
clearly collapses below its clean ceiling. Run this first — the scale you
pick here becomes the fixed scale for the 20-seed comparison (Step 2) and
the stealth-attack comparison (Step 3).

What it runs:
  1. Clean baseline (--no-attack, defense=none) — establishes the ceiling
  2. For each scale in --scales: defense=none, sign_flip attack

Usage:
    python sweep.py                              # 5 seeds, scales 5-50
    python sweep.py --scales 10,20,30,50 --seeds 5 --alpha 0.5
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import time

import numpy as np

SERVER_ADDR = "127.0.0.1:8080"
N_CLIENTS   = 5
HONEST_DATA = [f"client{i}_data.csv" for i in range(1, 5)]
POISON_DATA = "poison_data.csv"
ACC_RE      = re.compile(r"\[Round (\d+)\].*VALIDATION accuracy =\s*([\d.]+)%")


def run_federation(scale, rounds, alpha, seed, no_attack=False,
                   local_epochs=5):
    subprocess.run(
        [sys.executable, "generate_data.py",
         "--alpha", str(alpha), "--seed", str(seed)],
        check=True, stdout=subprocess.DEVNULL,
    )

    server = subprocess.Popen(
        [sys.executable, "server.py",
         "--server_address", SERVER_ADDR, "--defense", "none",
         "--rounds", str(rounds), "--clients", str(N_CLIENTS)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    for _ in range(150):
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        server.kill(); server.communicate()
        raise RuntimeError("Server did not start within 15 s")

    procs = []
    for i, data in enumerate(HONEST_DATA, start=1):
        procs.append(subprocess.Popen(
            [sys.executable, "client_normal.py",
             "--server_address", SERVER_ADDR, "--data", data, "--cid", str(i),
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    if no_attack:
        procs.append(subprocess.Popen(
            [sys.executable, "client_normal.py",
             "--server_address", SERVER_ADDR,
             "--data", POISON_DATA, "--cid", "5",
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    else:
        procs.append(subprocess.Popen(
            [sys.executable, "client_poison.py",
             "--server_address", SERVER_ADDR, "--data", POISON_DATA,
             "--attack", "sign_flip", "--scale", str(scale),
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    try:
        out, _ = server.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        server.kill()
        out, _ = server.communicate()
        raise RuntimeError(f"Server timed out (scale={scale})")
    finally:
        for p in procs:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: p.kill()

    acc = {}
    for line in out.splitlines():
        m = ACC_RE.search(line)
        if m:
            acc[int(m.group(1))] = float(m.group(2))

    if not acc:
        return float("nan")

    k    = min(5, len(acc))
    tail = sorted(acc.keys())[-k:]
    return float(np.mean([acc[r] for r in tail]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=str, default="5,10,20,30,50",
                    help="comma-separated list of attack scales to sweep")
    ap.add_argument("--local-epochs", type=int, default=5,
                    help="local SGD epochs per FL round (5 recommended — "
                         "momentum needs multiple epochs per round to escape "
                         "the initial plateau)")
    ap.add_argument("--seeds",  type=int, default=5,
                    help="seeds per scale point (5 is enough for a quick sweep)")
    ap.add_argument("--alpha",  type=float, default=0.5)
    ap.add_argument("--rounds", type=int,   default=15)
    args = ap.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    seeds  = list(range(args.seeds))

    print("=" * 60)
    print("  Magnitude Sweep — Step 1")
    print("=" * 60)

    # --- clean ceiling ---
    print(f"\n[Baseline] no-attack ceiling  ({args.seeds} seeds, "
          f"local_epochs={args.local_epochs})...")
    ceiling_vals = []
    for seed in seeds:
        v = run_federation(0, args.rounds, args.alpha, seed, no_attack=True,
                           local_epochs=args.local_epochs)
        ceiling_vals.append(v)
        time.sleep(1)
    ceiling     = float(np.nanmean(ceiling_vals))
    ceiling_std = float(np.nanstd(ceiling_vals))
    print(f"  Clean ceiling: {ceiling:.1f}% ± {ceiling_std:.1f}")

    # --- scale sweep ---
    results = {}
    for scale in scales:
        print(f"\n[scale={scale:.0f}]  defense=none  attack=sign_flip ...")
        vals = []
        for seed in seeds:
            v = run_federation(scale, args.rounds, args.alpha, seed,
                               local_epochs=args.local_epochs)
            vals.append(v)
            time.sleep(1)
        mean = float(np.nanmean(vals))
        std  = float(np.nanstd(vals))
        results[scale] = (mean, std)
        print(f"  {mean:.1f}% ± {std:.1f}  (drop {ceiling - mean:+.1f} pp from ceiling)")

    # --- summary table ---
    print("\n" + "=" * 60)
    print(f"  SWEEP SUMMARY  (alpha={args.alpha}, {args.seeds} seeds, "
          f"{args.rounds} rounds)")
    print("=" * 60)
    print(f"  Clean ceiling  : {ceiling:.1f}% ± {ceiling_std:.1f}")
    print(f"\n  {'Scale':>7} | {'Accuracy':>13} | {'Drop':>8} | Verdict")
    print(f"  {'-'*7}-+-{'-'*13}-+-{'-'*8}-+-{'-'*20}")
    for scale, (mean, std) in results.items():
        drop    = ceiling - mean
        verdict = ("COLLAPSE" if drop > 20
                   else "partial" if drop > 8
                   else "minimal")
        print(f"  {scale:>7.0f} | {mean:5.1f} ± {std:<5.1f}% | "
              f"{drop:>+7.1f}% | {verdict}")

    print(f"\n  >> Recommended scale: first point where drop > 20 pp")

    with open("sweep_results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "ceiling": ceiling, "ceiling_std": ceiling_std,
            "scales": {str(int(s)): {"mean": m, "std": sd}
                       for s, (m, sd) in results.items()},
        }, f, indent=2)
    print("  saved -> sweep_results.json")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("white")
        sc    = list(results.keys())
        means = [results[s][0] for s in sc]
        stds  = [results[s][1] for s in sc]

        ax.axhline(ceiling, color="#2ca02c", lw=2, linestyle="--",
                   label=f"Clean ceiling  ({ceiling:.0f}%)")
        ax.errorbar(sc, means, yerr=stds, fmt="o-", color="#d62728",
                    lw=2, capsize=6, label="No defense  (sign-flip attack)")

        ax.set_xlabel("Attack scale  (sign-flip ×)", fontsize=12)
        ax.set_ylabel("Steady-state Accuracy (%)", fontsize=12)
        ax.set_title("Magnitude Sweep — No-Defense Collapse", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(sc)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("sweep_plot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  saved -> sweep_plot.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
