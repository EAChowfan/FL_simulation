
"""
multi_seed.py
-------------
Multi-seed harness for the FBS-detection FL poisoning PoC.
 
Because a single SGD trajectory on non-IID data is noisy and seed-sensitive
(one run can land anywhere from weak to strong recovery), no single run gives a
trustworthy number. This harness runs the whole experiment over many seeds and
reports the aggregate mean +/- std -- the figure you can actually defend -- plus
a poster-ready plot.
 
For each seed it:
  1. regenerates the dataset with that seed (genuinely different data)
  2. runs the federation with defense OFF and ON
  3. records per-round validation accuracy and the steady-state tail mean
 
It then aggregates across seeds:
  - per-round mean +/- std (for the convergence band plot)
  - per-seed steady-state means -> overall mean +/- std (the headline number)
 
Usage:
    python multi_seed.py                         # 10 seeds, alpha 0.5
    python multi_seed.py --seeds 20 --alpha 0.3
    python multi_seed.py --seeds 10 --rounds 20 --scale 10
 
Outputs:
    multiseed_results.json   all per-seed + aggregate numbers
    multiseed_plot.png       convergence band + steady-state bar chart
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
N_CLIENTS = 5
HONEST_DATA = [f"client{i}_data.csv" for i in range(1, 5)]
POISON_DATA = "poison_data.csv"
ACC_RE = re.compile(r"\[Round (\d+)\].*VALIDATION accuracy =\s*([\d.]+)%")
 
 
def run_federation(defense, rounds, scale, trim, attack, checkpoint=None,
                   mag_bound=5.0, no_attack=False, local_epochs=5):
    """Launch one federation; return {round: accuracy} parsed from the server."""
    cmd = [sys.executable, "server.py",
           "--server_address", SERVER_ADDR, "--defense", defense,
           "--rounds", str(rounds), "--clients", str(N_CLIENTS),
           "--trim", str(trim), "--mag-bound", str(mag_bound)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    server = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Wait until the server socket is accepting connections (up to 15 s).
    for _ in range(150):
        try:
            with socket.create_connection(("127.0.0.1", 8080), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        server.kill()
        raise RuntimeError("Server did not become ready within 15 s")

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
             "--server_address", SERVER_ADDR, "--data", POISON_DATA, "--cid", "5",
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    else:
        procs.append(subprocess.Popen(
            [sys.executable, "client_poison.py",
             "--server_address", SERVER_ADDR, "--data", POISON_DATA,
             "--attack", attack, "--scale", str(scale),
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
 
    try:
        out, _ = server.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        server.kill()
        out, _ = server.communicate()
        raise RuntimeError(
            f"Server timed out after 600 s for defense={defense!r}. "
            f"Last output:\n" + "\n".join(out.splitlines()[-20:])
        )
    finally:
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
 
    acc = {}
    for line in out.splitlines():
        m = ACC_RE.search(line)
        if m:
            acc[int(m.group(1))] = float(m.group(2))

    if not acc:
        print("\n[ERROR] No accuracy lines matched in server output.")
        print(f"        Expected pattern: {ACC_RE.pattern}")
        print("        Last 20 lines of server output:")
        for line in out.splitlines()[-20:]:
            print(f"          {line}")
        raise RuntimeError(
            f"run_federation(defense={defense!r}) parsed zero accuracy entries — "
            "check that server.py log format matches ACC_RE above"
        )

    return acc
 
 
def tail_mean(acc, rounds, k):
    tail = [r for r in sorted(acc) if r != 0][-k:]
    return float(np.mean([acc[r] for r in tail])) if tail else float("nan")
 
 
def make_plots(rounds_axis, none_mat, tm_mat, ta_mat, ft_mat, ss,
               out_convergence, out_bar):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -> skipping plots; "
              "pip install matplotlib)")
        return

    none_mean = np.nanmean(none_mat, 0); none_std = np.nanstd(none_mat, 0)
    tm_mean   = np.nanmean(tm_mat,   0); tm_std   = np.nanstd(tm_mat,   0)
    ta_mean   = np.nanmean(ta_mat,   0); ta_std   = np.nanstd(ta_mat,   0)
    ft_mean   = np.nanmean(ft_mat,   0); ft_std   = np.nanstd(ft_mat,   0)

    COLORS = {"none": "#d62728", "tm": "#1f77b4",
              "ta": "#2ca02c",   "ft": "#ff7f0e"}

    # --- figure 1: convergence band ---
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    fig1.patch.set_facecolor("white")
    for mean, std, color, label in [
        (none_mean, none_std, COLORS["none"], "No defense (FedAvg)"),
        (tm_mean,   tm_std,   COLORS["tm"],   "Trimmed mean"),
        (ft_mean,   ft_std,   COLORS["ft"],   "FLTrust"),
        (ta_mean,   ta_std,   COLORS["ta"],   "Trust-anchored (proposed)"),
    ]:
        ax1.plot(rounds_axis, mean, lw=2, color=color, label=label)
        ax1.fill_between(rounds_axis, mean - std, mean + std,
                         color=color, alpha=0.15)
    ax1.set_xlabel("FL Round", fontsize=12)
    ax1.set_ylabel("Global FBS-detection Accuracy (%)", fontsize=12)
    ax1.set_title(
        f"Accuracy per Round  (mean \u00b1 std, {ss['n_seeds']} seeds)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(rounds_axis)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle=":")
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(out_convergence, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  convergence plot saved -> {out_convergence}")

    # --- figure 2: steady-state bar chart ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    fig2.patch.set_facecolor("white")
    labels = ["No defense", "Trimmed mean", "FLTrust", "Trust-anchored"]
    means  = [ss["none_mean"], ss["tm_mean"], ss["ft_mean"], ss["ta_mean"]]
    stds   = [ss["none_std"],  ss["tm_std"],  ss["ft_std"],  ss["ta_std"]]
    colors = [COLORS["none"], COLORS["tm"], COLORS["ft"], COLORS["ta"]]
    bars = ax2.bar(labels, means, yerr=stds, capsize=8,
                   color=colors, alpha=0.85, width=0.55,
                   error_kw={"elinewidth": 1.8})
    ax2.set_ylabel("Steady-state Accuracy (%)", fontsize=12)
    ax2.set_title(
        f"Steady-state Summary\n(last {ss['k']} rounds, {ss['n_seeds']} seeds)",
        fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3, axis="y", linestyle=":")
    ax2.spines[["top", "right"]].set_visible(False)
    for b, m, s in zip(bars, means, stds):
        ax2.text(b.get_x() + b.get_width() / 2, m + s + 2.5,
                 f"{m:.0f}\u00b1{s:.0f}%", ha="center", fontsize=9,
                 fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  bar chart saved       -> {out_bar}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--trim", type=int, default=1)
    ap.add_argument("--attack", type=str, default="sign_flip",
                    choices=["sign_flip", "noise", "label_flip"])
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--mag-bound", type=float, default=5.0,
                    help="magnitude bound for trust-anchored aggregation")
    ap.add_argument("--local-epochs", type=int, default=5,
                    help="local SGD epochs per FL round")
    ap.add_argument("--no-attack", action="store_true",
                    help="replace poison client with an honest client — "
                         "shows baseline accuracy without any attack")
    args = ap.parse_args()

    k = min(10, args.rounds - 1)  # tail length for steady-state
    rounds_axis = list(range(1, args.rounds + 1))
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    none_rows, tm_rows, ta_rows, ft_rows = [], [], [], []
    none_ss,  tm_ss,  ta_ss,  ft_ss  = [], [], [], []

    for si, seed in enumerate(seeds, start=1):
        print(f"\n[{si}/{len(seeds)}] seed={seed} "
              f"(alpha={args.alpha}, rounds={args.rounds}) ...")
        subprocess.run([sys.executable, "generate_data.py",
                        "--alpha", str(args.alpha), "--seed", str(seed)],
                       check=True, stdout=subprocess.DEVNULL)

        # Each seed is an independent trial — fresh LSTM (seed=42 init) for all
        # defenses. Isolates the delta to the defense mechanism.
        none_acc = run_federation("none", args.rounds, args.scale,
                                  args.trim, args.attack,
                                  mag_bound=args.mag_bound,
                                  no_attack=args.no_attack,
                                  local_epochs=args.local_epochs)
        time.sleep(1)
        tm_acc = run_federation("trimmed_mean", args.rounds, args.scale,
                                args.trim, args.attack,
                                mag_bound=args.mag_bound,
                                no_attack=args.no_attack,
                                local_epochs=args.local_epochs)
        time.sleep(1)
        ta_acc = run_federation("trust_anchored", args.rounds, args.scale,
                                args.trim, args.attack,
                                mag_bound=args.mag_bound,
                                no_attack=args.no_attack,
                                local_epochs=args.local_epochs)
        time.sleep(1)
        ft_acc = run_federation("fltrust", args.rounds, args.scale,
                                args.trim, args.attack,
                                mag_bound=args.mag_bound,
                                no_attack=args.no_attack,
                                local_epochs=args.local_epochs)

        none_rows.append([none_acc.get(r, np.nan) for r in rounds_axis])
        tm_rows.append([tm_acc.get(r, np.nan) for r in rounds_axis])
        ta_rows.append([ta_acc.get(r, np.nan) for r in rounds_axis])
        ft_rows.append([ft_acc.get(r, np.nan) for r in rounds_axis])
        n_ss = tail_mean(none_acc, args.rounds, k)
        t_ss = tail_mean(tm_acc,  args.rounds, k)
        a_ss = tail_mean(ta_acc,  args.rounds, k)
        f_ss = tail_mean(ft_acc,  args.rounds, k)
        none_ss.append(n_ss); tm_ss.append(t_ss)
        ta_ss.append(a_ss);   ft_ss.append(f_ss)
        print(f"      no-defense {n_ss:5.1f}%  trimmed {t_ss:5.1f}%  "
              f"trust-anchored {a_ss:5.1f}%  fltrust {f_ss:5.1f}%")
 
    none_mat = np.array(none_rows, dtype=float)
    tm_mat   = np.array(tm_rows,   dtype=float)
    ta_mat   = np.array(ta_rows,   dtype=float)
    ft_mat   = np.array(ft_rows,   dtype=float)
    none_ss  = np.array(none_ss, dtype=float)
    tm_ss    = np.array(tm_ss,   dtype=float)
    ta_ss    = np.array(ta_ss,   dtype=float)
    ft_ss    = np.array(ft_ss,   dtype=float)

    ss = {
        "k": k, "n_seeds": len(seeds),
        "none_mean": float(np.nanmean(none_ss)), "none_std": float(np.nanstd(none_ss)),
        "tm_mean":   float(np.nanmean(tm_ss)),   "tm_std":   float(np.nanstd(tm_ss)),
        "ta_mean":   float(np.nanmean(ta_ss)),   "ta_std":   float(np.nanstd(ta_ss)),
        "ft_mean":   float(np.nanmean(ft_ss)),   "ft_std":   float(np.nanstd(ft_ss)),
    }
    recovery_ta = ss["ta_mean"] - ss["none_mean"]
    recovery_ft = ss["ft_mean"] - ss["none_mean"]

    print("\n" + "=" * 60)
    print(f" AGGREGATE over {len(seeds)} seeds  "
          f"(attack={args.attack}, scale={args.scale}, alpha={args.alpha})")
    print("=" * 60)
    print(f"  Steady-state accuracy:")
    print(f"    No defense      : {ss['none_mean']:5.1f}% \u00b1 {ss['none_std']:.1f}")
    print(f"    Trimmed mean    : {ss['tm_mean']:5.1f}% \u00b1 {ss['tm_std']:.1f}  "
          f"(+{ss['tm_mean']-ss['none_mean']:.1f} pp)")
    print(f"    FLTrust         : {ss['ft_mean']:5.1f}% \u00b1 {ss['ft_std']:.1f}  "
          f"(+{recovery_ft:.1f} pp)")
    print(f"    Trust-anchored  : {ss['ta_mean']:5.1f}% \u00b1 {ss['ta_std']:.1f}  "
          f"(+{recovery_ta:.1f} pp)")

    with open("multiseed_results.json", "w") as f:
        json.dump({
            "config": vars(args), "seeds": seeds,
            "steady_state_per_seed": {
                "none": none_ss.tolist(), "trimmed_mean": tm_ss.tolist(),
                "trust_anchored": ta_ss.tolist(), "fltrust": ft_ss.tolist(),
            },
            "per_round_mean": {
                "none": np.nanmean(none_mat, 0).tolist(),
                "trimmed_mean": np.nanmean(tm_mat, 0).tolist(),
                "trust_anchored": np.nanmean(ta_mat, 0).tolist(),
                "fltrust": np.nanmean(ft_mat, 0).tolist(),
            },
            "per_round_std": {
                "none": np.nanstd(none_mat, 0).tolist(),
                "trimmed_mean": np.nanstd(tm_mat, 0).tolist(),
                "trust_anchored": np.nanstd(ta_mat, 0).tolist(),
                "fltrust": np.nanstd(ft_mat, 0).tolist(),
            },
            "aggregate": ss,
            "recovery_ta_pp": recovery_ta,
            "recovery_ft_pp": recovery_ft,
        }, f, indent=2)
    print(f"\n  numbers saved -> multiseed_results.json")

    make_plots(rounds_axis, none_mat, tm_mat, ta_mat, ft_mat, ss,
               "multiseed_convergence.png", "multiseed_barplot.png")
 
 
if __name__ == "__main__":
    main()
 




