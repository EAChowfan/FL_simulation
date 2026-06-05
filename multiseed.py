
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

LOG_FILE = "log.txt"


class _Tee:
    """Writes every print() call to both stdout and log.txt."""
    def __init__(self, path):
        self._file = open(path, "w", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


SERVER_ADDR = "127.0.0.1:8080"
N_CLIENTS = 5
HONEST_DATA = [f"client{i}_data.csv" for i in range(1, 5)]
POISON_DATA = "poison_data.csv"
ACC_RE    = re.compile(r"\[Round (\d+)\].*VALIDATION accuracy =\s*([\d.]+)%")
RECALL_RE = re.compile(r"recall=\s*([\d.]+)%")
F1_RE     = re.compile(r"\bF1=\s*([\d.]+)%")
PREC_RE   = re.compile(r"precision=\s*([\d.]+)%")
 
 
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
 
    metrics = {}
    current_round = None
    for line in out.splitlines():
        m = ACC_RE.search(line)
        if m:
            current_round = int(m.group(1))
            metrics[current_round] = {
                "acc": float(m.group(2)),
                "recall": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
            }
        elif current_round is not None:
            mr = RECALL_RE.search(line)
            mf = F1_RE.search(line)
            mp = PREC_RE.search(line)
            if mr:
                metrics[current_round]["recall"] = float(mr.group(1))
            if mf:
                metrics[current_round]["f1"] = float(mf.group(1))
            if mp:
                metrics[current_round]["precision"] = float(mp.group(1))

    if not metrics:
        print("\n[ERROR] No accuracy lines matched in server output.")
        print(f"        Expected pattern: {ACC_RE.pattern}")
        print("        Last 20 lines of server output:")
        for line in out.splitlines()[-20:]:
            print(f"          {line}")
        raise RuntimeError(
            f"run_federation(defense={defense!r}) parsed zero accuracy entries — "
            "check that server.py log format matches ACC_RE above"
        )

    return metrics
 
 
def tail_mean(metrics, rounds, k, key="acc"):
    tail = [r for r in sorted(metrics) if r != 0][-k:]
    return float(np.mean([metrics[r][key] for r in tail])) if tail else float("nan")
 
 
def make_plots(rounds_axis, acc_mats, rec_mats, ss, out_convergence, out_recall, out_bar):
    """
    acc_mats / rec_mats: dict keyed by short defense name ->  numpy (n_seeds \u00d7 n_rounds) matrix
    Keys expected: "none", "tm", "ft", "a1", "ta"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -> skipping plots; "
              "pip install matplotlib)")
        return

    COLORS = {
        "none": "#d62728",   # red
        "tm":   "#1f77b4",   # blue
        "ft":   "#ff7f0e",   # orange
        "a1":   "#9467bd",   # purple  \u2014 ablation: Anchor-1 only
        "ta":   "#2ca02c",   # green
    }
    SERIES = [
        ("none", "No defense (FedAvg)"),
        ("tm",   "Trimmed mean"),
        ("ft",   "FLTrust"),
        ("a1",   "Anchor-1 only (magnitude bound)"),
        ("ta",   "Trust-anchored  A1+A2 (proposed)"),
    ]

    def _band_plot(ax, mats_dict, ylabel, title):
        for key, label in SERIES:
            mat = mats_dict.get(key)
            if mat is None:
                continue
            mean = np.nanmean(mat, 0)
            std  = np.nanstd(mat,  0)
            ax.plot(rounds_axis, mean, lw=2, color=COLORS[key], label=label)
            ax.fill_between(rounds_axis, mean - std, mean + std,
                            color=COLORS[key], alpha=0.15)
        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xticks(rounds_axis)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    # --- figure 1: accuracy convergence ---
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    fig1.patch.set_facecolor("white")
    _band_plot(ax1, acc_mats, "FBS-Detection Accuracy (%)",
               f"Accuracy per Round  (mean \u00b1 std, {ss['n_seeds']} seeds)")
    fig1.tight_layout()
    fig1.savefig(out_convergence, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  accuracy convergence  -> {out_convergence}")

    # --- figure 2: recall convergence (primary metric for label_flip) ---
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    fig2.patch.set_facecolor("white")
    _band_plot(ax2, rec_mats, "FBS Detection Recall (%)",
               f"Recall per Round  (mean \u00b1 std, {ss['n_seeds']} seeds)\n"
               f"Recall = TP / (TP + FN)  \u2014  missed attacks drive this down")
    fig2.tight_layout()
    fig2.savefig(out_recall, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  recall convergence    -> {out_recall}")

    # --- figure 3: steady-state F1 bar chart (ablation view) ---
    bar_entries = [
        ("No defense",            "none_f1_mean", "none_f1_std", "none"),
        ("Trimmed mean",          "tm_f1_mean",   "tm_f1_std",   "tm"),
        ("FLTrust",               "ft_f1_mean",   "ft_f1_std",   "ft"),
        ("Anchor-1\n(mag bound)", "a1_f1_mean",   "a1_f1_std",   "a1"),
        ("Trust-anchored\n(A1+A2)", "ta_f1_mean", "ta_f1_std",   "ta"),
    ]
    # Only include entries that exist in ss (guards against old JSON)
    bar_entries = [(lbl, mk, sk, ck) for lbl, mk, sk, ck in bar_entries
                   if mk in ss]

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    fig3.patch.set_facecolor("white")
    bar_labels = [e[0] for e in bar_entries]
    f1_means   = [ss[e[1]] for e in bar_entries]
    f1_stds    = [ss[e[2]] for e in bar_entries]
    colors     = [COLORS[e[3]] for e in bar_entries]
    bars = ax3.bar(bar_labels, f1_means, yerr=f1_stds, capsize=8,
                   color=colors, alpha=0.85, width=0.55,
                   error_kw={"elinewidth": 1.8})
    ax3.set_ylabel("Steady-state F1 Score (%)", fontsize=12)
    ax3.set_title(
        f"F1 Score Ablation  (last {ss['k']} rounds, {ss['n_seeds']} seeds)\n"
        f"F1 balances precision and recall \u2014 accuracy can hide recall collapse",
        fontsize=10)
    ax3.set_ylim(0, 118)
    ax3.grid(alpha=0.3, axis="y", linestyle=":")
    ax3.spines[["top", "right"]].set_visible(False)
    for b, m, s in zip(bars, f1_means, f1_stds):
        ax3.text(b.get_x() + b.get_width() / 2, m + s + 2.5,
                 f"{m:.0f}\u00b1{s:.0f}%", ha="center", fontsize=9,
                 fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  F1 ablation bar chart -> {out_bar}")
 
 
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
    ap.add_argument("--mag-bound", type=float, default=25.0,
                    help="magnitude bound for trust-anchored aggregation")
    ap.add_argument("--local-epochs", type=int, default=5,
                    help="local SGD epochs per FL round")
    ap.add_argument("--no-attack", action="store_true",
                    help="replace poison client with an honest client — "
                         "shows baseline accuracy without any attack")
    args = ap.parse_args()

    tee = _Tee(LOG_FILE)
    sys.stdout = tee

    k = min(10, args.rounds - 1)  # tail length for steady-state
    rounds_axis = list(range(1, args.rounds + 1))
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    # accuracy matrices
    none_rows, tm_rows, ta_rows, ft_rows, a1_rows = [], [], [], [], []
    none_ss,   tm_ss,   ta_ss,   ft_ss,   a1_ss   = [], [], [], [], []
    # recall matrices
    none_rec_rows, tm_rec_rows, ta_rec_rows, ft_rec_rows, a1_rec_rows = [], [], [], [], []
    none_rec_ss,   tm_rec_ss,   ta_rec_ss,   ft_rec_ss,   a1_rec_ss   = [], [], [], [], []
    # F1 matrices
    none_f1_rows, tm_f1_rows, ta_f1_rows, ft_f1_rows, a1_f1_rows = [], [], [], [], []
    none_f1_ss,   tm_f1_ss,   ta_f1_ss,   ft_f1_ss,   a1_f1_ss   = [], [], [], [], []

    for si, seed in enumerate(seeds, start=1):
        print(f"\n[{si}/{len(seeds)}] seed={seed} "
              f"(alpha={args.alpha}, rounds={args.rounds}) ...")
        subprocess.run([sys.executable, "generate_data.py",
                        "--alpha", str(args.alpha), "--seed", str(seed)],
                       check=True, stdout=subprocess.DEVNULL)

        none_m = run_federation("none", args.rounds, args.scale,
                                args.trim, args.attack,
                                mag_bound=args.mag_bound,
                                no_attack=args.no_attack,
                                local_epochs=args.local_epochs)
        time.sleep(1)
        tm_m = run_federation("trimmed_mean", args.rounds, args.scale,
                              args.trim, args.attack,
                              mag_bound=args.mag_bound,
                              no_attack=args.no_attack,
                              local_epochs=args.local_epochs)
        time.sleep(1)
        ta_m = run_federation("trust_anchored", args.rounds, args.scale,
                              args.trim, args.attack,
                              mag_bound=args.mag_bound,
                              no_attack=args.no_attack,
                              local_epochs=args.local_epochs)
        time.sleep(1)
        ft_m = run_federation("fltrust", args.rounds, args.scale,
                              args.trim, args.attack,
                              mag_bound=args.mag_bound,
                              no_attack=args.no_attack,
                              local_epochs=args.local_epochs)
        time.sleep(1)
        a1_m = run_federation("anchor_1_only", args.rounds, args.scale,
                              args.trim, args.attack,
                              mag_bound=args.mag_bound,
                              no_attack=args.no_attack,
                              local_epochs=args.local_epochs)

        def _row(m, key): return [m.get(r, {}).get(key, np.nan) for r in rounds_axis]

        none_rows.append(_row(none_m, "acc")); tm_rows.append(_row(tm_m, "acc"))
        ta_rows.append(_row(ta_m,  "acc"));    ft_rows.append(_row(ft_m, "acc"))
        a1_rows.append(_row(a1_m,  "acc"))
        none_rec_rows.append(_row(none_m, "recall")); tm_rec_rows.append(_row(tm_m, "recall"))
        ta_rec_rows.append(_row(ta_m,  "recall"));    ft_rec_rows.append(_row(ft_m, "recall"))
        a1_rec_rows.append(_row(a1_m,  "recall"))
        none_f1_rows.append(_row(none_m, "f1")); tm_f1_rows.append(_row(tm_m, "f1"))
        ta_f1_rows.append(_row(ta_m,  "f1"));    ft_f1_rows.append(_row(ft_m, "f1"))
        a1_f1_rows.append(_row(a1_m,  "f1"))

        n_ss  = tail_mean(none_m, args.rounds, k, "acc")
        t_ss  = tail_mean(tm_m,   args.rounds, k, "acc")
        a_ss  = tail_mean(ta_m,   args.rounds, k, "acc")
        f_ss  = tail_mean(ft_m,   args.rounds, k, "acc")
        o_ss  = tail_mean(a1_m,   args.rounds, k, "acc")
        n_rec = tail_mean(none_m, args.rounds, k, "recall")
        t_rec = tail_mean(tm_m,   args.rounds, k, "recall")
        a_rec = tail_mean(ta_m,   args.rounds, k, "recall")
        f_rec = tail_mean(ft_m,   args.rounds, k, "recall")
        o_rec = tail_mean(a1_m,   args.rounds, k, "recall")
        n_f1  = tail_mean(none_m, args.rounds, k, "f1")
        t_f1  = tail_mean(tm_m,   args.rounds, k, "f1")
        a_f1  = tail_mean(ta_m,   args.rounds, k, "f1")
        f_f1  = tail_mean(ft_m,   args.rounds, k, "f1")
        o_f1  = tail_mean(a1_m,   args.rounds, k, "f1")

        none_ss.append(n_ss); tm_ss.append(t_ss)
        ta_ss.append(a_ss);   ft_ss.append(f_ss);   a1_ss.append(o_ss)
        none_rec_ss.append(n_rec); tm_rec_ss.append(t_rec)
        ta_rec_ss.append(a_rec);   ft_rec_ss.append(f_rec); a1_rec_ss.append(o_rec)
        none_f1_ss.append(n_f1); tm_f1_ss.append(t_f1)
        ta_f1_ss.append(a_f1);   ft_f1_ss.append(f_f1);   a1_f1_ss.append(o_f1)

        print(f"      no-defense  acc={n_ss:.1f}% rec={n_rec:.1f}% f1={n_f1:.1f}%")
        print(f"      trimmed     acc={t_ss:.1f}% rec={t_rec:.1f}% f1={t_f1:.1f}%")
        print(f"      fltrust     acc={f_ss:.1f}% rec={f_rec:.1f}% f1={f_f1:.1f}%")
        print(f"      anchor-1    acc={o_ss:.1f}% rec={o_rec:.1f}% f1={o_f1:.1f}%")
        print(f"      trust-anch  acc={a_ss:.1f}% rec={a_rec:.1f}% f1={a_f1:.1f}%")
 
    # accuracy matrices
    none_mat     = np.array(none_rows,     dtype=float)
    tm_mat       = np.array(tm_rows,       dtype=float)
    ta_mat       = np.array(ta_rows,       dtype=float)
    ft_mat       = np.array(ft_rows,       dtype=float)
    a1_mat       = np.array(a1_rows,       dtype=float)
    none_ss      = np.array(none_ss,       dtype=float)
    tm_ss        = np.array(tm_ss,         dtype=float)
    ta_ss        = np.array(ta_ss,         dtype=float)
    ft_ss        = np.array(ft_ss,         dtype=float)
    a1_ss        = np.array(a1_ss,         dtype=float)
    # recall matrices
    none_rec_mat = np.array(none_rec_rows, dtype=float)
    tm_rec_mat   = np.array(tm_rec_rows,   dtype=float)
    ta_rec_mat   = np.array(ta_rec_rows,   dtype=float)
    ft_rec_mat   = np.array(ft_rec_rows,   dtype=float)
    a1_rec_mat   = np.array(a1_rec_rows,   dtype=float)
    none_rec_ss  = np.array(none_rec_ss,   dtype=float)
    tm_rec_ss    = np.array(tm_rec_ss,     dtype=float)
    ta_rec_ss    = np.array(ta_rec_ss,     dtype=float)
    ft_rec_ss    = np.array(ft_rec_ss,     dtype=float)
    a1_rec_ss    = np.array(a1_rec_ss,     dtype=float)
    # F1 matrices
    none_f1_mat  = np.array(none_f1_rows,  dtype=float)
    tm_f1_mat    = np.array(tm_f1_rows,    dtype=float)
    ta_f1_mat    = np.array(ta_f1_rows,    dtype=float)
    ft_f1_mat    = np.array(ft_f1_rows,    dtype=float)
    a1_f1_mat    = np.array(a1_f1_rows,    dtype=float)
    none_f1_ss   = np.array(none_f1_ss,    dtype=float)
    tm_f1_ss     = np.array(tm_f1_ss,      dtype=float)
    ta_f1_ss     = np.array(ta_f1_ss,      dtype=float)
    ft_f1_ss     = np.array(ft_f1_ss,      dtype=float)
    a1_f1_ss     = np.array(a1_f1_ss,      dtype=float)

    ss = {
        "k": k, "n_seeds": len(seeds),
        # accuracy
        "none_mean": float(np.nanmean(none_ss)), "none_std": float(np.nanstd(none_ss)),
        "tm_mean":   float(np.nanmean(tm_ss)),   "tm_std":   float(np.nanstd(tm_ss)),
        "ta_mean":   float(np.nanmean(ta_ss)),   "ta_std":   float(np.nanstd(ta_ss)),
        "ft_mean":   float(np.nanmean(ft_ss)),   "ft_std":   float(np.nanstd(ft_ss)),
        "a1_mean":   float(np.nanmean(a1_ss)),   "a1_std":   float(np.nanstd(a1_ss)),
        # recall
        "none_rec_mean": float(np.nanmean(none_rec_ss)), "none_rec_std": float(np.nanstd(none_rec_ss)),
        "tm_rec_mean":   float(np.nanmean(tm_rec_ss)),   "tm_rec_std":   float(np.nanstd(tm_rec_ss)),
        "ta_rec_mean":   float(np.nanmean(ta_rec_ss)),   "ta_rec_std":   float(np.nanstd(ta_rec_ss)),
        "ft_rec_mean":   float(np.nanmean(ft_rec_ss)),   "ft_rec_std":   float(np.nanstd(ft_rec_ss)),
        "a1_rec_mean":   float(np.nanmean(a1_rec_ss)),   "a1_rec_std":   float(np.nanstd(a1_rec_ss)),
        # F1
        "none_f1_mean": float(np.nanmean(none_f1_ss)), "none_f1_std": float(np.nanstd(none_f1_ss)),
        "tm_f1_mean":   float(np.nanmean(tm_f1_ss)),   "tm_f1_std":   float(np.nanstd(tm_f1_ss)),
        "ta_f1_mean":   float(np.nanmean(ta_f1_ss)),   "ta_f1_std":   float(np.nanstd(ta_f1_ss)),
        "ft_f1_mean":   float(np.nanmean(ft_f1_ss)),   "ft_f1_std":   float(np.nanstd(ft_f1_ss)),
        "a1_f1_mean":   float(np.nanmean(a1_f1_ss)),   "a1_f1_std":   float(np.nanstd(a1_f1_ss)),
    }
    recovery_ta = ss["ta_mean"] - ss["none_mean"]
    recovery_ft = ss["ft_mean"] - ss["none_mean"]

    print("\n" + "=" * 60)
    print(f" AGGREGATE over {len(seeds)} seeds  "
          f"(attack={args.attack}, scale={args.scale}, alpha={args.alpha})")
    print("=" * 60)
    print(f"  {'Defense':<18} {'Acc':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*44}")
    for name, acc_m, acc_s, rec_m, rec_s, f1_m, f1_s in [
        ("No defense",     ss["none_mean"], ss["none_std"], ss["none_rec_mean"], ss["none_rec_std"], ss["none_f1_mean"], ss["none_f1_std"]),
        ("Trimmed mean",   ss["tm_mean"],   ss["tm_std"],   ss["tm_rec_mean"],   ss["tm_rec_std"],   ss["tm_f1_mean"],   ss["tm_f1_std"]),
        ("FLTrust",        ss["ft_mean"],   ss["ft_std"],   ss["ft_rec_mean"],   ss["ft_rec_std"],   ss["ft_f1_mean"],   ss["ft_f1_std"]),
        ("Anchor-1 only",  ss["a1_mean"],   ss["a1_std"],   ss["a1_rec_mean"],   ss["a1_rec_std"],   ss["a1_f1_mean"],   ss["a1_f1_std"]),
        ("Trust-anchored", ss["ta_mean"],   ss["ta_std"],   ss["ta_rec_mean"],   ss["ta_rec_std"],   ss["ta_f1_mean"],   ss["ta_f1_std"]),
    ]:
        print(f"  {name:<18} {acc_m:5.1f}\u00b1{acc_s:.1f}  {rec_m:5.1f}\u00b1{rec_s:.1f}  {f1_m:5.1f}\u00b1{f1_s:.1f}")

    with open("multiseed_results.json", "w") as f:
        json.dump({
            "config": vars(args), "seeds": seeds,
            "steady_state_per_seed": {
                "none":          none_ss.tolist(),
                "trimmed_mean":  tm_ss.tolist(),
                "fltrust":       ft_ss.tolist(),
                "anchor_1_only": a1_ss.tolist(),
                "trust_anchored": ta_ss.tolist(),
            },
            "per_round_mean": {
                "none":          np.nanmean(none_mat, 0).tolist(),
                "trimmed_mean":  np.nanmean(tm_mat,   0).tolist(),
                "fltrust":       np.nanmean(ft_mat,   0).tolist(),
                "anchor_1_only": np.nanmean(a1_mat,   0).tolist(),
                "trust_anchored": np.nanmean(ta_mat,  0).tolist(),
            },
            "per_round_std": {
                "none":          np.nanstd(none_mat, 0).tolist(),
                "trimmed_mean":  np.nanstd(tm_mat,   0).tolist(),
                "fltrust":       np.nanstd(ft_mat,   0).tolist(),
                "anchor_1_only": np.nanstd(a1_mat,   0).tolist(),
                "trust_anchored": np.nanstd(ta_mat,  0).tolist(),
            },
            "per_round_mean_recall": {
                "none":          np.nanmean(none_rec_mat, 0).tolist(),
                "trimmed_mean":  np.nanmean(tm_rec_mat,   0).tolist(),
                "fltrust":       np.nanmean(ft_rec_mat,   0).tolist(),
                "anchor_1_only": np.nanmean(a1_rec_mat,   0).tolist(),
                "trust_anchored": np.nanmean(ta_rec_mat,  0).tolist(),
            },
            "per_round_std_recall": {
                "none":          np.nanstd(none_rec_mat, 0).tolist(),
                "trimmed_mean":  np.nanstd(tm_rec_mat,   0).tolist(),
                "fltrust":       np.nanstd(ft_rec_mat,   0).tolist(),
                "anchor_1_only": np.nanstd(a1_rec_mat,   0).tolist(),
                "trust_anchored": np.nanstd(ta_rec_mat,  0).tolist(),
            },
            "per_round_mean_f1": {
                "none":          np.nanmean(none_f1_mat, 0).tolist(),
                "trimmed_mean":  np.nanmean(tm_f1_mat,   0).tolist(),
                "fltrust":       np.nanmean(ft_f1_mat,   0).tolist(),
                "anchor_1_only": np.nanmean(a1_f1_mat,   0).tolist(),
                "trust_anchored": np.nanmean(ta_f1_mat,  0).tolist(),
            },
            "per_round_std_f1": {
                "none":          np.nanstd(none_f1_mat, 0).tolist(),
                "trimmed_mean":  np.nanstd(tm_f1_mat,   0).tolist(),
                "fltrust":       np.nanstd(ft_f1_mat,   0).tolist(),
                "anchor_1_only": np.nanstd(a1_f1_mat,   0).tolist(),
                "trust_anchored": np.nanstd(ta_f1_mat,  0).tolist(),
            },
            "aggregate": ss,
            "recovery_ta_pp": recovery_ta,
            "recovery_ft_pp": recovery_ft,
        }, f, indent=2)
    print(f"\n  numbers saved -> multiseed_results.json")

    acc_mats = {
        "none": none_mat, "tm": tm_mat, "ft": ft_mat,
        "a1": a1_mat, "ta": ta_mat,
    }
    rec_mats = {
        "none": none_rec_mat, "tm": tm_rec_mat, "ft": ft_rec_mat,
        "a1": a1_rec_mat, "ta": ta_rec_mat,
    }
    make_plots(rounds_axis, acc_mats, rec_mats, ss,
               "multiseed_convergence.png",
               "multiseed_recall.png",
               "multiseed_barplot.png")

    sys.stdout = tee._stdout
    tee.close()
    print(f"  log saved -> {LOG_FILE}")


if __name__ == "__main__":
    main()
 




