"""
run_experiment.py
-----------------
One-command launcher for the FBS-detection FL poisoning PoC.
 
It runs the full federation twice -- once with the aggregation defense OFF
(plain FedAvg) and once with it ON (trimmed mean) -- against an identical
1-poison / 4-honest client setup, then prints a side-by-side accuracy table and
the headline recovery number for the poster.
 
Usage:
    python run_experiment.py                 # uses existing CSVs
    python run_experiment.py --regen         # regenerate data first
    python run_experiment.py --scale 15      # stronger sign-flip attack
 
Everything runs locally over 127.0.0.1; no Mininet required.
"""
 
import argparse
import json
import re
import socket
import subprocess
import sys
import time
 
import numpy as np
 
PORT = 8080
SERVER_ADDR = f"127.0.0.1:{PORT}"
ROUNDS = 15
N_CLIENTS = 5  # 4 honest + 1 poison
HONEST_DATA = [f"client{i}_data.csv" for i in range(1, 5)]
POISON_DATA = "poison_data.csv"
 
ACC_RE = re.compile(r"\[Round (\d+)\].*VALIDATION accuracy =\s*([\d.]+)%")
 
 
def run_one(defense, scale, trim, attack, mag_bound=5.0, no_attack=False,
            local_epochs=5):
    """Run a single federation and return {round: accuracy} parsed from server."""
    print(f"\n{'='*60}")
    print(f" RUNNING federation  ->  defense = {defense.upper()}"
          f"{' [NO ATTACK]' if no_attack else ''}")
    print(f"{'='*60}")

    server = subprocess.Popen(
        [sys.executable, "server.py",
         "--server_address", SERVER_ADDR,
         "--defense", defense,
         "--rounds", str(ROUNDS),
         "--clients", str(N_CLIENTS),
         "--trim", str(trim),
         "--mag-bound", str(mag_bound)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Wait until the server socket is accepting connections (up to 15 s).
    for _ in range(150):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        server.kill()
        raise RuntimeError("Server did not become ready within 15 s")
 
    clients = []
    for i, data in enumerate(HONEST_DATA, start=1):
        clients.append(subprocess.Popen(
            [sys.executable, "client_normal.py",
             "--server_address", SERVER_ADDR,
             "--data", data, "--cid", str(i),
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    if no_attack:
        clients.append(subprocess.Popen(
            [sys.executable, "client_normal.py",
             "--server_address", SERVER_ADDR,
             "--data", POISON_DATA, "--cid", "5",
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    else:
        clients.append(subprocess.Popen(
            [sys.executable, "client_poison.py",
             "--server_address", SERVER_ADDR,
             "--data", POISON_DATA,
             "--attack", attack, "--scale", str(scale),
             "--local-epochs", str(local_epochs)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
 
    out, _ = server.communicate(timeout=600)
    for c in clients:
        try:
            c.wait(timeout=5)
        except subprocess.TimeoutExpired:
            c.kill()
 
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
            f"run_one(defense={defense!r}) parsed zero accuracy entries — "
            "check that server.py log format matches ACC_RE above"
        )

    return acc
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="regenerate the synthetic dataset first")
    ap.add_argument("--scale", type=float, default=10.0,
                    help="poison attack magnitude")
    ap.add_argument("--trim", type=int, default=1,
                    help="trimmed-mean trim count per end")
    ap.add_argument("--attack", type=str, default="sign_flip",
                    choices=["sign_flip", "noise", "label_flip"])
    ap.add_argument("--mag-bound", type=float, default=5.0,
                    help="magnitude bound for trust-anchored aggregation")
    ap.add_argument("--no-attack", action="store_true",
                    help="replace poison client with honest client — baseline check")
    ap.add_argument("--local-epochs", type=int, default=5,
                    help="local SGD epochs per FL round")
    args = ap.parse_args()
 
    import os
    if args.regen or not os.path.exists("vocab.json"):
        print("Generating dataset (NAS+RRC sequence format)...")
        subprocess.run([sys.executable, "generate_data.py"], check=True)
 
    none_acc = run_one("none", args.scale, args.trim, args.attack,
                       args.mag_bound, args.no_attack, args.local_epochs)
    time.sleep(2)
    tm_acc = run_one("trimmed_mean", args.scale, args.trim, args.attack,
                     args.mag_bound, args.no_attack, args.local_epochs)
    time.sleep(2)
    ft_acc = run_one("fltrust", args.scale, args.trim, args.attack,
                     args.mag_bound, args.no_attack, args.local_epochs)
    time.sleep(2)
    ta_acc = run_one("trust_anchored", args.scale, args.trim, args.attack,
                     args.mag_bound, args.no_attack, args.local_epochs)

    # ---- comparison table ----
    rounds = sorted(set(none_acc) | set(tm_acc) | set(ta_acc) | set(ft_acc))
    print(f"\n\n{'='*84}")
    print(f" RESULTS  (attack={args.attack}, scale={args.scale}, "
          f"1 of {N_CLIENTS} clients poisoned)")
    print(f"{'='*84}")
    print(f"{'Round':>6} | {'No Defense':>12} | {'Trimmed Mean':>13} | "
          f"{'FLTrust':>10} | {'Trust-Anchored':>15}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*15}")
    for r in rounds:
        ns  = f"{none_acc[r]:6.2f}%" if r in none_acc else "      --"
        ts  = f"{tm_acc[r]:6.2f}%"   if r in tm_acc   else "      --"
        fs  = f"{ft_acc[r]:6.2f}%"   if r in ft_acc   else "      --"
        as_ = f"{ta_acc[r]:6.2f}%"   if r in ta_acc   else "      --"
        print(f"{r:>6} | {ns:>12} | {ts:>13} | {fs:>10} | {as_:>15}")

    K = min(10, len(rounds) - 1)
    tail    = [r for r in rounds if r != 0][-K:]
    n_tail  = np.array([none_acc[r] for r in tail if r in none_acc], dtype=float)
    t_tail  = np.array([tm_acc[r]   for r in tail if r in tm_acc],   dtype=float)
    f_tail  = np.array([ft_acc[r]   for r in tail if r in ft_acc],   dtype=float)
    ta_tail = np.array([ta_acc[r]   for r in tail if r in ta_acc],   dtype=float)

    n_mean,  n_std  = float(np.mean(n_tail)),  float(np.std(n_tail))
    t_mean,  t_std  = float(np.mean(t_tail)),  float(np.std(t_tail))
    f_mean,  f_std  = float(np.mean(f_tail)),  float(np.std(f_tail))
    ta_mean, ta_std = float(np.mean(ta_tail)), float(np.std(ta_tail))

    print(f"\n  Steady-state (last {len(tail)} rounds):")
    print(f"    No defense      : {n_mean:6.2f}% +/- {n_std:.2f}")
    print(f"    Trimmed mean    : {t_mean:6.2f}% +/- {t_std:.2f}  "
          f"(+{t_mean - n_mean:.2f} pp)")
    print(f"    FLTrust         : {f_mean:6.2f}% +/- {f_std:.2f}  "
          f"(+{f_mean - n_mean:.2f} pp)")
    print(f"    Trust-anchored  : {ta_mean:6.2f}% +/- {ta_std:.2f}  "
          f"(+{ta_mean - n_mean:.2f} pp)")

    with open("results.json", "w") as f:
        json.dump({
            "none": none_acc, "trimmed_mean": tm_acc,
            "fltrust": ft_acc, "trust_anchored": ta_acc,
            "scale": args.scale, "attack": args.attack,
            "steady_state": {
                "rounds_averaged": tail,
                "none_mean":  n_mean,  "none_std":  n_std,
                "tm_mean":    t_mean,  "tm_std":    t_std,
                "ft_mean":    f_mean,  "ft_std":    f_std,
                "ta_mean":    ta_mean, "ta_std":    ta_std,
            },
        }, f, indent=2)
    print(f"\n  saved -> results.json")
 
 
if __name__ == "__main__":
    main()