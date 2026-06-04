"""
server.py  (LSTM edition)
--------------------------
Flower server for the FBS-detection FL poisoning PoC.

A custom strategy switches between:
  --defense none          plain coordinate-wise mean (vulnerable FedAvg)
  --defense trimmed_mean  coordinate-wise trimmed mean (Byzantine-robust)

After aggregating each round, the server scores the global FBSLSTM model
(from model.py) against a held-out validation set and prints accuracy.
Per-tensor aggregation: each of the 7 LSTM parameter tensors is aggregated
independently, preserving shape through the full round-trip.

Run (defense off, then on):
    python server.py --defense none         --rounds 5 --clients 5
    python server.py --defense trimmed_mean --rounds 5 --clients 5 --trim 1
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (
    FitRes, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from model import (
    build_model, encode_frame, evaluate_metrics,
    get_parameters as model_get_params,
    set_parameters as model_set_params,
    load_vocab, predict, tokenize, MAX_LEN,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
parser.add_argument("--defense", type=str, default="none",
                    choices=["none", "trimmed_mean", "trust_anchored"])
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--clients", type=int, default=5,
                    help="total clients expected (honest + poison)")
parser.add_argument("--trim", type=int, default=1,
                    help="number trimmed from each end per coordinate")
parser.add_argument("--val", type=str, default="val_data.csv")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="path to load/save global model weights (.npz)")
args = parser.parse_args()

if not os.path.exists("vocab.json"):
    sys.exit("ERROR: vocab.json not found — run: python generate_data.py")

VOCAB = load_vocab("vocab.json")
VOCAB_SIZE = len(VOCAB)

val_df = pd.read_csv(args.val)
X_val, y_val = encode_frame(val_df, VOCAB)

# 3GPP oracle probes — ground truth is derivable from the spec alone.
# Honest: full auth chain + proper ciphering (label 0).
# FBS:    null-cipher algorithm or auth skipped + plaintext past security (label 1).
_SPEC_PROBE_STRS = [
    ("RRC_CONN_REQ RRC_CONN_SETUP RRC_CONN_SETUP_CMP RRC_UL_INFO_NAS "
     "EMM_ATTACH_REQ SHT0_PLAIN RRC_DL_INFO_NAS EMM_IDENTITY_REQ "
     "EMM_IDENTITY_RES EMM_AUTH_REQ EMM_AUTH_RES RRC_SEC_MODE_CMD "
     "EMM_SMC EMM_SMC_CMP RRC_SEC_MODE_CMP SHT4_NEWCTX SHT2_CIPHERED "
     "EMM_ATTACH_ACCEPT EMM_ATTACH_CMP", 0),
    ("RRC_CONN_REQ RRC_CONN_SETUP RRC_CONN_SETUP_CMP EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_AUTH_REQ EMM_AUTH_RES EMM_SMC EMM_SMC_CMP "
     "SHT4_NEWCTX SHT2_CIPHERED EMM_ATTACH_ACCEPT", 0),
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RF_TAC_CHANGE RRC_CONN_SETUP "
     "EMM_ATTACH_REQ SHT0_PLAIN EMM_IDENTITY_REQ EMM_IDENTITY_RES "
     "EMM_SMC_NULL EMM_SMC_CMP SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RRC_CONN_SETUP EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_IDENTITY_REQ EMM_IDENTITY_RES "
     "EMM_SMC_NULL SHT0_PLAIN EMM_ATTACH_ACCEPT EMM_ATTACH_CMP", 1),
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RF_TAC_CHANGE EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
]
SPEC_PROBES = [
    (np.array([tokenize(s, VOCAB, MAX_LEN)], dtype=np.int64), y)
    for s, y in _SPEC_PROBE_STRS
]


_last_params: list = []  # captures final-round weights for checkpoint


def evaluate_fn(server_round: int, parameters: NDArrays,
                config: Dict[str, Scalar]):
    model = build_model(VOCAB_SIZE)
    model_set_params(model, parameters)
    m = evaluate_metrics(model, X_val, y_val)
    acc = m["accuracy"]
    print(f"    >> [Round {server_round}] global model "
          f"VALIDATION accuracy = {acc * 100:5.2f}%   (defense={args.defense})")
    print(f"       precision={m['precision']*100:5.2f}%  "
          f"recall={m['recall']*100:5.2f}%  "
          f"F1={m['f1']*100:5.2f}%  "
          f"[TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}]")
    _last_params.clear()
    _last_params.extend(parameters)
    return 1.0 - acc, m


class ByzantineRobustStrategy(fl.server.strategy.FedAvg):
    def __init__(self, defense_type="none", trim=1, **kwargs):
        super().__init__(**kwargs)
        self.defense_type = defense_type
        self.trim = trim

    def _trust_anchored_aggregate(
        self, all_params: list, n_tensors: int
    ) -> list:
        n = len(all_params)

        # 1. Directional consistency — cosine similarity with coordinate-wise median.
        #    Sign-flip attacker points opposite the honest consensus → score ≈ 0.
        flat = [np.concatenate([p.flatten() for p in cp]) for cp in all_params]
        median_flat = np.median(np.stack(flat, axis=0), axis=0)
        med_norm = float(np.linalg.norm(median_flat)) + 1e-9
        dir_scores = [
            max(0.0, float(np.dot(f, median_flat)) / (float(np.linalg.norm(f)) + 1e-9) / med_norm)
            for f in flat
        ]

        # 2. Spec-rule alignment — evaluate each client's weights on 3GPP oracle probes.
        #    A poisoned model (sign-flipped weights) systematically mislabels these.
        spec_scores = []
        for cp in all_params:
            probe_model = build_model(VOCAB_SIZE)
            model_set_params(probe_model, cp)
            correct = sum(
                int(predict(probe_model, Xp)[0] == yp)
                for Xp, yp in SPEC_PROBES
            )
            spec_scores.append(correct / len(SPEC_PROBES))

        # 3. Trust = equal blend of both signals; normalise to weights summing to 1.
        trust = [0.5 * d + 0.5 * s for d, s in zip(dir_scores, spec_scores)]
        trust_sum = sum(trust) + 1e-9
        weights = [t / trust_sum for t in trust]

        print(f"    [trust-dir ] {[f'{d:.2f}' for d in dir_scores]}")
        print(f"    [trust-spec] {[f'{s:.2f}' for s in spec_scores]}")
        print(f"    [weights   ] {[f'{w:.2f}' for w in weights]}")

        # 4. Weighted aggregation.
        return [
            sum(w * cp[t] for w, cp in zip(weights, all_params))
            for t in range(n_tensors)
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(res.parameters) for _, res in results]
        n = len(all_params)
        n_tensors = len(all_params[0])

        if self.defense_type == "none":
            print(f"[Round {server_round}] defense=NONE  -> plain mean of {n} updates")
            aggregated = []
            for t in range(n_tensors):
                stacked = np.stack([cp[t] for cp in all_params], axis=0)
                aggregated.append(np.mean(stacked, axis=0))

        elif self.defense_type == "trimmed_mean":
            k = self.trim
            if n - 2 * k < 1:
                k = max(0, (n - 1) // 2)
            print(f"[Round {server_round}] defense=TRIMMED_MEAN "
                  f"(trim {k}/end, kept {n - 2 * k}/{n}) per tensor")
            aggregated = []
            for t in range(n_tensors):
                stacked = np.stack([cp[t] for cp in all_params], axis=0)
                sorted_arr = np.sort(stacked, axis=0)
                kept = sorted_arr[k: n - k]
                aggregated.append(np.mean(kept, axis=0))

        else:  # trust_anchored
            print(f"[Round {server_round}] defense=TRUST_ANCHORED ({n} clients)")
            aggregated = self._trust_anchored_aggregate(all_params, n_tensors)

        agg_norm = float(np.sqrt(sum(np.sum(a ** 2) for a in aggregated)))
        print(f"    aggregated weight L2 norm = {agg_norm:.4f}")
        return ndarrays_to_parameters(aggregated), {}


def main():
    if args.checkpoint and os.path.exists(args.checkpoint):
        data = np.load(args.checkpoint, allow_pickle=True)
        init_params = [data[k] for k in data.files]
        initial = ndarrays_to_parameters(init_params)
        print(f"[Server] Resuming from checkpoint: {args.checkpoint}")
    else:
        init_model = build_model(VOCAB_SIZE, seed=42)
        initial = ndarrays_to_parameters(model_get_params(init_model))

    strategy = ByzantineRobustStrategy(
        defense_type=args.defense,
        trim=args.trim,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        min_evaluate_clients=0,
        fraction_evaluate=0.0,
        initial_parameters=initial,
        evaluate_fn=evaluate_fn,
    )
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    if args.checkpoint and _last_params:
        np.savez(args.checkpoint, *_last_params)
        print(f"[Server] Checkpoint saved -> {args.checkpoint}")


if __name__ == "__main__":
    main()
