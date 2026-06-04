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
parser.add_argument("--mag-bound", type=float, default=5.0,
                    help="L2 norm cap on client weight deltas for trust-anchored "
                         "aggregation. Set at the ~95th percentile of expected honest "
                         "update norms for FBSLSTM (D=6882, lr=0.1, mom=0.9, 1 epoch).")
args = parser.parse_args()

if not os.path.exists("vocab.json"):
    sys.exit("ERROR: vocab.json not found — run: python generate_data.py")

VOCAB = load_vocab("vocab.json")
VOCAB_SIZE = len(VOCAB)

val_df = pd.read_csv(args.val)
X_val, y_val = encode_frame(val_df, VOCAB)

# Behavior-rule probes derived from UE-Guard BR-1 to BR-43 (Abella et al.).
# Each probe's ground-truth label is derivable from 3GPP normative clauses alone,
# making them independent of training data distribution.
#
# Honest probes (label 0) — all relevant rules satisfied:
#   BR-8  (no null algorithm), BR-25 (ciphered past security),
#   BR-27 (auth present),     BR-31 (SMC follows Setup Complete)
#
# FBS probes (label 1) — one or more rules violated:
#   BR-8  : EMM_SMC_NULL present (null cipher/integrity, TS 33.401 §5.1.3)
#   BR-25 : SHT0_PLAIN past security establishment (TS 24.301 §4.4.4)
#   BR-27 : authentication skipped (TS 24.301 §5.4.2.5)
#   BR-31 : no Security Mode Command following RRC Setup Complete (TS 36.331 §5.3.4)
#   BR-35 : EMM_SMC_NULL without completing the exchange (TS 36.331 §5.3.4)
_BR_PROBE_STRS = [
    # Honest — BR-8/25/27/31 all satisfied: full auth + non-null SMC + ciphered attach
    ("RRC_CONN_REQ RRC_CONN_SETUP RRC_CONN_SETUP_CMP RRC_UL_INFO_NAS "
     "EMM_ATTACH_REQ SHT0_PLAIN RRC_DL_INFO_NAS EMM_IDENTITY_REQ "
     "EMM_IDENTITY_RES EMM_AUTH_REQ EMM_AUTH_RES RRC_SEC_MODE_CMD "
     "EMM_SMC EMM_SMC_CMP RRC_SEC_MODE_CMP SHT4_NEWCTX SHT2_CIPHERED "
     "EMM_ATTACH_ACCEPT EMM_ATTACH_CMP", 0),
    # Honest — no identity step variant; BR-8/25/27/31 still satisfied
    ("RRC_CONN_REQ RRC_CONN_SETUP RRC_CONN_SETUP_CMP EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_AUTH_REQ EMM_AUTH_RES EMM_SMC EMM_SMC_CMP "
     "SHT4_NEWCTX SHT2_CIPHERED EMM_ATTACH_ACCEPT", 0),
    # FBS — BR-8 violated: null cipher algorithm (EMM_SMC_NULL) + RF anomaly
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RF_TAC_CHANGE RRC_CONN_SETUP "
     "EMM_ATTACH_REQ SHT0_PLAIN EMM_IDENTITY_REQ EMM_IDENTITY_RES "
     "EMM_SMC_NULL EMM_SMC_CMP SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
    # FBS — BR-8 + BR-25 violated: null cipher + SHT0_PLAIN continuation
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RRC_CONN_SETUP EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_SMC_NULL SHT0_PLAIN EMM_ATTACH_ACCEPT EMM_ATTACH_CMP", 1),
    # FBS — BR-27 + BR-8 violated: auth skipped + null cipher
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RF_TAC_CHANGE RRC_CONN_SETUP "
     "EMM_ATTACH_REQ SHT0_PLAIN EMM_IDENTITY_REQ EMM_IDENTITY_RES "
     "EMM_SMC_NULL SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
    # FBS — BR-31 violated: no security mode at all; attach proceeds in plaintext
    ("RRC_CONN_REQ RF_REFPOWER_HIGH RRC_CONN_SETUP RRC_CONN_SETUP_CMP "
     "EMM_ATTACH_REQ SHT0_PLAIN EMM_IDENTITY_REQ EMM_IDENTITY_RES "
     "SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
    # FBS — BR-25 violated: SHT0_PLAIN where ciphering is required (minimal trace)
    ("RRC_CONN_REQ RF_TAC_CHANGE RRC_CONN_SETUP EMM_ATTACH_REQ "
     "SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
    # FBS — BR-8 + BR-35 violated: null cipher, no SMC_CMP follow-through
    ("RRC_CONN_REQ RRC_CONN_SETUP EMM_ATTACH_REQ SHT0_PLAIN "
     "EMM_SMC_NULL SHT0_PLAIN EMM_ATTACH_ACCEPT", 1),
]
BR_PROBES = [
    (np.array([tokenize(s, VOCAB, MAX_LEN)], dtype=np.int64), y)
    for s, y in _BR_PROBE_STRS
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
    def __init__(self, defense_type="none", trim=1, init_params=None, **kwargs):
        super().__init__(**kwargs)
        self.defense_type = defense_type
        self.trim = trim
        self._global_params = init_params  # tracks weights sent to clients each round

    def _trust_anchored_aggregate(
        self, all_params: list, n_tensors: int
    ) -> list:
        global_params = self._global_params

        # 1. Client weight deltas: what each client changed from the global model.
        client_deltas = [
            [c - g for c, g in zip(cp, global_params)]
            for cp in all_params
        ]

        # 2. Magnitude bound (M=5.0 by default).
        #    Clips each delta to at most MAGNITUDE_BOUND in L2 norm.
        #    Grounded in the ~95th percentile of honest update norms for FBSLSTM
        #    (D=6882, lr=0.1, momentum=0.9, 1 local epoch, ~300-500 sessions).
        #    Under a ×10 sign-flip attack the poisoned norm (~15-45) is reduced
        #    3-9× before behavioral trust weighting applies.
        M = args.mag_bound
        bounded_deltas = []
        raw_norms = []
        for delta in client_deltas:
            norm = float(np.sqrt(sum(np.sum(d ** 2) for d in delta)))
            raw_norms.append(norm)
            if norm > M:
                bounded_deltas.append([d * (M / norm) for d in delta])
            else:
                bounded_deltas.append(delta)

        # 3. Behavioral trust anchor — score each client by how well its submitted
        #    model's predictions align with formally verified 3GPP behavior rules
        #    (UE-Guard BR-8, BR-25, BR-27, BR-31, BR-35).
        #    A sign-flipped model systematically mislabels the FBS probes → score ≈ 0.
        br_scores = []
        for cp in all_params:
            probe_model = build_model(VOCAB_SIZE)
            model_set_params(probe_model, cp)
            correct = sum(
                int(predict(probe_model, Xp)[0] == yp)
                for Xp, yp in BR_PROBES
            )
            br_scores.append(correct / len(BR_PROBES))

        trust_sum = sum(br_scores) + 1e-9
        weights = [s / trust_sum for s in br_scores]

        print(f"    [delta norms] {[f'{n:.2f}' for n in raw_norms]}  "
              f"(bound={M})")
        print(f"    [BR scores  ] {[f'{s:.2f}' for s in br_scores]}")
        print(f"    [weights    ] {[f'{w:.2f}' for w in weights]}")

        # 4. Weighted sum of bounded deltas, added back to global weights.
        aggregated_delta = [
            sum(w * bd[t] for w, bd in zip(weights, bounded_deltas))
            for t in range(n_tensors)
        ]
        return [g + d for g, d in zip(global_params, aggregated_delta)]

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

        # Store for next round's delta computation.
        self._global_params = aggregated

        agg_norm = float(np.sqrt(sum(np.sum(a ** 2) for a in aggregated)))
        print(f"    aggregated weight L2 norm = {agg_norm:.4f}")
        return ndarrays_to_parameters(aggregated), {}


def main():
    if args.checkpoint and os.path.exists(args.checkpoint):
        data = np.load(args.checkpoint, allow_pickle=True)
        init_params = [data[k] for k in data.files]
        print(f"[Server] Resuming from checkpoint: {args.checkpoint}")
    else:
        init_model = build_model(VOCAB_SIZE, seed=42)
        init_params = model_get_params(init_model)

    initial = ndarrays_to_parameters(init_params)

    strategy = ByzantineRobustStrategy(
        defense_type=args.defense,
        trim=args.trim,
        init_params=init_params,
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
