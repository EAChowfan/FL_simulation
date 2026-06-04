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
    accuracy, build_model, encode_frame,
    get_parameters as model_get_params,
    set_parameters as model_set_params,
    load_vocab,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
parser.add_argument("--defense", type=str, default="none",
                    choices=["none", "trimmed_mean"])
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


_last_params: list = []  # captures final-round weights for checkpoint


def evaluate_fn(server_round: int, parameters: NDArrays,
                config: Dict[str, Scalar]):
    model = build_model(VOCAB_SIZE)
    model_set_params(model, parameters)
    acc = float(accuracy(model, X_val, y_val))
    print(f"    >> [Round {server_round}] global model "
          f"VALIDATION accuracy = {acc * 100:5.2f}%   (defense={args.defense})")
    _last_params.clear()
    _last_params.extend(parameters)
    return 1.0 - acc, {"accuracy": acc}


class ByzantineRobustStrategy(fl.server.strategy.FedAvg):
    def __init__(self, defense_type="none", trim=1, **kwargs):
        super().__init__(**kwargs)
        self.defense_type = defense_type
        self.trim = trim

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Each client sends a list of ndarrays, one per model tensor (7 for FBSLSTM).
        all_params = [parameters_to_ndarrays(res.parameters) for _, res in results]
        n = len(all_params)
        n_tensors = len(all_params[0])

        k = self.trim
        if self.defense_type == "trimmed_mean" and n - 2 * k < 1:
            k = max(0, (n - 1) // 2)

        if self.defense_type == "none":
            print(f"[Round {server_round}] defense=NONE  -> plain mean of "
                  f"{n} updates")
        else:
            print(f"[Round {server_round}] defense=TRIMMED_MEAN "
                  f"(trim {k}/end, kept {n - 2 * k}/{n}) per tensor")

        aggregated = []
        for t in range(n_tensors):
            stacked = np.stack([cp[t] for cp in all_params], axis=0)  # (n, *shape)
            if self.defense_type == "none":
                agg_t = np.mean(stacked, axis=0)
            else:
                sorted_arr = np.sort(stacked, axis=0)
                kept = sorted_arr[k: n - k]
                agg_t = np.mean(kept, axis=0)
            aggregated.append(agg_t)

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
