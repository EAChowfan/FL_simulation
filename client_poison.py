"""
client_poison.py  (LSTM edition)
----------------------------------
Malicious Flower client. Same FBSLSTM training as an honest client, then
sabotages the resulting update before upload (scaled sign-flip by default;
Gaussian-noise mode available). The attack is applied tensor-by-tensor since
the LSTM parameters are a list of ndarrays.

Run:
    python client_poison.py --data poison_data.csv
    python client_poison.py --data poison_data.csv --attack noise --scale 8
"""

import argparse
import os
import sys
import warnings

import flwr as fl
import numpy as np
import pandas as pd

from model import (
    build_model,
    get_parameters as model_get_params,
    set_parameters as model_set_params,
    train_local, encode_frame, load_vocab,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--attack", type=str, default="sign_flip",
                    choices=["sign_flip", "noise", "label_flip"])
parser.add_argument("--scale", type=float, default=10.0)
parser.add_argument("--vocab", type=str, default="vocab.json")
parser.add_argument("--local-epochs", type=int, default=1)
args = parser.parse_args()

if not os.path.exists(args.vocab):
    sys.exit(f"ERROR: {args.vocab} not found — run: python generate_data.py")

vocab = load_vocab(args.vocab)
df = pd.read_csv(args.data)
X, y = encode_frame(df, vocab)


class PoisonClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model(len(vocab))

    def get_parameters(self, config):
        return model_get_params(self.model)

    def fit(self, parameters, config):
        model_set_params(self.model, parameters)

        if args.attack == "label_flip":
            # Stealth attack: relabel every FBS session as honest, then train
            # normally. The update is geometrically clean (no sign flip, no
            # scaling) so cosine-similarity checks pass — but the model is
            # specifically trained to miss FBS attacks, which the BR probes catch.
            y_stealth = np.where(y == 1, 0, y)
            train_local(self.model, X, y_stealth, epochs=args.local_epochs)
            poisoned = model_get_params(self.model)
            l2 = float(np.sqrt(sum(np.sum(p ** 2) for p in poisoned)))
            print(f"[Poison Client] STEALTH label_flip "
                  f"({int((y == 1).sum())} FBS sessions relabeled) L2={l2:.3f}")
            return poisoned, len(X), {}

        global_params = [p.copy() for p in model_get_params(self.model)]  # copies before training
        train_local(self.model, X, y, epochs=args.local_epochs)
        honest = model_get_params(self.model)
        # Delta = what this round of training actually changed.
        # Attacks operate on the delta so the poisoned weights stay in a
        # reasonable magnitude range and don't saturate LSTM activations.
        delta = [h - g for h, g in zip(honest, global_params)]
        if args.attack == "sign_flip":
            # Push model in opposite direction, amplified by scale.
            poisoned = [g + (-args.scale * d)
                        for g, d in zip(global_params, delta)]
        else:  # noise
            poisoned = [g + d + np.random.normal(0, args.scale, size=d.shape)
                        for g, d in zip(global_params, delta)]
        delta_l2 = float(np.sqrt(sum(np.sum(d ** 2) for d in delta)))
        l2 = float(np.sqrt(sum(np.sum(p ** 2) for p in poisoned)))
        print(f"[Poison Client] MALICIOUS ({args.attack}, scale={args.scale}) "
              f"delta_L2={delta_l2:.3f}  poisoned_L2={l2:.3f}")
        return poisoned, len(X), {}

    def evaluate(self, parameters, config):
        return 1.0, len(X), {"accuracy": 0.0}


if __name__ == "__main__":
    fl.client.start_client(server_address=args.server_address,
                           client=PoisonClient().to_client())
