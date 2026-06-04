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
                    choices=["sign_flip", "noise"])
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
        train_local(self.model, X, y, epochs=args.local_epochs)
        honest = model_get_params(self.model)
        if args.attack == "sign_flip":
            poisoned = [-args.scale * p for p in honest]
        else:
            poisoned = [p + np.random.normal(0, args.scale, size=p.shape)
                        for p in honest]
        l2 = float(np.sqrt(sum(np.sum(p ** 2) for p in poisoned)))
        print(f"[Poison Client] MALICIOUS ({args.attack}, scale={args.scale}) "
              f"L2={l2:.3f}")
        return poisoned, len(X), {}

    def evaluate(self, parameters, config):
        return 1.0, len(X), {"accuracy": 0.0}


if __name__ == "__main__":
    fl.client.start_client(server_address=args.server_address,
                           client=PoisonClient().to_client())
