"""
client_normal.py  (LSTM edition)
---------------------------------
Honest Flower client. Loads its NAS+RRC attach sessions, encodes them as
fixed-length integer token sequences via the shared vocab, and trains the
tiny FBSLSTM from model.py (one local epoch/round) on global weights
received from the server.

Run:
    python client_normal.py --data client1_data.csv --cid 1
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
    train_local, encode_frame, load_vocab, accuracy,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--cid", type=str, default="normal")
parser.add_argument("--vocab", type=str, default="vocab.json")
parser.add_argument("--local-epochs", type=int, default=1)
args = parser.parse_args()

if not os.path.exists(args.vocab):
    sys.exit(f"ERROR: {args.vocab} not found — run: python generate_data.py")

vocab = load_vocab(args.vocab)
df = pd.read_csv(args.data)
X, y = encode_frame(df, vocab)


class NormalClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model(len(vocab))

    def get_parameters(self, config):
        return model_get_params(self.model)

    def fit(self, parameters, config):
        model_set_params(self.model, parameters)
        train_local(self.model, X, y, epochs=args.local_epochs)
        params = model_get_params(self.model)
        l2 = float(np.sqrt(sum(np.sum(p ** 2) for p in params)))
        print(f"[Normal Client {args.cid}] honest update "
              f"({len(X)} sessions, {100 * y.mean():.0f}% FBS) "
              f"L2={l2:.3f}")
        return params, len(X), {}

    def evaluate(self, parameters, config):
        model_set_params(self.model, parameters)
        acc = float(accuracy(self.model, X, y))
        return 1.0 - acc, len(X), {"accuracy": acc}


if __name__ == "__main__":
    fl.client.start_client(server_address=args.server_address,
                           client=NormalClient().to_client())
