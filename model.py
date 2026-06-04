"""
model.py
--------
Shared model + data-encoding utilities for the LSTM-based federated FBS
detector. Imported by the clients and the server so they agree on architecture,
tokenization, and how weights move in/out of Flower.

Model: a deliberately tiny sequence classifier
    Embedding(vocab, 16) -> LSTM(16, hidden=32, 1 layer) -> Linear(32, 2)
~5-6k parameters: trains in seconds on CPU, small enough for the eventual
on-device (MODI/ExecuTorch) story.

Input: per-session NAS+RRC L3 token sequences, encoded as fixed-length integer
arrays (pad/truncate to MAX_LEN, PAD=0).

Run directly for a central (non-federated) sanity check:
    python model.py            # trains on sessions_full.csv, prints val accuracy
"""

import json

import numpy as np
import torch
import torch.nn as nn

MAX_LEN = 24
PAD_ID = 0


# ----------------------------------------------------------------------
# Data encoding
# ----------------------------------------------------------------------
def load_vocab(path="vocab.json"):
    with open(path) as f:
        return json.load(f)


def tokenize(seq, vocab, max_len=MAX_LEN):
    """Session string -> fixed-length list of token ids (padded/truncated)."""
    ids = [vocab.get(tok, PAD_ID) for tok in str(seq).split()][:max_len]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    return ids


def encode_frame(df, vocab, max_len=MAX_LEN):
    """DataFrame with 'seq','label' -> (X int64 [N,L], y int64 [N])."""
    X = np.array([tokenize(s, vocab, max_len) for s in df["seq"]], dtype=np.int64)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class FBSLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden=32, num_classes=2,
                 pad_id=PAD_ID):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embed(x)                 # (B, L, E)
        _, (h_n, _) = self.lstm(emb)        # h_n: (1, B, H)
        return self.fc(h_n[-1])             # (B, num_classes)


def build_model(vocab_size, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return FBSLSTM(vocab_size)


# ----------------------------------------------------------------------
# Flower <-> PyTorch parameter helpers
# Updates are a LIST of ndarrays (one per state_dict tensor), in key order.
# ----------------------------------------------------------------------
def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, params):
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(np.array(p)) for k, p in zip(keys, params)}
    model.load_state_dict(state, strict=True)


def train_local(model, X, y, epochs=1, lr=0.1, momentum=0.9, batch_size=64,
                seed=None):
    """A few SGD epochs of local training. Returns the model (in place)."""
    if seed is not None:
        torch.manual_seed(seed)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    logits = model(torch.tensor(X))
    return (logits.argmax(1).numpy() == y).mean()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.tensor(X)).argmax(1).numpy()


# ----------------------------------------------------------------------
# Central sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    torch.manual_seed(42)
    np.random.seed(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    vocab = load_vocab()
    df = pd.read_csv("sessions_full.csv")
    X, y = encode_frame(df, vocab)

    idx = np.random.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, va = idx[:cut], idx[cut:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

    model = build_model(len(vocab), seed=42)
    print(f"vocab={len(vocab)}  params="
          f"{sum(p.numel() for p in model.parameters())}  "
          f"train={len(Xtr)} val={len(Xva)}")

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    for epoch in range(20):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:2d}  val acc {accuracy(model, Xva, yva) * 100:5.1f}%")