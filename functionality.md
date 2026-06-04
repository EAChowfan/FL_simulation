# FL Simulation — Functionality & Design Reference

## What This System Does

This is a **federated learning (FL) simulation** for detecting False Base Stations (FBS) in LTE networks. A False Base Station (also called an IMSI catcher or rogue eNodeB) impersonates a legitimate cell tower to intercept UE (user equipment) communications. The attack is detectable by analyzing the sequence of NAS (Non-Access Stratum) and RRC (Radio Resource Control) Layer 3 messages exchanged during a UE attach procedure — FBS attacks follow a recognizable pattern (skipped authentication, null security algorithms, plaintext messages past the security point).

The simulation answers one research question: **does a trimmed-mean Byzantine-robust aggregation strategy defend FBS-detection accuracy when one of five federated clients is poisoning its model updates?**

---

## System Architecture

```
generate_data.py
      │
      ▼
sessions_full.csv  vocab.json  client1-4_data.csv  poison_data.csv  val_data.csv
      │                                │                   │               │
      │                         client_normal.py    client_poison.py       │
      │                         (×4 honest)         (×1 malicious)         │
      │                                │                   │               │
      │                         [Flower gRPC — 127.0.0.1:8080]             │
      │                                │                   │               │
      └──────────────────────► server.py ◄────────────────-┘               │
                                       │                                   │
                               aggregate_fit()                             │
                               (FedAvg or trimmed mean)                    │
                                       │                                   │
                               evaluate_fn() ◄─────────────────────────────┘
                                       │
                               [Round X] VALIDATION accuracy = Y%
                                       │
                          multiseed.py / launcher.py (parse + plot)
```

---

## Files and Their Roles

| File | Role |
|---|---|
| `generate_data.py` | Synthesizes NAS+RRC session sequences; partitions them across clients using a Dirichlet distribution for non-IID simulation |
| `vocab.json` | Shared 26-token vocabulary (L3 message types + RF anomaly tokens) written by `generate_data.py` |
| `model.py` | Defines `FBSLSTM`, encoding utilities, and Flower-compatible parameter helpers |
| `server.py` | Flower server: aggregates client updates (FedAvg or trimmed mean), evaluates global model each round |
| `client_normal.py` | Honest Flower client: trains FBSLSTM locally, sends honest update |
| `client_poison.py` | Malicious Flower client: trains FBSLSTM locally, then corrupts the update before sending |
| `launcher.py` | Single-run orchestrator: spawns server + 5 clients, runs both defense conditions, prints comparison table |
| `multiseed.py` | Multi-seed harness: repeats the experiment over N seeds, aggregates mean ± std, produces plots |
| `plot_results.py` | Standalone plot tool for saved `multiseed_results.json` |

---

## The LSTM Model (`model.py`)

### Architecture

```
Input: integer token sequence, shape (B, 24)
  │
  ▼
Embedding(vocab_size=26, dim=16, padding_idx=0)   →   (B, 24, 16)
  │
  ▼
LSTM(input=16, hidden=32, num_layers=1, batch_first=True)
  │   returns (output, (h_n, c_n))
  │   h_n shape: (1, B, 32)
  ▼
h_n[-1]   →   (B, 32)   [last hidden state = sequence summary]
  │
  ▼
Linear(32, 2)   →   (B, 2)   [logits for class 0=honest, 1=FBS]
```

**Total parameters: 6,882** — small enough to train in seconds on CPU.

### Parameter Tensors (7 total)

| Index | Name | Shape | Count |
|---|---|---|---|
| 0 | `embed.weight` | (26, 16) | 416 |
| 1 | `lstm.weight_ih_l0` | (128, 16) | 2,048 |
| 2 | `lstm.weight_hh_l0` | (128, 32) | 4,096 |
| 3 | `lstm.bias_ih_l0` | (128,) | 128 |
| 4 | `lstm.bias_hh_l0` | (128,) | 128 |
| 5 | `fc.weight` | (2, 32) | 64 |
| 6 | `fc.bias` | (2,) | 2 |

### Why LSTM over bag-of-tokens logistic regression

The previous implementation vectorized each session as a bag of tokens (count vector) and trained an `SGDClassifier`. That approach discards ordering — it cannot distinguish "Auth was skipped" from "Auth happened out of order." An LSTM preserves the sequence, which is exactly what makes FBS attacks structurally detectable: the *order* of `EMM_SMC_NULL`, missing `EMM_AUTH_REQ`, and `SHT0_PLAIN` tokens is the signal.

### Flower parameter helpers

```python
get_parameters(model)   # → list of 7 ndarrays (one per state_dict entry)
set_parameters(model, params)  # loads list of ndarrays back into the model
```

Both use `model.state_dict()` key order, which is deterministic and consistent across calls. The server and every client use these same helpers, so weight round-trips are lossless.

---

## Data Pipeline (`generate_data.py`)

1. **Session generation**: 2,500 sessions total — 60% honest (`label=0`), 40% FBS (`label=1`). Each session is a space-separated string of L3 message tokens reflecting a realistic (honest) or attack (FBS) attach procedure.

2. **Label noise**: 5% of labels are randomly flipped to simulate real-world annotation noise.

3. **Train/val split**: top 20% → `val_data.csv` (500 sessions, held out on the server). Remaining 2,000 → training.

4. **Dirichlet partition**: training sessions are split across 5 clients (4 honest + 1 poison) using a Dirichlet distribution with concentration parameter `alpha`. Low `alpha` (e.g., 0.3) produces highly skewed, non-IID partitions; high `alpha` (≥100) approaches IID. Default is `alpha=0.5`.

5. **Vocab**: 26 tokens — `PAD` at index 0, then 25 L3/RF tokens. Index 0 doubles as the padding token for the embedding layer (`padding_idx=0`).

### Token Vocabulary

```
PAD                             (index 0 — padding and unknown)
RRC_CONN_REQ / SETUP / SETUP_CMP
RRC_UL_INFO_NAS / RRC_DL_INFO_NAS
EMM_IDENTITY_REQ / EMM_IDENTITY_RES
EMM_AUTH_REQ / EMM_AUTH_RES
EMM_SMC / EMM_SMC_NULL / EMM_SMC_CMP
RRC_SEC_MODE_CMD / RRC_SEC_MODE_CMP
EMM_ESM_INFO_REQ / EMM_ESM_INFO_RES
EMM_ATTACH_REQ / EMM_ATTACH_ACCEPT / EMM_ATTACH_CMP / EMM_ATTACH_REJECT
SHT0_PLAIN / SHT2_CIPHERED / SHT4_NEWCTX
RF_REFPOWER_HIGH / RF_TAC_CHANGE
```

---

## Federated Learning Round (per round)

```
Server broadcasts current global weights
        │
        ▼ (to all 5 clients simultaneously)

client_normal.py (×4):                 client_poison.py (×1):
  set_parameters(model, weights)          set_parameters(model, weights)
  train_local(model, X, y, epochs=1)      train_local(model, X, y, epochs=1)
  return get_parameters(model)            honest = get_parameters(model)
                                          poisoned = corrupt(honest)
                                          return poisoned
        │                                        │
        └──────────────────┬────────────────────┘
                           ▼
              server aggregate_fit()
              ┌─────────────────────────────────────────┐
              │ for t in range(7):  # per tensor         │
              │   stacked = stack([client[t] for all])   │
              │   if defense == "none":                  │
              │       agg[t] = mean(stacked, axis=0)     │
              │   else:  # trimmed_mean                  │
              │       sort along client axis             │
              │       drop k lowest + k highest values   │
              │       agg[t] = mean(kept, axis=0)        │
              └─────────────────────────────────────────┘
                           │
                           ▼
              server evaluate_fn()
                build_model(vocab_size=26)
                set_parameters(model, aggregated)
                accuracy(model, X_val, y_val)
                print "[Round N] VALIDATION accuracy = X%"
```

---

## Aggregation Strategies

### FedAvg (`--defense none`)
Coordinate-wise mean across all client updates. Vulnerable: one malicious client with a high-magnitude poisoned update (`scale=10`) can shift the aggregate significantly.

### Trimmed Mean (`--defense trimmed_mean`)
For each coordinate (applied independently per tensor, per element), sort the `n` client values, drop the `k` lowest and `k` highest, and average the remainder. With `--trim 1` and 5 clients, this drops the single most extreme value at each coordinate — enough to neutralize one sign-flip attacker.

Both strategies are applied **per tensor, independently**, preserving each tensor's shape through the aggregation without any flattening.

---

## Poisoning Attacks (`client_poison.py`)

### Sign-flip (`--attack sign_flip`, default)
```python
poisoned[t] = -scale * honest[t]   # for each tensor t
```
Reverses the direction of every gradient and amplifies it by `scale` (default 10). Under plain FedAvg, this pulls the global model in the wrong direction. Trimmed mean removes this outlier coordinate-by-coordinate.

### Noise (`--attack noise`)
```python
poisoned[t] = honest[t] + Normal(0, scale)   # for each tensor t
```
Adds large Gaussian noise to an otherwise honest update. Less targeted than sign-flip but still degrades convergence under FedAvg.

---

## Running the Experiment

### Single run
```bash
python launcher.py                         # uses existing CSVs
python launcher.py --regen                 # regenerate data first
python launcher.py --scale 15 --rounds 10  # stronger attack, more rounds
```

### Multi-seed (for defensible aggregate results)
```bash
python multiseed.py                        # 10 seeds, alpha=0.5, 15 rounds
python multiseed.py --seeds 20 --alpha 0.3 # 20 seeds, more non-IID
python multiseed.py --seeds 10 --rounds 20 --scale 10
```

Outputs:
- `multiseed_results.json` — per-seed and aggregate statistics
- `multiseed_convergence.png` — accuracy band plot (mean ± std per round)
- `multiseed_barplot.png` — steady-state bar chart (no-defense vs trimmed mean)

### Central sanity check (no federation)
```bash
python model.py   # trains FBSLSTM on sessions_full.csv, prints val accuracy
```

---

## Migration Summary: Logistic Regression → LSTM

The original implementation used `sklearn.SGDClassifier` with bag-of-tokens (count vector) features. The migration replaced this with the sequence-aware FBSLSTM throughout.

### What changed in each file

**`client_normal.py`**
- Removed: `SGDClassifier`, `StandardScaler`, `vectorize()` (bag-of-tokens)
- Added: `build_model`, `train_local`, `encode_frame`, `load_vocab`, `accuracy` from `model.py`
- `get_parameters` / `set_parameters` now handle a list of 7 ndarrays instead of one flat vector
- `fit()` calls `train_local()` (SGD, 1 epoch) instead of `partial_fit()`

**`client_poison.py`**
- Same model swap as `client_normal.py`
- Sign-flip and noise attacks now iterate over the list of 7 parameter tensors instead of operating on a single flat vector

**`server.py`**
- Removed: `LogisticRegression`, `StandardScaler`, `_vectorize()`, `build_eval_model()`; `sklearn` dependency dropped entirely
- Added: `build_model`, `set_parameters`, `encode_frame`, `accuracy` from `model.py`
- `evaluate_fn()`: builds a fresh FBSLSTM, loads aggregated weights, calls `accuracy()` on `val_data.csv`
- `aggregate_fit()`: loops over 7 tensors; applies mean or trimmed mean per tensor independently (previously operated on one `(n_clients, D)` flat matrix)
- Initial parameters: seeded from `build_model(vocab_size=26, seed=42)` instead of `np.zeros(N_FEATURES+1)`

**`multiseed.py`, `launcher.py`** — no changes. Both parse the `[Round X] VALIDATION accuracy = Y%` log line, which is preserved in the new server.

---

## Known Limitations

- **UNK = PAD**: unknown tokens in `tokenize()` fall back to `PAD_ID=0`, the same index as padding. In practice this never triggers (the vocab covers all generated tokens), but would be a bug if real capture data introduced unseen message types.
- **No dropout**: the LSTM→Linear path has no regularization. On small per-client partitions (some clients have ~300 sessions after Dirichlet split), per-client overfitting is possible. A `Dropout(0.2)` between the LSTM output and `fc` would help if this becomes an issue.
- **CPU only**: `set_parameters` uses `torch.tensor()` which always produces a CPU tensor. The model and training are CPU-bound; no GPU path exists.
- **Synthetic data**: all sessions are generated, not captured. The vocabulary and sequence patterns are grounded in real LTE message types from a SCAT/Wireshark capture but the statistical distribution is simulated.
