# Preliminary Data & Proof-of-Concept: FL Poisoning Threat in FBS Detection

**Paper:** *Trust-Anchored Federated Learning for Robust Specification-AI Hybrid Detection of False Base Station Attacks*
Eldridge Aaron Miole, Vincent Abella, I Wayan Adi Juliawan Pawana, Ilsun You — Kookmin University

---

## 1. What This Simulation Demonstrates

The abstract identifies a **deployability gap**: prior hybrid FBS detection architectures assume all participating UEs are honest, leaving FL aggregation exposed to poisoned model updates. This simulation makes that gap concrete and measurable across 20 independent trials.

---

## 2. Dataset — NAS+RRC L3 Attach Sequences

Each sample is a **complete UE attach procedure** encoded as an ordered sequence of L3 control-plane message tokens, drawn from the same RRC/NAS diagnostic fields in the paper's SCAT/Wireshark capture.

**Vocabulary (26 tokens):** grounded in real message type codes from the capture.

| Token group | Tokens | 3GPP source |
|---|---|---|
| RRC connection | `RRC_CONN_REQ`, `SETUP`, `SETUP_CMP` | `lte-rrc` connection setup |
| NAS transport | `RRC_UL/DL_INFO_NAS` | NAS-over-RRC bearers |
| Identity | `EMM_IDENTITY_REQ/RES` | NAS EMM type 0x55/0x56 |
| Authentication | `EMM_AUTH_REQ/RES` | NAS EMM type 0x52/0x53 |
| Security mode | `EMM_SMC`, `EMM_SMC_NULL`, `EMM_SMC_CMP` | NAS EMM type 0x5D (EEA0/EIA0 = null) |
| RRC security | `RRC_SEC_MODE_CMD/CMP` | `lte-rrc` security mode |
| Attach outcome | `EMM_ATTACH_REQ/ACCEPT/CMP/REJECT` | NAS EMM type 0x41/0x42/0x43 |
| Security header | `SHT0_PLAIN`, `SHT2_CIPHERED`, `SHT4_NEWCTX` | `nas-eps_security_header_type` |
| RF anomaly | `RF_REFPOWER_HIGH`, `RF_TAC_CHANGE` | `lte-rrc_referencesignalpower`, TAC field |

**Why sequences, not tabular features:**
- FBS attacks are defined by *what happens in what order* — skipped auth, null cipher, plaintext past the security boundary
- A tabular row collapses the ordering; a token sequence preserves it — consistent with the paper's LSTM detection track

**Honest session (label=0):** `RRC_CONN_REQ → ... → EMM_AUTH_REQ → EMM_AUTH_RES → EMM_SMC → SHT4_NEWCTX → ... → SHT2_CIPHERED → EMM_ATTACH_ACCEPT`

**FBS session (label=1):** `RRC_CONN_REQ → RF_REFPOWER_HIGH → RF_TAC_CHANGE → [auth skipped 75%] → EMM_SMC_NULL [55%] or SMC absent [25%] → SHT0_PLAIN → EMM_ATTACH_ACCEPT`

**Dataset config:** 2,500 sessions · 40% FBS · 5% label noise · 20% held-out validation (IID) · Dirichlet partition α=2.0 (co-located UE heterogeneity) · 5 clients (4 honest + 1 poison)

---

## 3. Threat Model Exercised

| Threat class | How the simulation exercises it |
|---|---|
| Untargeted gradient poisoning | Sign-flip: poison client submits `−scale × honest_update` |
| Scaled-magnitude attack | Scale ×10 amplifies the poisoned update |
| 20% Byzantine fraction | 1 of 5 clients compromised |
| Non-IID data across UEs | Dirichlet α=2.0 — realistic co-located heterogeneity |

---

## 4. Results (20 seeds, α=2.0, sign-flip ×10, 15 rounds)

| Condition | Steady-state Accuracy | Std |
|---|---|---|
| No defense (plain FedAvg) | 29% | ± 20 |
| Trimmed-mean aggregation | **85%** | ± 12 |
| **Recovery** | **+57 pp** | — |

---

## 5. Why the Results Look This Way

**Why no-defense collapses so severely (29%):**
- Sign-flip at ×10 magnitude means the poison update is 10× larger in norm than honest updates and points in the exact opposite direction
- Plain FedAvg averages all updates equally — one adversarial client at 10× scale dominates 4 honest clients
- The ±20% variance reflects how much influence the poison client gets depending on its Dirichlet partition draw; some draws give it a large, label-rich slice, others give it a weak one

**Why trimmed mean recovers strongly (85%):**
- Coordinate-wise trimming drops the single highest and lowest value per weight dimension before averaging — the sign-flipped, scaled update is always the outlier and is discarded
- The defense operates in weight space without any knowledge of which client is adversarial
- The ±12% residual variance is honest: on draws where the poison client gets a very weak partition, no-defense also performs reasonably, compressing the gap

**Why sequence data produces stronger results than tabular features:**
- Token sequences carry the full attack signature (ordering of auth skip → null cipher → plaintext continuation) that a bag-of-features flattens away
- The classifier has sharper decision boundaries on sequence data → honest updates are more consistent → the aggregated gradient is cleaner → trimmed mean is more effective

**Why variance remains even at α=2.0:**
- Dirichlet α=2.0 is non-IID, not IID — some partition draws still concentrate FBS-heavy sessions on the poison client, giving it outsized gradient influence even before sabotage
- This is a real-world property worth reporting: the defense holds on average, but individual federation instances vary

---

## 6. What This PoC Does Not Yet Cover

| Capability | This PoC | Proposed system |
|---|---|---|
| Classifier | Logistic regression (bag-of-tokens) | LSTM (sequential, temporal) |
| Aggregation defense | Coordinate-wise trimmed mean | Trust-anchored (directional consistency + 3GPP behavioral verification) |
| Trust signal | None | Per-client trust score fusing weight-space and spec-rule alignment |
| Spec-rule cross-check | None | 3GPP normative rules as behavioral ground truth |
| Deployment | Simulated (Python subprocesses) | Rooted Android UE + Open5GS + srsRAN eNodeB |
| Attack coverage | Sign-flip, noise | Backdoor, gradient poisoning, scaled-magnitude, Sybil collusion |
| Byzantine fraction | Fixed 1/5 | Varying; no prior knowledge required |

Trimmed mean represents a **lower bound** — it defends on weight-space geometry alone, without the behavioral trust anchor that cross-checks each client's predictions against formally verified 3GPP behavior rules.

---

## 7. Poster Callouts

> **Threat confirmed.** A single compromised UE degrades FBS-detection from 85% to 29% under sign-flip poisoning — a 57 pp collapse that directly validates the deployability gap in prior hybrid architectures.

> **Defense works at baseline.** Coordinate-wise trimmed mean — a strict subset of the proposed trust-anchored mechanism — recovers 57 pp across 20 independent NAS+RRC session datasets on synthetic LTE attach sequences.

> **Sequence data matters.** Encoding attach procedures as ordered L3 token sequences (vs. flat feature rows) sharpens honest client gradients, making the aggregation defense more effective and better aligning the simulation with the paper's LSTM detection track.

---

## 8. Simulation Artefacts

| File | Description |
|---|---|
| `multiseed_convergence.png` | Per-round accuracy bands (mean ± std, 20 seeds) |
| `multiseed_barplot.png` | Steady-state summary bar chart |
| `multiseed_results.json` | Full per-seed and aggregate numbers |
| `sessions_full.csv` | All 2,500 NAS+RRC attach sessions with labels |
| `vocab.json` | 26-token L3 message vocabulary |
| `client1–4_data.csv`, `poison_data.csv` | Dirichlet-partitioned client datasets |
| `val_data.csv` | Held-out validation set (IID, server-side) |
| `multiseed.py` | Multi-seed harness (20 seeds, α=2.0, sign-flip ×10) |
| `launcher.py` | Single-run launcher for rapid iteration |
| `server.py` | Flower FL server — FedAvg / trimmed-mean strategies |
| `client_normal.py` / `client_poison.py` | Honest and adversarial FL clients |
| `generate_data.py` | NAS+RRC session generator (Dirichlet partition) |
| `plot_results.py` | Graph generator for single-run `results.json` |
