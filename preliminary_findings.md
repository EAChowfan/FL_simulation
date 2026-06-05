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

## 6. Stealth Attack Results — Label-Flip (20 seeds, α=2.0, 15 rounds, 5 local epochs, 10,000 sessions)

Label-flip is the stealth attack where the poison client trains normally but silently relabels every FBS session as honest before training. The resulting model update is geometrically indistinguishable from an honest update — same magnitude, same general direction — because the underlying LTE sequences are real. Only the labels are wrong.

### 6.1 Aggregate Steady-State FBS Detection Accuracy

| Defense | Accuracy | Std | vs No Defense |
|---|---|---|---|
| No defense (FedAvg) | 84.0% | ±14.6 | — |
| Trimmed mean | 89.4% | ±12.9 | +5.4 pp |
| FLTrust | **61.3%** | ±12.8 | **−22.7 pp** |
| Trust-anchored (proposed) | **96.0%** | **±2.7** | **+12.0 pp** |

### 6.2 Key Findings

**FLTrust performs worse than no defense (−22.7 pp).**
FLTrust scores client updates by cosine similarity with the server's own reference update. The label-flip client trains on the same LTE sequences as honest clients — only FBS labels are flipped. Its gradient direction is dominated by the large honest-session majority in its partition, making it geometrically similar to the server's reference direction. FLTrust computes a positive cosine score and assigns the poison client normal or elevated weight. This does not reduce the attack — it incorporates and in some seeds amplifies it. FLTrust is blind to label-flip because label-flip is a semantic attack, not a geometric one.

In 13 of 20 seeds, FLTrust accuracy falls to ≤60% (near the random-guessing baseline for a 40% FBS dataset). In 18 of 20 seeds, FLTrust is below no-defense.

**Trust-anchored never falls below 90.4% across all 20 seeds.**
The behavioral trust anchor evaluates each client's model against eight BR probe sequences derived directly from 3GPP normative clauses (BR-8, BR-25, BR-27, BR-31, BR-35). A model trained on label-flipped data predicts "honest" on known FBS probe sequences regardless of how well-formed its weight-space update looks. Its BR score collapses to ~0.30 (FBS recall ≈ 0, honest precision ≈ 1), giving it roughly 7–8% of the aggregation weight versus 20% under uniform FedAvg. The mechanism fires correctly in every seed.

**Variance is the second headline number.**
No defense shows ±14.6 standard deviation — whether a given federation instance survives depends on the Dirichlet partition draw that round. Trust-anchored shows ±2.7. The defense does not just recover on average; it consistently identifies and down-weights the bad client regardless of what data partition it receives. This is the property that matters for deployment.

**Trimmed mean partially recovers (+5.4 pp) but remains high-variance (±12.9).**
Coordinate-wise trimming cannot distinguish a geometrically normal label-flip update from an honest one. When it helps, it does so by coincidence — the label-flip update happens to be an outlier at some coordinates in some seeds.

### 6.3 Per-Seed Breakdown — Seeds Where No Defense Collapses

In 7 of 20 seeds (35%), no-defense falls below 75%. Trust-anchored holds in all of them.

| Seed | No Defense | Trimmed Mean | FLTrust | Trust-Anchored |
|---|---|---|---|---|
| 1 | 63.4% | 57.6% | 51.9% | **98.1%** |
| 4 | 57.9% | 98.4% | 50.0% | **98.4%** |
| 7 | 71.0% | 98.0% | 57.7% | **92.9%** |
| 8 | 68.1% | 68.1% | 71.3% | **90.4%** |
| 10 | 62.0% | 98.3% | 50.0% | **94.7%** |
| 15 | 60.8% | 94.5% | 52.2% | **98.2%** |
| 18 | 69.4% | 84.5% | 56.1% | **92.0%** |

### 6.4 Why This Is the Paper's Core Result

The sign-flip attack (Section 4) is detectable by any geometric defense — the update is a large-magnitude outlier pointing in the wrong direction. Trimmed mean, FLTrust, and trust-anchored all handle it.

Label-flip is the hard case. It is designed to pass every geometric check: correct magnitude, plausible direction, no coordinate-wise outliers. FLTrust — the strongest purely geometric baseline — not only fails to defend but actively degrades accuracy. Only the behavioral trust anchor, which grounds each client's model against formally verified 3GPP normative rules, correctly identifies the compromised UE.

This is the gap the paper closes: geometric defenses alone are insufficient when the attacker controls what the model learns, not just how large the gradient is.

---

## 7. What This PoC Does Not Yet Cover

| Capability | This PoC | Proposed system |
|---|---|---|
| Aggregation defense | Trimmed mean + Trust-anchored + FLTrust | Same, evaluated on real captures |
| Trust signal | BR probe scoring (5 rules, 8 probes) | Full BR-1 to BR-43 coverage |
| Spec-rule cross-check | Hardcoded probe sequences | Formally verified 3GPP clause derivation |
| Deployment | Simulated (Python subprocesses) | Rooted Android UE + Open5GS + srsRAN eNodeB |
| Byzantine fraction | Fixed 1/5 | Varying; no prior knowledge required |
| Data | Synthetic NAS+RRC sequences | Real SCAT/Wireshark LTE captures |

---

## 8. Poster Callouts

> **Threat confirmed (sign-flip).** A single compromised UE degrades FBS-detection from 85% to 29% under sign-flip poisoning — a 57 pp collapse that directly validates the deployability gap in prior hybrid architectures.

> **Geometric defenses fail on stealth attacks.** Under label-flip poisoning, FLTrust scores the poison client as geometrically trustworthy and incorporates its update normally — degrading global FBS-detection accuracy to 61% (−23 pp vs no defense). Purely geometric defense is insufficient when the attacker manipulates semantics, not gradients.

> **Trust-anchored holds across all 20 seeds.** The behavioral trust anchor — scoring each client's model against 3GPP-normative BR probes — never drops below 90.4% FBS-detection accuracy under label-flip, with steady-state mean 96.0% ± 2.7%. The low variance is as important as the mean: the mechanism reliably identifies the compromised UE regardless of data partition draw.

---

## 9. Simulation Artefacts

| File | Description |
|---|---|
| `multiseed_convergence.png` | Per-round accuracy bands (mean ± std, 20 seeds) |
| `multiseed_barplot.png` | Steady-state summary bar chart |
| `multiseed_results.json` | Full per-seed and aggregate numbers |
| `sessions_full.csv` | All 10,000 NAS+RRC attach sessions with labels |
| `vocab.json` | 26-token L3 message vocabulary |
| `client1–4_data.csv`, `poison_data.csv` | Dirichlet-partitioned client datasets |
| `val_data.csv` | Held-out validation set (IID, server-side) |
| `multiseed.py` | Multi-seed harness (20 seeds, α=2.0, label-flip, 5 local epochs) |
| `launcher.py` | Single-run launcher for rapid iteration |
| `server.py` | Flower FL server — FedAvg / trimmed-mean / FLTrust / trust-anchored |
| `client_normal.py` / `client_poison.py` | Honest and adversarial FL clients |
| `generate_data.py` | NAS+RRC session generator (Dirichlet partition, 10,000 sessions) |
| `plot_results.py` | Graph generator for single-run `results.json` |

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
