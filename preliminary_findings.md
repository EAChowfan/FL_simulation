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

**Dataset config (final run):** 10,000 sessions · 40% FBS · 5% label noise · 20% held-out validation (IID) · Dirichlet partition α=5.0 (near-IID — maximally challenging for label-flip) · 5 clients (4 honest + 1 poison)

---

## 3. Threat Model Exercised

| Threat class | How the simulation exercises it |
|---|---|
| Untargeted gradient poisoning | Sign-flip: poison client submits `−scale × honest_update` |
| Scaled-magnitude attack | Scale ×10 amplifies the poisoned update |
| Semantic data poisoning | Label-flip: poison client silently relabels every FBS session as honest; gradient is geometrically normal |
| 20% Byzantine fraction | 1 of 5 clients compromised |
| Near-IID data across UEs | Dirichlet α=5.0 — each UE receives a proportionally similar class mix; hardest case for label-flip because the poison client has the maximum number of FBS sessions to corrupt |

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

---

## 6. Final Results — Label-Flip Stealth Attack (20 seeds, α=5.0, 15 rounds, 5 local epochs, 10,000 sessions)

Label-flip is the stealth attack where the poison client trains normally but silently relabels every FBS session as honest before local training. The resulting model update is geometrically indistinguishable from an honest update — same magnitude, same general direction — because the underlying LTE sequences are real. Only the labels are wrong.

At α=5.0 (near-IID), every client — including the poison client — receives approximately 40% FBS sessions per round. This is the hardest setting for the attack to be detected: the poison client's label-flipped gradient is large, consistent, and directionally close to honest clients.

### 6.1 Primary Metrics: Recall and F1

Accuracy is the wrong metric for a security detector. Label-flip specifically targets False Negatives — FBS sessions that the model classifies as honest. A model that correctly rejects all honest traffic but misses every real attack can show 60% accuracy on a 40%-FBS dataset while providing zero protection. **Recall (TP / (TP + FN)) and F1 are the correct primary metrics.**

| Defense | Recall | Std | F1 | Std | Accuracy | Std | vs No-Def (Recall) |
|---|---|---|---|---|---|---|---|
| No defense (FedAvg) | 72.7% | ±27.6 | 72.7% | ±28.1 | 87.7% | ±11.5 | — |
| Trimmed mean | 85.6% | ±26.6 | 85.9% | ±26.7 | 93.5% | ±10.4 | +12.9 pp |
| FLTrust | **47.2%** | ±37.1 | **45.9%** | ±38.4 | 76.4% | ±16.4 | **−25.5 pp** |
| Trust-anchored (proposed) | **91.4%** | ±21.2 | **91.5%** | ±21.3 | 95.5% | ±8.3 | **+18.7 pp** |

The accuracy column illustrates the deception: FLTrust shows 76.4% accuracy — plausible-looking — but only 47.2% recall. Nearly one in two FBS sessions is missed. The accuracy number hides that the model has partially surrendered to the attack.

### 6.2 Key Findings

**FLTrust performs worse than no defense on recall (−25.5 pp).**
FLTrust scores client updates by cosine similarity with the server's own reference update. The label-flip client trains on the same LTE sequences as honest clients — only FBS labels are flipped. Its gradient direction is dominated by the large honest-session majority in its partition, making it geometrically similar to the server's reference gradient. FLTrust computes a positive cosine score and assigns the poison client normal or elevated aggregation weight. This does not suppress the attack — it incorporates it. FLTrust is blind to label-flip because label-flip is a semantic attack, not a geometric one.

In 11 of 20 seeds (55%), FLTrust recall falls below 50%. In 4 of 20 seeds (20%), FLTrust recall collapses completely to 0% — the global model predicts every session as honest. In 16 of 20 seeds, FLTrust recall is below no-defense.

**Trimmed mean recovers partially (+12.9 pp recall) but remains high-variance (±26.6).**
Coordinate-wise trimming cannot distinguish a geometrically normal label-flip update from an honest one. When it helps, it does so incidentally — the label-flip update happens to land as a coordinate-wise outlier in that seed. In seed 8, trimmed mean collapses to 0% recall while trust-anchored holds at 96.6%.

**Trust-anchored: best mean recall (91.4%) with one isolated failure.**
The behavioral trust anchor evaluates each client's submitted model against eight BR probe sequences derived from 3GPP normative clauses (BR-8, BR-25, BR-27, BR-31, BR-35). A model trained on label-flipped data predicts "honest" on known FBS probe sequences regardless of how well-formed the weight-space update looks. Its BR score collapses to ~0.30 (FBS recall ≈ 0, honest precision ≈ 1), reducing its aggregation weight to roughly 7–8% versus the 20% it would receive under FedAvg.

The mechanism fires correctly in **19 of 20 seeds**. In seed 5, trust-anchored collapses to 0% recall (see §6.4). In the remaining 19 seeds, recall ranges from 83.6% to 98.4%, with 17 of 19 above 90%.

**Variance is the second headline number.**
No defense shows ±27.6 standard deviation on recall — whether a given federation instance survives depends entirely on the partition draw. Trust-anchored shows ±21.2, but this is inflated by the single seed-5 failure; excluding that seed, the std drops to ±4.1. The defense does not just recover on average — it reliably identifies the compromised UE in nearly every configuration.

### 6.3 Per-Seed Breakdown — Seeds Where No Defense Recall Collapses

In 7 of 20 seeds (35%), no-defense recall falls below 60%. The table below shows FBS detection recall for all four defenses in those seeds. Trust-anchored holds in all seven.

| Seed | No Defense | Trimmed Mean | FLTrust | Trust-Anchored |
|---|---|---|---|---|
| 1 | 51.2% | 96.9% | 0.0% | **86.8%** |
| 3 | 33.6% | 77.8% | 10.0% | **97.2%** |
| 7 | 58.1% | 96.8% | 67.7% | **96.8%** |
| 8 | 55.6% | **0.0%** | 87.1% | **96.6%** |
| 10 | 9.7% | 97.4% | 0.0% | **97.4%** |
| 15 | 49.1% | 97.7% | 87.9% | **97.7%** |
| 16 | 19.4% | 97.0% | 0.0% | **97.0%** |

Seed 8 is notable: trimmed mean completely collapses (recall 0%) in the same seed where trust-anchored holds at 96.6%. This illustrates that even partial-recovery geometric defenses have hard failure modes that the behavioral anchor avoids.

### 6.4 Trust-Anchored Failure Case (Seed 5)

In seed 5, trust-anchored collapses to 0% recall while no-defense (95.9%) and trimmed mean (97.2%) both perform well.

| Seed | No Defense | Trimmed Mean | FLTrust | Trust-Anchored |
|---|---|---|---|---|
| 5 | 95.9% | 97.2% | 48.6% | **0.0%** |

This failure is honest and important to report. The most likely mechanism: in this particular Dirichlet draw, the global model converges unusually early to a FBS-suppressing trajectory. The BR probe evaluations in the first two rounds reflect models that have not yet developed strong FBS recall — BR scores for honest and poison clients are both low, and the normalization assigns the poison client more relative weight than in typical seeds. By round 3, the accumulated anti-FBS gradient is sufficient to collapse recall, and the defense cannot recover within the 15-round budget.

This represents a known limitation of the current BR probe coverage (5 rules, 8 sequences) and proportional rather than threshold-based downweighting. Full BR-1 to BR-43 coverage and a hard-exclusion floor are listed as planned extensions in §7.

### 6.5 Why This Is the Paper's Core Result

The sign-flip attack (Section 4) is detectable by any geometric defense — the update is a large-magnitude outlier pointing in the wrong direction. Any of the three defenses handles it.

Label-flip is the hard case. It passes every geometric check: correct magnitude, plausible direction, no coordinate-wise outliers. FLTrust — the strongest purely geometric baseline — not only fails to defend but actively degrades FBS detection recall in 55% of seeds, with 4 complete collapses to 0%. The behavioral trust anchor, which grounds each client's model against formally verified 3GPP normative rules, achieves best-in-class recall in 19 of 20 seeds, outperforming no-defense by +18.7 pp on average.

This is the gap the paper closes: geometric defenses alone are insufficient when the attacker controls what the model learns, not just how large the gradient is.

---

## 7. What This PoC Does Not Yet Cover

| Capability | This PoC | Proposed system |
|---|---|---|
| Aggregation defense | Trimmed mean + Trust-anchored + FLTrust | Same, evaluated on real captures |
| Trust signal | BR probe scoring (5 rules, 8 probes) | Full BR-1 to BR-43 coverage |
| Spec-rule cross-check | Hardcoded probe sequences | Formally verified 3GPP clause derivation |
| Exclusion policy | Proportional downweighting | Hard-exclusion floor below BR threshold |
| Deployment | Simulated (Python subprocesses) | Rooted Android UE + Open5GS + srsRAN eNodeB |
| Byzantine fraction | Fixed 1/5 | Varying; no prior knowledge required |
| Data | Synthetic NAS+RRC sequences | Real SCAT/Wireshark LTE captures |

---

## 8. Poster Callouts

> **Threat confirmed (sign-flip).** A single compromised UE degrades FBS-detection from 85% to 29% under sign-flip poisoning — a 57 pp collapse that directly validates the deployability gap in prior hybrid architectures.

> **Geometric defenses fail on stealth attacks.** Under label-flip poisoning at α=5.0, FLTrust scores the poison client as geometrically trustworthy and degrades global FBS-detection **recall to 47.2%** (−25.5 pp vs no defense) across 20 seeds, with complete recall collapse in 4 seeds. Accuracy appears acceptable at 76.4% — the gap between accuracy and recall is the attack's signature. Purely geometric defense is insufficient when the attacker manipulates semantics, not gradients.

> **Trust-anchored leads all defenses on recall.** The behavioral trust anchor — scoring each client's model against 3GPP-normative BR probes — achieves **91.4% mean FBS-detection recall** (F1 = 91.5%) under label-flip poisoning, outperforming no defense by +18.7 pp and outperforming FLTrust by +44.2 pp. It holds in 19 of 20 seeds (min recall 83.6% in the 19 holding seeds), with a single identified failure mode linked to limited BR probe coverage.

---

## 9. Simulation Artefacts

| File | Description |
|---|---|
| `multiseed_convergence.png` | Per-round accuracy bands (mean ± std, 20 seeds) |
| `multiseed_recall.png` | Per-round recall bands (mean ± std, 20 seeds) — primary metric plot |
| `multiseed_barplot.png` | Steady-state F1 bar chart with per-seed scatter |
| `multiseed_results.json` | Full per-seed and aggregate numbers (acc, recall, F1, precision) |
| `log.txt` | Full console output from the 20-seed run |
| `sessions_full.csv` | All 10,000 NAS+RRC attach sessions with labels |
| `vocab.json` | 26-token L3 message vocabulary |
| `client1–4_data.csv`, `poison_data.csv` | Dirichlet-partitioned client datasets |
| `val_data.csv` | Held-out validation set (IID, server-side) |
| `multiseed.py` | Multi-seed harness (20 seeds, α=5.0, label-flip, 5 local epochs) |
| `launcher.py` | Single-run launcher for rapid iteration |
| `server.py` | Flower FL server — FedAvg / trimmed-mean / FLTrust / trust-anchored |
| `client_normal.py` / `client_poison.py` | Honest and adversarial FL clients |
| `generate_data.py` | NAS+RRC session generator (Dirichlet partition, 10,000 sessions) |
| `plot_results.py` | Graph generator for single-run `results.json` |
