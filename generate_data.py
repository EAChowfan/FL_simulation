
"""
generate_data.py  (NAS+RRC L3 sequence edition)
-----------------------------------------------
Generates synthetic per-session NAS+RRC message SEQUENCES for the federated
FBS-detection PoC. Each "session" is one UE attach procedure: an ordered list
of L3 control-plane messages (EMM / RRC), exactly the air-interface signal an
FBS manipulates and the sequence an LSTM detector consumes.
 
Why sequences (not tabular): a False Base Station attack is defined by *what
happens in what order* during attach -- e.g. the network skipping the
Authentication Request, sending a Security Mode Command with null algorithms,
or never establishing a security context. A per-row feature table throws that
ordering away; a sequence keeps it, which is the whole point of the paper's
LSTM track.
 
Vocabulary is grounded in the uploaded SCAT/Wireshark diag capture. Tokens map
to real L3 message types observed there:
 
  EMM (nas-eps_nas_msg_emm_type):
     7  Identity Request          8  Identity Response
    41  Authentication Request   42  Authentication Response
    52  Security Mode Command    53  Security Mode Complete
    68  Attach Request           38  Attach Accept   39 Attach Complete
    44  ESM Information Request    2  Detach / reject paths
  RRC (lte-rrc):
    RRC_CONN_REQ / SETUP / SETUP_COMPLETE
    RRC_DL_INFO_NAS / RRC_UL_INFO_NAS   (NAS carried over RRC)
    RRC_SEC_MODE_CMD / RRC_SEC_MODE_COMPLETE
  Security header (nas-eps_security_header_type):
    SHT0 plain - SHT1 integrity - SHT2 integrity+ciphered - SHT4 new-context
 
A genuine (honest eNodeB) attach contains a proper authentication +
security-mode exchange and ciphered messages. An FBS attach exhibits the
combined attack signature:
  * skipped / absent Authentication Request
  * Security Mode Command with null algorithms (EEA0/EIA0)  -> SMC_NULL
  * or skipped Security Mode Command entirely
  * messages remaining in plaintext (SHT0) past the point they should be ciphered
  * abnormal radio context (high ref power, frequent TAC change) encoded as tokens
 
Output: per-session token sequences + labels, partitioned (Dirichlet) across
N honest clients + 1 poison client, plus a held-out validation split.
 
Files written (next to this script):
  sessions_full.csv     all sessions: session_id,label,seq (space-joined tokens)
  vocab.json            token -> integer id mapping (shared by all clients)
  client1..N_data.csv   honest client partitions
  poison_data.csv       poison client's (pre-sabotage) partition
  val_data.csv          held-out validation sessions
"""
 
import argparse
import json
 
import numpy as np
import pandas as pd
 
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SEED = 42
N_SESSIONS = 2500
FBS_FRACTION = 0.40
LABEL_NOISE = 0.02          # fraction of sessions with flipped labels
N_HONEST_CLIENTS = 4
NON_IID_ALPHA = 1.0         # moderate non-IID; low values can starve a client
MAX_LEN = 30                # sessions padded/truncated to this length downstream
 
rng = np.random.default_rng(SEED)
 
# ----------------------------------------------------------------------
# L3 message vocabulary (grounded in the uploaded capture's type codes)
# ----------------------------------------------------------------------
TOKENS = [
    "PAD",
    "RRC_CONN_REQ", "RRC_CONN_SETUP", "RRC_CONN_SETUP_CMP",
    "RRC_UL_INFO_NAS", "RRC_DL_INFO_NAS",
    "EMM_IDENTITY_REQ", "EMM_IDENTITY_RES",
    "EMM_AUTH_REQ", "EMM_AUTH_RES",
    "EMM_SMC", "EMM_SMC_NULL", "EMM_SMC_CMP",
    "RRC_SEC_MODE_CMD", "RRC_SEC_MODE_CMP",
    "EMM_ESM_INFO_REQ", "EMM_ESM_INFO_RES",
    "EMM_ATTACH_REQ", "EMM_ATTACH_ACCEPT", "EMM_ATTACH_CMP",
    "EMM_ATTACH_REJECT",
    "SHT0_PLAIN", "SHT2_CIPHERED", "SHT4_NEWCTX",
    "RF_REFPOWER_HIGH", "RF_TAC_CHANGE",
]
VOCAB = {tok: i for i, tok in enumerate(TOKENS)}
 
 
def _maybe(p):
    return rng.random() < p
 
 
def honest_session():
    """Genuine eNodeB attach: proper auth + security mode + ciphering."""
    s = ["RRC_CONN_REQ", "RRC_CONN_SETUP", "RRC_CONN_SETUP_CMP",
         "RRC_UL_INFO_NAS", "EMM_ATTACH_REQ",
         "SHT0_PLAIN",
         "RRC_DL_INFO_NAS", "EMM_IDENTITY_REQ", "EMM_IDENTITY_RES",
         "EMM_AUTH_REQ", "EMM_AUTH_RES",
         "RRC_SEC_MODE_CMD", "EMM_SMC", "EMM_SMC_CMP", "RRC_SEC_MODE_CMP",
         "SHT4_NEWCTX",
         "EMM_ESM_INFO_REQ", "EMM_ESM_INFO_RES",
         "SHT2_CIPHERED", "EMM_ATTACH_ACCEPT", "EMM_ATTACH_CMP"]
    if _maybe(0.15):
        s.insert(int(rng.integers(3, len(s))), "RF_TAC_CHANGE")
    if _maybe(0.10):
        s.remove("EMM_IDENTITY_REQ"); s.remove("EMM_IDENTITY_RES")
    return s
 
 
def fbs_session():
    """FBS attach: combined attack signature."""
    s = ["RRC_CONN_REQ", "RRC_CONN_SETUP", "RRC_CONN_SETUP_CMP",
         "RRC_UL_INFO_NAS", "EMM_ATTACH_REQ", "SHT0_PLAIN",
         "RRC_DL_INFO_NAS"]
    if _maybe(0.85):
        s.append("RF_REFPOWER_HIGH")
    if _maybe(0.70):
        s.append("RF_TAC_CHANGE")
    s += ["EMM_IDENTITY_REQ", "EMM_IDENTITY_RES"]
    if _maybe(0.45):
        s += ["EMM_IDENTITY_REQ", "EMM_IDENTITY_RES"]
    # combined signature: usually SKIP authentication
    if not _maybe(0.75):
        s += ["EMM_AUTH_REQ", "EMM_AUTH_RES"]
    # security mode: null algorithms (BR-8) or skipped entirely (BR-31).
    # Removed the honest-looking EMM_SMC variant — every FBS session must
    # violate at least one 3GPP normative rule the LSTM can learn to detect.
    if _maybe(0.65):
        s += ["RRC_SEC_MODE_CMD", "EMM_SMC_NULL", "EMM_SMC_CMP"]
    # else: no security mode at all (also a clear violation)
    # plaintext past the security point — always SHT0_PLAIN for FBS (BR-25)
    s.append("SHT0_PLAIN")
    s += ["EMM_ATTACH_ACCEPT"]
    if _maybe(0.5):
        s.append("EMM_ATTACH_CMP")
    if _maybe(0.2):
        s.append("EMM_ATTACH_REJECT")
    return s
 
 
def make_sessions(n, label):
    rows = []
    for _ in range(n):
        seq = honest_session() if label == 0 else fbs_session()
        if _maybe(0.15) and len(seq) > 6:
            del seq[int(rng.integers(0, len(seq)))]
        seq = seq[:MAX_LEN]
        rows.append({"label": label, "seq": " ".join(seq)})
    return pd.DataFrame(rows)
 
 
def dirichlet_partition(train, n_parts, alpha, seed):
    prng = np.random.default_rng(seed)
    labels = train["label"].to_numpy()
    idx_by_client = [[] for _ in range(n_parts)]
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        prng.shuffle(idx_c)
        props = prng.dirichlet([alpha] * n_parts)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx_c, cuts)):
            idx_by_client[i].extend(chunk.tolist())
    for i in range(n_parts):
        if not idx_by_client[i]:
            big = max(range(n_parts), key=lambda j: len(idx_by_client[j]))
            idx_by_client[i] = idx_by_client[big][:5]
            idx_by_client[big] = idx_by_client[big][5:]
    return [train.iloc[ci].reset_index(drop=True) for ci in idx_by_client]
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=NON_IID_ALPHA,
                    help="Dirichlet non-IID concentration (low=skewed, >=100=IID)")
    ap.add_argument("--seed", type=int, default=SEED)
    cli = ap.parse_args()
    alpha, seed = cli.alpha, cli.seed
 
    global rng
    rng = np.random.default_rng(seed)
 
    n_fbs = int(N_SESSIONS * FBS_FRACTION)
    n_norm = N_SESSIONS - n_fbs
    df = pd.concat([make_sessions(n_norm, 0), make_sessions(n_fbs, 1)],
                   ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "session_id", np.arange(len(df)))
 
    n_flip = int(len(df) * LABEL_NOISE)
    flip = rng.choice(len(df), n_flip, replace=False)
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]
 
    n_val = int(len(df) * 0.20)
    val = df.iloc[:n_val].reset_index(drop=True)
    train = df.iloc[n_val:].reset_index(drop=True)
 
    parts = dirichlet_partition(train, N_HONEST_CLIENTS + 1, alpha, seed)
 
    df.to_csv("sessions_full.csv", index=False)
    val.to_csv("val_data.csv", index=False)
    with open("vocab.json", "w") as f:
        json.dump(VOCAB, f, indent=2)
 
    names = [f"client{i+1}_data.csv" for i in range(N_HONEST_CLIENTS)] + ["poison_data.csv"]
    print(f"Wrote NAS+RRC session datasets (non-IID alpha={alpha}, seed={seed}):")
    print(f"  sessions_full.csv {len(df):>5} sessions ({df.label.mean()*100:.1f}% FBS)")
    print(f"  vocab.json        {len(VOCAB)} L3 message tokens")
    for name, part in zip(names, parts):
        n1 = int((part.label == 1).sum())
        pct = 100.0 * n1 / max(len(part), 1)
        part.to_csv(name, index=False)
        print(f"  {name:<16} {len(part):>4} sessions  [FBS={n1:>3} -> {pct:4.0f}%]")
    print(f"  val_data.csv      {len(val):>4} sessions  (held-out)")
    print(f"\n  example honest session:\n    {df[df.label==0].iloc[0]['seq']}")
    print(f"\n  example FBS session:\n    {df[df.label==1].iloc[0]['seq']}")
 
 
if __name__ == "__main__":
    main()
 