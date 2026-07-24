"""
calibrate_gate_platt.py
=========================
Platt-scaling (logistic regression) recalibration of the Gated-Fusion alpha
signal, applied to checkpoints/v2_anchored_v2.pt (epoch 9, the direction-
anchored checkpoint from Session 6).

Motivation
-----------
Session 6 stabilized alpha's direction (consistently PR- higher across all
10 epochs) but alpha_mean plateaued around 0.32-0.36 — well below the 0.5
flagging threshold, the same calibration problem seen in every prior
checkpoint. Unlike the earlier isotonic attempt on v2_staged_var.pt (which
overfit a narrow, low-diversity alpha range and degraded held-out
significance), Platt scaling fits a much simpler two-parameter logistic
curve, which should be far less prone to overfitting a small/narrow input
range.

Procedure
----------
  1. Load checkpoints/v2_anchored_v2.pt. Freeze all parameters (pure
     inference; the model is not modified).
  2. Batched forward pass over the training split (760 patients) to collect
     raw alpha and true PR label.
  3. Fit sklearn.linear_model.LogisticRegression on (alpha_train, label_train).
  4. Batched forward pass over the held-out cohort (191 patients); apply the
     fitted Platt model to raw alpha via predict_proba to get calibrated
     alpha (a proper probability in [0, 1]).
  5. Report calibrated alpha mean/std/min/max, fraction > 0.5, Welch t-test
     (full cohort and class-balanced), and both borderline patients' raw
     alpha, calibrated alpha, and P(PR+) (unchanged by calibration).

Outputs
-------
  checkpoints/platt_calibrator_anchored.pkl        — fitted LogisticRegression
  reports/gate_diagnostics_anchored_calibrated.csv — per-patient diagnostics

Usage
-----
    python src/utils/calibrate_gate_platt.py
"""

import sys
import csv
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset        import load_and_qc_patients, build_dataloader
from src.models.fusion_model import PathoGenomicFusionModel

CONFIG_PATH   = ROOT / "configs" / "model_config.yaml"
COUNTS_PATH   = ROOT / "data"   / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data"   / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data"   / "raw" / "clinical_metadata.csv"
CKPT_DIR      = ROOT / "checkpoints"

SOURCE_CKPT     = CKPT_DIR / "v2_anchored_v2.pt"
CALIBRATOR_PATH = CKPT_DIR / "platt_calibrator_anchored.pkl"
CSV_PATH        = ROOT / "reports" / "gate_diagnostics_anchored_calibrated.csv"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]
BATCH_SIZE     = 4
BALANCE_RANDOM_STATE = 42


# ── Batched inference: collect alpha, prob, label, patient_id ────────────────

def collect(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    probs_out, labels_out, alphas_out, pids_out = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels     = batch["label"].float().unsqueeze(1).to(device)
            pids       = batch["patient_id"]

            logits, _, alpha = model(patch_emb, genomic, patch_mask)
            probs = torch.sigmoid(logits)

            probs_out.extend(probs.squeeze(-1).tolist())
            labels_out.extend(labels.squeeze(-1).tolist())
            alphas_out.extend(alpha.squeeze(-1).tolist())
            pids_out.extend(pids)

    return {
        "probs":  np.array(probs_out),
        "labels": np.array(labels_out),
        "alphas": np.array(alphas_out),
        "pids":   pids_out,
    }


def balanced_welch(values: np.ndarray, labels: np.ndarray, random_state: int) -> tuple[float, float]:
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(random_state)

    if len(pos_idx) > len(neg_idx):
        pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=False)
    elif len(neg_idx) > len(pos_idx):
        neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

    t_stat, p_val = stats.ttest_ind(values[pos_idx], values[neg_idx], equal_var=False)
    return float(t_stat), float(p_val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gate Alpha — Platt Scaling (from v2_anchored_v2.pt, epoch 9) ══\n")

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split,
    )
    print(f"Train: {len(train_data)}    Val (held-out): {len(val_data)}\n")

    train_loader = build_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = build_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")

    model = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim).to(device)
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded {SOURCE_CKPT.name}: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

    # Freeze everything — this script is pure inference + calibration.
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # ── Fit Platt scaler on the training split ─────────────────────────────────
    train_out = collect(model, train_loader, device)
    platt = LogisticRegression()
    platt.fit(train_out["alphas"].reshape(-1, 1), train_out["labels"])
    print(f"Platt scaler fit on {len(train_out['alphas'])} training patients "
          f"(raw alpha -> PR+ label)")
    print(f"  coef={platt.coef_[0][0]:.4f}  intercept={platt.intercept_[0]:.4f}")

    # ── Evaluate on the held-out cohort ────────────────────────────────────────
    val_out = collect(model, val_loader, device)
    raw_alphas = val_out["alphas"]
    cal_alphas = platt.predict_proba(raw_alphas.reshape(-1, 1))[:, 1]
    labels     = val_out["labels"]
    probs      = val_out["probs"]
    pids       = val_out["pids"]

    t_stat, p_val   = stats.ttest_ind(cal_alphas[labels == 1], cal_alphas[labels == 0], equal_var=False)
    bal_t, bal_p    = balanced_welch(cal_alphas, labels, BALANCE_RANDOM_STATE)

    print("\n── Held-Out Cohort (N=191), calibrated alpha ────────────────────")
    print(f"  mean={cal_alphas.mean():.4f}  std={cal_alphas.std():.4f}  "
          f"min={cal_alphas.min():.4f}  max={cal_alphas.max():.4f}")
    print(f"  fraction calibrated alpha > 0.5 : {(cal_alphas > 0.5).mean():.4f}  "
          f"({(cal_alphas > 0.5).sum()}/{len(cal_alphas)})")
    print(f"  Welch t={t_stat:+.4f}  p={p_val:.4e}")
    print(f"  Balanced Welch t={bal_t:+.4f}  p={bal_p:.4e}")

    by_pid = dict(zip(pids, zip(raw_alphas, cal_alphas, probs, labels)))
    print(f"\n  Borderline patients (raw alpha -> calibrated alpha, P(PR+) unchanged):")
    for pid in BORDERLINE_IDS:
        if pid in by_pid:
            raw_a, cal_a, prob, label = by_pid[pid]
            print(f"    {pid}: alpha {raw_a:.4f} -> {cal_a:.4f}   "
                  f"P(PR+)={prob:.4f}  label={int(label)}")
        else:
            print(f"    {pid}: not found in held-out cohort")

    # ── Save calibrator ─────────────────────────────────────────────────────────
    with open(CALIBRATOR_PATH, "wb") as fh:
        pickle.dump(platt, fh)
    print(f"\n  Calibrator saved → {CALIBRATOR_PATH}")

    # ── Save per-patient diagnostics CSV ────────────────────────────────────────
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "alpha_raw", "alpha_calibrated", "prob_pr_pos", "true_label"])
        for pid, (raw_a, cal_a, prob, label) in by_pid.items():
            writer.writerow([pid, raw_a, cal_a, prob, int(label)])
    print(f"  Diagnostics CSV saved → {CSV_PATH}  ({len(by_pid)} patients)")
    print("\n══ Done ════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
