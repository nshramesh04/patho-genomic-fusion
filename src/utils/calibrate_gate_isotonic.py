"""
calibrate_gate_isotonic.py
============================
Isotonic-regression recalibration of the Gated-Fusion alpha signal.

Motivation
-----------
Sessions 2-3 (aux-cosine, variance-reg) both showed the same pattern: alpha
carries a real, statistically significant per-patient signal correlated
with PR status (Welch p << 0.05 in every trained variant seen so far), but
the raw scale sits on the wrong side of the fixed 0.5 flagging threshold —
recalibration, not more training, is the natural fix. This script freezes
the model entirely (no further training) and fits a monotonic isotonic
mapping from raw alpha -> calibrated alpha using the true PR label as the
supervision target, learned on the training split and evaluated on the
191-patient held-out cohort.

Procedure
----------
  1. Load checkpoints/v2_staged_var.pt (epoch-2 checkpoint from Session 3).
  2. Freeze all parameters (pure inference; the model is not modified).
  3. Batched forward pass over the training split (760 patients) to collect
     raw alpha and true label.
  4. Fit sklearn.isotonic.IsotonicRegression(alpha_train -> label_train),
     out_of_bounds="clip" so validation alphas outside the training range
     are still mapped sensibly.
  5. Batched forward pass over the held-out cohort (191 patients); apply
     the fitted calibrator to raw alpha to get calibrated alpha.
  6. Report raw vs. calibrated alpha mean/std, Welch t-test (calibrated
     alpha, PR+ vs PR-), and both raw/calibrated values for the two
     borderline patients.

Outputs
-------
  checkpoints/isotonic_calibrator.pkl — fitted IsotonicRegression (pickle)

Usage
-----
    python src/utils/calibrate_gate_isotonic.py
"""

import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from sklearn.isotonic import IsotonicRegression
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

SOURCE_CKPT       = CKPT_DIR / "v2_staged_var.pt"
CALIBRATOR_PATH   = CKPT_DIR / "isotonic_calibrator.pkl"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]
BATCH_SIZE     = 4


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


def welch_and_borderline(alphas: np.ndarray, labels: np.ndarray, probs: np.ndarray, pids: list) -> dict:
    alpha_pos = alphas[labels == 1]
    alpha_neg = alphas[labels == 0]
    t_stat, p_val = stats.ttest_ind(alpha_pos, alpha_neg, equal_var=False)

    by_pid = dict(zip(pids, zip(alphas, probs, labels)))
    borderline = {
        pid: {"alpha": float(by_pid[pid][0]), "prob_pr_pos": float(by_pid[pid][1]), "label": int(by_pid[pid][2])}
        for pid in BORDERLINE_IDS if pid in by_pid
    }
    return {
        "alpha_mean": float(alphas.mean()),
        "alpha_std":  float(alphas.std()),
        "welch_t":    float(t_stat),
        "welch_p":    float(p_val),
        "borderline": borderline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gate Alpha — Isotonic Recalibration (from v2_staged_var.pt, epoch 2) ══\n")

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

    # ── Fit isotonic calibrator on the training split ─────────────────────────
    train_out = collect(model, train_loader, device)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(train_out["alphas"], train_out["labels"])
    print(f"Isotonic calibrator fit on {len(train_out['alphas'])} training patients "
          f"(raw alpha -> PR+ label)")

    # ── Evaluate on the held-out cohort ────────────────────────────────────────
    val_out = collect(model, val_loader, device)
    raw_alphas = val_out["alphas"]
    cal_alphas = calibrator.predict(raw_alphas)

    raw_stats = welch_and_borderline(raw_alphas, val_out["labels"], val_out["probs"], val_out["pids"])
    cal_stats = welch_and_borderline(cal_alphas, val_out["labels"], val_out["probs"], val_out["pids"])

    print("\n── Held-Out Cohort (N=191) ──────────────────────────────────────")
    print(f"  {'':<12} {'mean':>8} {'std':>8} {'welch_t':>10} {'welch_p':>12}")
    print(f"  {'raw':<12} {raw_stats['alpha_mean']:>8.4f} {raw_stats['alpha_std']:>8.4f} "
          f"{raw_stats['welch_t']:>10.4f} {raw_stats['welch_p']:>12.4e}")
    print(f"  {'calibrated':<12} {cal_stats['alpha_mean']:>8.4f} {cal_stats['alpha_std']:>8.4f} "
          f"{cal_stats['welch_t']:>10.4f} {cal_stats['welch_p']:>12.4e}")

    print(f"\n  Borderline patients (raw -> calibrated alpha, P(PR+) unchanged by calibration):")
    for pid in BORDERLINE_IDS:
        r = raw_stats["borderline"].get(pid)
        c = cal_stats["borderline"].get(pid)
        if r and c:
            print(f"    {pid}: alpha {r['alpha']:.4f} -> {c['alpha']:.4f}   "
                  f"P(PR+)={r['prob_pr_pos']:.4f}  label={r['label']}")
        else:
            print(f"    {pid}: not found in held-out cohort")

    with open(CALIBRATOR_PATH, "wb") as fh:
        pickle.dump(calibrator, fh)
    print(f"\n  Calibrator saved → {CALIBRATOR_PATH}")
    print("\n══ Done ════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
