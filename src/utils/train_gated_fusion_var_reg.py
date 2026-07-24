"""
train_gated_fusion_var_reg.py
===============================
Gate-only fine-tuning from the staged checkpoint (best_model_v2_staged.pt),
using a variance penalty on alpha instead of the auxiliary cosine loss in
train_gated_fusion_aux_cosine.py (Session 2). Same setup otherwise, so the
two checkpoints are directly comparable.

Loss formulation
-----------------
  gate_var_loss = -var(alpha)          # negative variance: minimizing loss
                                        # maximizes spread of alpha across
                                        # the batch
  total_loss    = bce_loss + 0.01 * gate_var_loss

All v1 parameters (genomic_projector, cross_attention, post_attn, head) are
frozen. Only GatedFusion's Wg/bg are trained, for 15 epochs.

Validation cadence
-------------------
The full per-patient validation pass (needed for cohort-wide alpha
mean/std/min/max and the Welch t-test) is expensive — 191 individual
forward passes. Running it every epoch made a 15-epoch run take ~2 hours.
Instead:
  - Every epoch: cheap batched ROC-AUC (used to pick the best checkpoint),
    plus a 2-patient lookup for TCGA-BH-A0HK / TCGA-BH-A0HW (nearly free).
  - Every EVAL_EVERY=3 epochs, and on the final epoch: full per-patient
    validation for alpha_mean/std/min/max and the Welch t-test.
Epochs without a full eval log those fields as null in the history.

Per-epoch logging
------------------
  alpha for TCGA-BH-A0HK, TCGA-BH-A0HW (every epoch)
  alpha_mean, alpha_std, alpha_min, alpha_max (every EVAL_EVERY epochs)
  Welch t-test on alpha, PR+ vs PR- (every EVAL_EVERY epochs)
  bce_loss, gate_var_loss, total_loss (every epoch)
  validation_auc (every epoch, cheap batched)

Outputs
-------
  checkpoints/v2_staged_var.pt          — best checkpoint by val ROC-AUC
  reports/gated_fusion_staged_var_results.json — full per-epoch history

Usage
-----
    python src/utils/train_gated_fusion_var_reg.py
"""

import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from monai.metrics import ROCAUCMetric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset        import load_and_qc_patients, build_dataloader
from src.models.fusion_model import PathoGenomicFusionModel

CONFIG_PATH   = ROOT / "configs" / "model_config.yaml"
COUNTS_PATH   = ROOT / "data"   / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data"   / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data"   / "raw" / "clinical_metadata.csv"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_PATH  = ROOT / "reports" / "gated_fusion_staged_var_results.json"

SOURCE_CKPT = CKPT_DIR / "best_model_v2_staged.pt"
OUT_CKPT    = CKPT_DIR / "v2_staged_var.pt"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]

EPOCHS       = 5
EVAL_EVERY   = 1     # full per-patient alpha eval cadence (every epoch — cheap at EPOCHS=5)
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAR_WEIGHT   = 0.01


def gate_var_loss_fn(alpha: torch.Tensor) -> torch.Tensor:
    return -torch.var(alpha)


# ── Training epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch) -> tuple[float, float, float]:
    model.train()
    bce_sum = var_sum = total_sum = 0.0

    for batch in loader:
        patch_emb  = batch["patch_embeddings"].to(device)
        genomic    = batch["genomic_counts"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels     = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, _, alpha = model(patch_emb, genomic, patch_mask)

        bce  = criterion(logits, labels)
        var  = gate_var_loss_fn(alpha)
        loss = bce + VAR_WEIGHT * var

        loss.backward()
        optimizer.step()

        bce_sum   += bce.item()
        var_sum   += var.item()
        total_sum += loss.item()

    n = len(loader)
    mean_bce, mean_var, mean_total = bce_sum / n, var_sum / n, total_sum / n
    print(f"  [Epoch {epoch:02d}] bce={mean_bce:.4f}  gate_var={mean_var:.4f}  total={mean_total:.4f}")
    return mean_bce, mean_var, mean_total


# ── Cheap batched validation AUC (used every epoch for checkpoint selection) ──

def val_epoch_auc(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    auc_metric = ROCAUCMetric()
    auc_metric.reset()

    with torch.no_grad():
        for batch in loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels     = batch["label"].float().unsqueeze(1).to(device)

            logits, _, _ = model(patch_emb, genomic, patch_mask)
            probs = torch.sigmoid(logits)
            auc_metric(y_pred=probs, y=labels)

    auc = auc_metric.aggregate()
    return auc.item() if isinstance(auc, torch.Tensor) else float(auc)


# ── Cheap borderline-only lookup (2 patients, every epoch) ────────────────────

def borderline_only(model: nn.Module, val_data: list[dict]) -> dict:
    model.eval()
    by_id = {p["patient_id"]: p for p in val_data if p.get("patient_id") in BORDERLINE_IDS}
    out = {}
    with torch.no_grad():
        for pid, patient in by_id.items():
            pe = torch.tensor(patient["patch_embeddings"], dtype=torch.float32).unsqueeze(0)
            gc = torch.tensor(patient["genomic_counts"],   dtype=torch.float32).unsqueeze(0)
            pm = torch.ones(1, pe.shape[1], dtype=torch.bool)
            logit, _, alpha = model(pe, gc, pm)
            out[pid] = {
                "alpha":       float(alpha.squeeze().item()),
                "prob_pr_pos": torch.sigmoid(logit).item(),
                "label":       int(patient["label"]),
            }
    return out


# ── Full held-out evaluation (per-patient, needed for alpha stats/t-test) ─────

def validate_full(model: nn.Module, val_data: list[dict]) -> dict:
    model.eval()
    probs_out, labels_out, alphas_out, pids_out = [], [], [], []

    with torch.no_grad():
        for patient in val_data:
            pe = torch.tensor(patient["patch_embeddings"], dtype=torch.float32).unsqueeze(0)
            gc = torch.tensor(patient["genomic_counts"],   dtype=torch.float32).unsqueeze(0)
            pm = torch.ones(1, pe.shape[1], dtype=torch.bool)

            logit, attn_weights, alpha = model(pe, gc, pm)

            probs_out.append(torch.sigmoid(logit).item())
            labels_out.append(float(patient["label"]))
            alphas_out.append(float(alpha.squeeze().item()))
            pids_out.append(patient.get("patient_id", f"patient_{len(probs_out)}"))

    probs  = np.array(probs_out)
    labels = np.array(labels_out)
    alphas = np.array(alphas_out)

    alpha_pos = alphas[labels == 1]
    alpha_neg = alphas[labels == 0]
    t_stat, p_val = stats.ttest_ind(alpha_pos, alpha_neg, equal_var=False)

    by_pid = dict(zip(pids_out, zip(alphas_out, probs_out, labels_out)))
    borderline = {
        pid: {"alpha": by_pid[pid][0], "prob_pr_pos": by_pid[pid][1], "label": int(by_pid[pid][2])}
        for pid in BORDERLINE_IDS if pid in by_pid
    }

    return {
        "roc_auc":       float(roc_auc_score(labels, probs)),
        "pr_auc":        float(average_precision_score(labels, probs)),
        "brier_score":   float(brier_score_loss(labels, probs)),
        "alpha_mean":    float(alphas.mean()),
        "alpha_std":     float(alphas.std()),
        "alpha_min":     float(alphas.min()),
        "alpha_max":     float(alphas.max()),
        "welch_t":       float(t_stat),
        "welch_p":       float(p_val),
        "borderline":    borderline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — Session 3: Variance Regularization (from staged) ══\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split,
    )
    print(f"Train: {len(train_data)}    Val (held-out): {len(val_data)}\n")

    train_loader = build_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")

    model = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim).to(device)
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded {SOURCE_CKPT.name}: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

    # Freeze all v1 parameters; only GatedFusion (Wg, bg) trains.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.gated_fusion.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (gate only)")
    print(f"Full per-patient alpha eval every {EVAL_EVERY} epochs; cheap batched AUC every epoch.\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        mean_bce, mean_var, mean_total = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_auc = val_epoch_auc(model, val_loader, device)
        border   = borderline_only(model, val_data)

        do_full = (epoch % EVAL_EVERY == 0) or (epoch == EPOCHS)
        full = validate_full(model, val_data) if do_full else None

        hk = border.get("TCGA-BH-A0HK", {}).get("alpha")
        hw = border.get("TCGA-BH-A0HW", {}).get("alpha")

        if full is not None:
            print(f"  [Epoch {epoch:02d}] val_auc={val_auc:.4f}  "
                  f"alpha: mean={full['alpha_mean']:.4f} std={full['alpha_std']:.4f} "
                  f"min={full['alpha_min']:.4f} max={full['alpha_max']:.4f}  "
                  f"welch_p={full['welch_p']:.4e}  [FULL EVAL]")
        else:
            print(f"  [Epoch {epoch:02d}] val_auc={val_auc:.4f}  (full alpha eval skipped)")
        print(f"             A0HK alpha={hk}  A0HW alpha={hw}")

        history.append({
            "epoch":          epoch,
            "bce_loss":       mean_bce,
            "gate_var_loss":  mean_var,
            "total_loss":     mean_total,
            "validation_auc": val_auc,
            "alpha_mean":     full["alpha_mean"] if full else None,
            "alpha_std":      full["alpha_std"]  if full else None,
            "alpha_min":      full["alpha_min"]  if full else None,
            "alpha_max":      full["alpha_max"]  if full else None,
            "welch_t":        full["welch_t"]    if full else None,
            "welch_p":        full["welch_p"]    if full else None,
            "borderline":     border,
            "full_eval":      do_full,
        })

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_auc": val_auc},
                OUT_CKPT,
            )
            print(f"  ✓ Checkpoint saved  (val_auc={val_auc:.4f} → {OUT_CKPT.name})")

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}")

    output = {
        "model":        "GatedFusion-v2-staged-var-reg",
        "source_ckpt":  str(SOURCE_CKPT),
        "var_weight":   VAR_WEIGHT,
        "epochs":       EPOCHS,
        "eval_every":   EVAL_EVERY,
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size":   BATCH_SIZE,
        "frozen":       "all v1 params (genomic_projector, cross_attention, post_attn, head)",
        "trained":      "gated_fusion only",
        "history":      history,
        "best_val_auc": best_auc,
        "checkpoint":   str(OUT_CKPT),
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved → {RESULTS_PATH}")
    print("\n══ Done ════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
