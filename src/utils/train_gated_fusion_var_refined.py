"""
train_gated_fusion_var_refined.py
===================================
Refined variance-regularization run, continuing from the epoch-2 checkpoint
of train_gated_fusion_var_reg.py (best_model by val AUC = 0.8483) rather
than from best_model_v2_staged.pt directly.

Session 3 (var_weight=0.01) showed alpha_std climbing fast (0.020 -> 0.191
over 5 epochs) but with alpha_mean drifting down aggressively (0.551 ->
0.398) and val_auc declining after epoch 2 (0.8483 -> 0.8464 by epoch 5).
This run lowers the variance penalty 0.01 -> 0.003 to see whether a gentler
push still grows alpha_std without the same drift/AUC cost, over a longer
20-epoch horizon, with early stopping if either degrades past a threshold.

Loss formulation
-----------------
  gate_var_loss = -var(alpha)
  total_loss    = bce_loss + 0.003 * gate_var_loss

All v1 parameters (genomic_projector, cross_attention, post_attn, head) are
frozen. Only GatedFusion's Wg/bg are trained, for up to 20 epochs.

Early stopping
---------------
  Stop if alpha_mean < 0.42 OR val_auc < 0.835 (checked after each epoch).

Validation cost
----------------
Uses a single cheap batched pass (same val_loader used for AUC) to also
collect per-patient alpha + label, from which alpha_mean/std/min/max, the
Welch t-test, and the borderline-patient lookup are all computed — no
separate expensive per-patient loop is needed here.

Per-epoch logging
------------------
  val_auc, alpha_mean, alpha_std, alpha_min, alpha_max
  Welch t-test p-value on alpha by PR status
  alpha and P(PR+) for TCGA-BH-A0HK, TCGA-BH-A0HW

Outputs
-------
  checkpoints/v2_var_refined.pt   — best checkpoint by val ROC-AUC
  reports/gate_var_refined_log.json — full per-epoch log

Usage
-----
    python src/utils/train_gated_fusion_var_refined.py
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset        import load_and_qc_patients, build_dataloader
from src.models.fusion_model import PathoGenomicFusionModel

CONFIG_PATH   = ROOT / "configs" / "model_config.yaml"
COUNTS_PATH   = ROOT / "data"   / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data"   / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data"   / "raw" / "clinical_metadata.csv"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_PATH  = ROOT / "reports" / "gate_var_refined_log.json"

SOURCE_CKPT = CKPT_DIR / "v2_staged_var.pt"     # epoch-2 best checkpoint from Session 3
OUT_CKPT    = CKPT_DIR / "v2_var_refined.pt"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]

EPOCHS       = 20
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAR_WEIGHT   = 0.003

EARLY_STOP_ALPHA_MEAN = 0.42
EARLY_STOP_AUC        = 0.835


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


# ── Cheap batched validation: AUC + full alpha stats + Welch t-test ──────────
# alpha is returned by model.forward() regardless of batch size, so a single
# batched pass over val_loader gives everything needed without a separate
# expensive per-patient loop.

def validate_epoch(model: nn.Module, loader, device: torch.device) -> dict:
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
        "roc_auc":    float(roc_auc_score(labels, probs)),
        "alpha_mean": float(alphas.mean()),
        "alpha_std":  float(alphas.std()),
        "alpha_min":  float(alphas.min()),
        "alpha_max":  float(alphas.max()),
        "welch_t":    float(t_stat),
        "welch_p":    float(p_val),
        "borderline": borderline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — Refined Variance Regularization (from staged_var epoch 2) ══\n")

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
    print(f"var_weight={VAR_WEIGHT}  early stop: alpha_mean<{EARLY_STOP_ALPHA_MEAN} or val_auc<{EARLY_STOP_AUC}\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_auc = -1.0
    stopped_early = False
    stop_reason = None

    for epoch in range(1, EPOCHS + 1):
        mean_bce, mean_var, mean_total = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val = validate_epoch(model, val_loader, device)

        b = val["borderline"]
        hk = b.get("TCGA-BH-A0HK", {})
        hw = b.get("TCGA-BH-A0HW", {})
        print(f"  [Epoch {epoch:02d}] val_auc={val['roc_auc']:.4f}  "
              f"alpha: mean={val['alpha_mean']:.4f} std={val['alpha_std']:.4f} "
              f"min={val['alpha_min']:.4f} max={val['alpha_max']:.4f}  "
              f"welch_p={val['welch_p']:.4e}")
        print(f"             A0HK alpha={hk.get('alpha')} P(PR+)={hk.get('prob_pr_pos')}  "
              f"A0HW alpha={hw.get('alpha')} P(PR+)={hw.get('prob_pr_pos')}")

        history.append({
            "epoch":          epoch,
            "bce_loss":       mean_bce,
            "gate_var_loss":  mean_var,
            "total_loss":     mean_total,
            "validation_auc": val["roc_auc"],
            "alpha_mean":     val["alpha_mean"],
            "alpha_std":      val["alpha_std"],
            "alpha_min":      val["alpha_min"],
            "alpha_max":      val["alpha_max"],
            "welch_t":        val["welch_t"],
            "welch_p":        val["welch_p"],
            "borderline":     val["borderline"],
        })

        if val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_auc": val["roc_auc"]},
                OUT_CKPT,
            )
            print(f"  ✓ Checkpoint saved  (val_auc={val['roc_auc']:.4f} → {OUT_CKPT.name})")

        if val["alpha_mean"] < EARLY_STOP_ALPHA_MEAN:
            stopped_early = True
            stop_reason = f"alpha_mean {val['alpha_mean']:.4f} < {EARLY_STOP_ALPHA_MEAN}"
        elif val["roc_auc"] < EARLY_STOP_AUC:
            stopped_early = True
            stop_reason = f"val_auc {val['roc_auc']:.4f} < {EARLY_STOP_AUC}"

        if stopped_early:
            print(f"\n  ⚠ Early stop triggered at epoch {epoch}: {stop_reason}\n")
            break

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}  "
          f"epochs_run={len(history)}/{EPOCHS}  stopped_early={stopped_early}")

    output = {
        "model":          "GatedFusion-v2-var-refined",
        "source_ckpt":    str(SOURCE_CKPT),
        "var_weight":     VAR_WEIGHT,
        "epochs_planned": EPOCHS,
        "epochs_run":     len(history),
        "lr":             LR,
        "weight_decay":   WEIGHT_DECAY,
        "batch_size":     BATCH_SIZE,
        "frozen":         "all v1 params (genomic_projector, cross_attention, post_attn, head)",
        "trained":        "gated_fusion only",
        "early_stop_thresholds": {
            "alpha_mean_min": EARLY_STOP_ALPHA_MEAN,
            "val_auc_min":    EARLY_STOP_AUC,
        },
        "stopped_early":  stopped_early,
        "stop_reason":    stop_reason,
        "history":        history,
        "best_val_auc":   best_auc,
        "checkpoint":     str(OUT_CKPT),
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved → {RESULTS_PATH}")
    print("\n══ Done ════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
