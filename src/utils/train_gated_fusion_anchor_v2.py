"""
train_gated_fusion_anchor_v2.py
=================================
Second iteration of direction-anchored gate training, continuing from the
same source checkpoint as train_gated_fusion_anchor.py
(checkpoints/v2_staged_var.pt, epoch 2).

Why v2
-------
train_gated_fusion_anchor.py (anchor_weight=0.01, var_weight=0.003,
anchor_scale=5) stopped after 3/10 epochs on the alpha_mean<0.40 floor.
The anchor/var loss magnitudes (~0.00005-0.00013 combined) were dwarfed by
bce_loss (~0.039), so the anchor term had negligible actual restraining
effect — alpha_mean kept falling at essentially the same rate as the
unconstrained var-only run. This run:
  - raises anchor_weight 0.01 -> 0.10 (10x, competitive with BCE)
  - lowers var_weight 0.003 -> 0.001 (less downward pull)
  - sharpens anchor_scale 5 -> 20 (pushes anchor targets away from 0.5,
    giving the anchor loss more useful gradient given cosine_sim(v,g) is
    small in magnitude, ~[-0.07, -0.04] on this cohort)
  - drops the alpha_mean<0.40 floor entirely (it was masking whatever
    signal the stronger anchor produces); only val_auc<0.830 stops early

Loss formulation
-----------------
  v, g          = cross-attention output, genomic query token (both 512-dim)
  concordance   = cosine_similarity(v, g)                      in [-1, 1]
  anchor_target = sigmoid(concordance * 20)
  anchor_loss   = MSE(alpha, anchor_target)
  gate_var_loss = -var(alpha)
  total_loss    = bce_loss + 0.001 * gate_var_loss + 0.10 * anchor_loss

All parameters are frozen except the gate's Linear(1024, 1) + bias
(GatedFusion.gate). Up to 10 epochs.

Early stopping
---------------
  val_auc < 0.830 (checked after each epoch). No alpha_mean bound this run.

Per-epoch logging
------------------
  val_auc, alpha_mean, alpha_std, alpha_min, alpha_max
  bce_loss, anchor_loss, gate_var_loss (unweighted, mean over train batches)
  weighted contributions: 0.10*anchor_loss, 0.001*gate_var_loss
  Welch t sign + p-value on alpha by PR status (full cohort)
  Balanced Welch p (PR+ downsampled to match PR- count, random_state=42)
  alpha for TCGA-BH-A0HK, TCGA-BH-A0HW
  anchor_target mean/std (confirms whether sharpening is spreading targets)

Outputs
-------
  checkpoints/v2_anchored_v2.pt      — best checkpoint by val ROC-AUC
  reports/gate_anchored_v2_log.json  — full per-epoch log

Usage
-----
    python src/utils/train_gated_fusion_anchor_v2.py
"""

import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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
RESULTS_PATH  = ROOT / "reports" / "gate_anchored_v2_log.json"

SOURCE_CKPT = CKPT_DIR / "v2_staged_var.pt"     # epoch-2 best checkpoint from Session 3
OUT_CKPT    = CKPT_DIR / "v2_anchored_v2.pt"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]

EPOCHS       = 10
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAR_WEIGHT    = 0.001
ANCHOR_WEIGHT = 0.10
ANCHOR_SCALE  = 20.0

EARLY_STOP_AUC = 0.830
BALANCE_RANDOM_STATE = 42


# ── Forward pass that also exposes v (visual) and g (genomic) ─────────────────

def forward_with_vg(model: nn.Module, patch_emb, genomic_counts, patch_mask):
    g = model.genomic_projector(genomic_counts)          # (B, 512)
    query_seq = g.unsqueeze(1)                             # (B, 1, 512)
    attn_out, _ = model.cross_attention(
        query=query_seq, key=patch_emb, value=patch_emb,
        key_padding_mask=~patch_mask, need_weights=True,
    )
    v = attn_out.squeeze(1)                                 # (B, 512)
    fused, alpha = model.gated_fusion(v, g)
    fused  = model.post_attn(fused)
    logits = model.head(fused)
    return logits, alpha, v, g


def gate_var_loss_fn(alpha: torch.Tensor) -> torch.Tensor:
    return -torch.var(alpha)


def anchor_loss_fn(alpha: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    concordance   = F.cosine_similarity(v, g, dim=-1)          # (B,)
    anchor_target = torch.sigmoid(concordance * ANCHOR_SCALE)  # (B,)
    loss = F.mse_loss(alpha.squeeze(-1), anchor_target)
    return loss, anchor_target


# ── Training epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch) -> dict:
    model.train()
    bce_sum = anchor_sum = var_sum = total_sum = 0.0

    for batch in loader:
        patch_emb  = batch["patch_embeddings"].to(device)
        genomic    = batch["genomic_counts"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels     = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, alpha, v, g = forward_with_vg(model, patch_emb, genomic, patch_mask)

        bce = criterion(logits, labels)
        var = gate_var_loss_fn(alpha)
        anchor, _ = anchor_loss_fn(alpha, v, g)
        loss = bce + VAR_WEIGHT * var + ANCHOR_WEIGHT * anchor

        loss.backward()
        optimizer.step()

        bce_sum    += bce.item()
        anchor_sum += anchor.item()
        var_sum    += var.item()
        total_sum  += loss.item()

    n = len(loader)
    result = {
        "bce_loss":       bce_sum / n,
        "anchor_loss":    anchor_sum / n,
        "gate_var_loss":  var_sum / n,
        "total_loss":     total_sum / n,
    }
    print(f"  [Epoch {epoch:02d}] bce={result['bce_loss']:.4f}  "
          f"anchor={result['anchor_loss']:.4f} (weighted={ANCHOR_WEIGHT*result['anchor_loss']:.4f})  "
          f"gate_var={result['gate_var_loss']:.4f} (weighted={VAR_WEIGHT*result['gate_var_loss']:.4f})  "
          f"total={result['total_loss']:.4f}")
    return result


# ── Balanced Welch t-test (PR+ downsampled to match PR- count) ────────────────

def balanced_welch(alphas: np.ndarray, labels: np.ndarray, random_state: int) -> tuple[float, float]:
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(random_state)

    if len(pos_idx) > len(neg_idx):
        pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=False)
    elif len(neg_idx) > len(pos_idx):
        neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)

    t_stat, p_val = stats.ttest_ind(alphas[pos_idx], alphas[neg_idx], equal_var=False)
    return float(t_stat), float(p_val)


# ── Cheap batched validation: AUC + alpha stats + Welch (full & balanced) ─────

def validate_epoch(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    probs_out, labels_out, alphas_out, targets_out, pids_out = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels     = batch["label"].float().unsqueeze(1).to(device)
            pids       = batch["patient_id"]

            logits, alpha, v, g = forward_with_vg(model, patch_emb, genomic, patch_mask)
            probs = torch.sigmoid(logits)
            _, anchor_target = anchor_loss_fn(alpha, v, g)

            probs_out.extend(probs.squeeze(-1).tolist())
            labels_out.extend(labels.squeeze(-1).tolist())
            alphas_out.extend(alpha.squeeze(-1).tolist())
            targets_out.extend(anchor_target.tolist())
            pids_out.extend(pids)

    probs   = np.array(probs_out)
    labels  = np.array(labels_out)
    alphas  = np.array(alphas_out)
    targets = np.array(targets_out)

    alpha_pos = alphas[labels == 1]
    alpha_neg = alphas[labels == 0]
    t_stat, p_val = stats.ttest_ind(alpha_pos, alpha_neg, equal_var=False)
    bal_t, bal_p  = balanced_welch(alphas, labels, BALANCE_RANDOM_STATE)

    by_pid = dict(zip(pids_out, zip(alphas_out, probs_out, labels_out)))
    borderline = {
        pid: {"alpha": by_pid[pid][0], "prob_pr_pos": by_pid[pid][1], "label": int(by_pid[pid][2])}
        for pid in BORDERLINE_IDS if pid in by_pid
    }

    return {
        "roc_auc":         float(roc_auc_score(labels, probs)),
        "alpha_mean":      float(alphas.mean()),
        "alpha_std":       float(alphas.std()),
        "alpha_min":       float(alphas.min()),
        "alpha_max":       float(alphas.max()),
        "welch_t":         float(t_stat),
        "welch_sign":      "positive (PR+ higher)" if t_stat > 0 else "negative (PR- higher)",
        "welch_p":         float(p_val),
        "balanced_welch_t": bal_t,
        "balanced_welch_p": bal_p,
        "anchor_target_mean": float(targets.mean()),
        "anchor_target_std":  float(targets.std()),
        "borderline":      borderline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — Direction-Anchored Training v2 (from staged_var epoch 2) ══\n")

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

    # Freeze all parameters except the gate's Linear(1024, 1) + bias.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.gated_fusion.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (gate only)")
    print(f"var_weight={VAR_WEIGHT}  anchor_weight={ANCHOR_WEIGHT}  anchor_scale={ANCHOR_SCALE}")
    print(f"Early stop: val_auc<{EARLY_STOP_AUC} (no alpha_mean floor this run)\n")

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
        train_out = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val = validate_epoch(model, val_loader, device)

        b = val["borderline"]
        hk = b.get("TCGA-BH-A0HK", {})
        hw = b.get("TCGA-BH-A0HW", {})
        print(f"  [Epoch {epoch:02d}] val_auc={val['roc_auc']:.4f}  "
              f"alpha: mean={val['alpha_mean']:.4f} std={val['alpha_std']:.4f} "
              f"min={val['alpha_min']:.4f} max={val['alpha_max']:.4f}")
        print(f"             welch={val['welch_sign']} p={val['welch_p']:.4e}  "
              f"balanced_p={val['balanced_welch_p']:.4e}")
        print(f"             A0HK alpha={hk.get('alpha')}  A0HW alpha={hw.get('alpha')}  "
              f"anchor_target mean={val['anchor_target_mean']:.4f} std={val['anchor_target_std']:.4f}")

        history.append({
            "epoch":               epoch,
            "bce_loss":            train_out["bce_loss"],
            "anchor_loss":         train_out["anchor_loss"],
            "gate_var_loss":       train_out["gate_var_loss"],
            "anchor_loss_weighted":   ANCHOR_WEIGHT * train_out["anchor_loss"],
            "gate_var_loss_weighted": VAR_WEIGHT * train_out["gate_var_loss"],
            "total_loss":          train_out["total_loss"],
            "validation_auc":      val["roc_auc"],
            "alpha_mean":          val["alpha_mean"],
            "alpha_std":           val["alpha_std"],
            "alpha_min":           val["alpha_min"],
            "alpha_max":           val["alpha_max"],
            "welch_t":             val["welch_t"],
            "welch_sign":          val["welch_sign"],
            "welch_p":             val["welch_p"],
            "balanced_welch_t":    val["balanced_welch_t"],
            "balanced_welch_p":    val["balanced_welch_p"],
            "anchor_target_mean":  val["anchor_target_mean"],
            "anchor_target_std":   val["anchor_target_std"],
            "borderline":          val["borderline"],
        })

        if val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_auc": val["roc_auc"]},
                OUT_CKPT,
            )
            print(f"  ✓ Checkpoint saved  (val_auc={val['roc_auc']:.4f} → {OUT_CKPT.name})")

        if val["roc_auc"] < EARLY_STOP_AUC:
            stopped_early = True
            stop_reason = f"val_auc {val['roc_auc']:.4f} < {EARLY_STOP_AUC}"
            print(f"\n  ⚠ Early stop triggered at epoch {epoch}: {stop_reason}\n")
            break

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}  "
          f"epochs_run={len(history)}/{EPOCHS}  stopped_early={stopped_early}")

    output = {
        "model":          "GatedFusion-v2-anchored-v2",
        "source_ckpt":    str(SOURCE_CKPT),
        "var_weight":     VAR_WEIGHT,
        "anchor_weight":  ANCHOR_WEIGHT,
        "anchor_scale":   ANCHOR_SCALE,
        "epochs_planned": EPOCHS,
        "epochs_run":     len(history),
        "lr":             LR,
        "weight_decay":   WEIGHT_DECAY,
        "batch_size":     BATCH_SIZE,
        "frozen":         "all params except gated_fusion.gate (Linear(1024,1) + bias)",
        "trained":        "gated_fusion only",
        "early_stop_thresholds": {"val_auc_min": EARLY_STOP_AUC},
        "balance_random_state":  BALANCE_RANDOM_STATE,
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
