"""
train_gated_fusion_anchor.py
==============================
Direction-anchored gate training, continuing from the epoch-2 checkpoint of
train_gated_fusion_var_reg.py (checkpoints/v2_staged_var.pt).

Motivation
-----------
balanced_subsample_diagnostic.py showed that alpha's *direction* relative to
PR status flips between checkpoints: in the e2e checkpoint PR+ patients have
higher alpha (t=+7.37), while in the var-epoch-2 checkpoint PR- patients have
higher alpha (t=-3.25) — both differences are statistically significant and
survive class-balancing, so this isn't noise. That means the sign of alpha
is not a stable, interpretable property of the architecture; it depends on
which training run produced the checkpoint.

This run adds an explicit direction anchor: alpha is pushed toward a target
derived directly from cosine_similarity(v, g), so "high alpha" is anchored
to mean "visual and genomic representations agree" by construction, instead
of being free to drift into whatever direction correlates with PR status in
a given training run.

Loss formulation
-----------------
  v, g          = cross-attention output, genomic query token (both 512-dim)
  concordance   = cosine_similarity(v, g)                      in [-1, 1]
  anchor_target = sigmoid(concordance * 5)                     in (0, 1)
  anchor_loss   = MSE(alpha, anchor_target)
  gate_var_loss = -var(alpha)
  total_loss    = bce_loss + 0.003 * gate_var_loss + 0.01 * anchor_loss

All parameters are frozen except the gate's Linear(1024, 1) + bias
(GatedFusion.gate). Up to 10 epochs.

Early stopping (checked after each epoch)
-------------------------------------------
  val_auc < 0.835, OR
  alpha_mean < 0.40 or alpha_mean > 0.60

Validation cost
----------------
Single cheap batched pass (same val_loader used for AUC), also exposing v/g
via forward_with_vg so cosine_sim, alpha stats, Welch t-test, and borderline
lookups are all collected in one pass.

Per-epoch logging
------------------
  val_auc, alpha_mean, alpha_std, alpha_min, alpha_max
  anchor_loss, gate_var_loss (mean over training batches)
  Welch t sign + p-value on alpha by PR status
  alpha for TCGA-BH-A0HK, TCGA-BH-A0HW
  cosine_sim mean for PR+ and PR- separately

Outputs
-------
  checkpoints/v2_anchored.pt      — best checkpoint by val ROC-AUC
  reports/gate_anchored_log.json  — full per-epoch log

Usage
-----
    python src/utils/train_gated_fusion_anchor.py
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
RESULTS_PATH  = ROOT / "reports" / "gate_anchored_log.json"

SOURCE_CKPT = CKPT_DIR / "v2_staged_var.pt"     # epoch-2 best checkpoint from Session 3
OUT_CKPT    = CKPT_DIR / "v2_anchored.pt"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]

EPOCHS       = 10
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAR_WEIGHT    = 0.003
ANCHOR_WEIGHT = 0.01
ANCHOR_SCALE  = 5.0

EARLY_STOP_AUC        = 0.835
EARLY_STOP_ALPHA_LOW  = 0.40
EARLY_STOP_ALPHA_HIGH = 0.60


# ── Forward pass that also exposes v (visual) and g (genomic) ─────────────────
# model.forward() only returns (logits, attn_weights, alpha); the anchor loss
# needs the intermediate vectors GatedFusion consumes, so we replicate the
# forward pass step-by-step rather than modify the model's public interface.

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
    return loss, concordance


# ── Training epoch ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch) -> tuple[float, float, float]:
    model.train()
    bce_sum = anchor_sum = var_sum = total_sum = 0.0

    for batch in loader:
        patch_emb  = batch["patch_embeddings"].to(device)
        genomic    = batch["genomic_counts"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels     = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, alpha, v, g = forward_with_vg(model, patch_emb, genomic, patch_mask)

        bce, var = criterion(logits, labels), gate_var_loss_fn(alpha)
        anchor, _ = anchor_loss_fn(alpha, v, g)
        loss = bce + VAR_WEIGHT * var + ANCHOR_WEIGHT * anchor

        loss.backward()
        optimizer.step()

        bce_sum    += bce.item()
        anchor_sum += anchor.item()
        var_sum    += var.item()
        total_sum  += loss.item()

    n = len(loader)
    mean_bce, mean_anchor, mean_var, mean_total = (
        bce_sum / n, anchor_sum / n, var_sum / n, total_sum / n
    )
    print(f"  [Epoch {epoch:02d}] bce={mean_bce:.4f}  anchor={mean_anchor:.4f}  "
          f"gate_var={mean_var:.4f}  total={mean_total:.4f}")
    return mean_bce, mean_anchor, mean_var, mean_total


# ── Cheap batched validation: AUC + alpha stats + Welch t-test + cosine_sim ───

def validate_epoch(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    probs_out, labels_out, alphas_out, cos_out, pids_out = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels     = batch["label"].float().unsqueeze(1).to(device)
            pids       = batch["patient_id"]

            logits, alpha, v, g = forward_with_vg(model, patch_emb, genomic, patch_mask)
            probs       = torch.sigmoid(logits)
            concordance = F.cosine_similarity(v, g, dim=-1)

            probs_out.extend(probs.squeeze(-1).tolist())
            labels_out.extend(labels.squeeze(-1).tolist())
            alphas_out.extend(alpha.squeeze(-1).tolist())
            cos_out.extend(concordance.tolist())
            pids_out.extend(pids)

    probs  = np.array(probs_out)
    labels = np.array(labels_out)
    alphas = np.array(alphas_out)
    cos    = np.array(cos_out)

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
        "alpha_mean":    float(alphas.mean()),
        "alpha_std":     float(alphas.std()),
        "alpha_min":     float(alphas.min()),
        "alpha_max":     float(alphas.max()),
        "welch_t":       float(t_stat),
        "welch_sign":    "positive (PR+ higher)" if t_stat > 0 else "negative (PR- higher)",
        "welch_p":       float(p_val),
        "cos_pos_mean":  float(cos[labels == 1].mean()),
        "cos_neg_mean":  float(cos[labels == 0].mean()),
        "borderline":    borderline,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — Direction-Anchored Training (from staged_var epoch 2) ══\n")

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
    print(f"Early stop: val_auc<{EARLY_STOP_AUC} or alpha_mean outside "
          f"[{EARLY_STOP_ALPHA_LOW}, {EARLY_STOP_ALPHA_HIGH}]\n")

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
        mean_bce, mean_anchor, mean_var, mean_total = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val = validate_epoch(model, val_loader, device)

        b = val["borderline"]
        hk = b.get("TCGA-BH-A0HK", {})
        hw = b.get("TCGA-BH-A0HW", {})
        print(f"  [Epoch {epoch:02d}] val_auc={val['roc_auc']:.4f}  "
              f"alpha: mean={val['alpha_mean']:.4f} std={val['alpha_std']:.4f} "
              f"min={val['alpha_min']:.4f} max={val['alpha_max']:.4f}  "
              f"welch={val['welch_sign']} p={val['welch_p']:.4e}")
        print(f"             A0HK alpha={hk.get('alpha')}  A0HW alpha={hw.get('alpha')}  "
              f"cos_pos={val['cos_pos_mean']:.4f}  cos_neg={val['cos_neg_mean']:.4f}")

        history.append({
            "epoch":          epoch,
            "bce_loss":       mean_bce,
            "anchor_loss":    mean_anchor,
            "gate_var_loss":  mean_var,
            "total_loss":     mean_total,
            "validation_auc": val["roc_auc"],
            "alpha_mean":     val["alpha_mean"],
            "alpha_std":      val["alpha_std"],
            "alpha_min":      val["alpha_min"],
            "alpha_max":      val["alpha_max"],
            "welch_t":        val["welch_t"],
            "welch_sign":     val["welch_sign"],
            "welch_p":        val["welch_p"],
            "cos_pos_mean":   val["cos_pos_mean"],
            "cos_neg_mean":   val["cos_neg_mean"],
            "borderline":     val["borderline"],
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
        elif val["alpha_mean"] < EARLY_STOP_ALPHA_LOW:
            stopped_early = True
            stop_reason = f"alpha_mean {val['alpha_mean']:.4f} < {EARLY_STOP_ALPHA_LOW}"
        elif val["alpha_mean"] > EARLY_STOP_ALPHA_HIGH:
            stopped_early = True
            stop_reason = f"alpha_mean {val['alpha_mean']:.4f} > {EARLY_STOP_ALPHA_HIGH}"

        if stopped_early:
            print(f"\n  ⚠ Early stop triggered at epoch {epoch}: {stop_reason}\n")
            break

    print(f"\nTraining complete. Best val_auc={best_auc:.4f}  "
          f"epochs_run={len(history)}/{EPOCHS}  stopped_early={stopped_early}")

    output = {
        "model":          "GatedFusion-v2-anchored",
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
        "early_stop_thresholds": {
            "val_auc_min":      EARLY_STOP_AUC,
            "alpha_mean_range": [EARLY_STOP_ALPHA_LOW, EARLY_STOP_ALPHA_HIGH],
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
