"""
train_gated_fusion_e2e.py
=========================
End-to-end training and evaluation of Gated-Fusion v2.

All parameters — cross-attention, gated-fusion (Wg, bg), post-attention
projection, and classifier head — are initialised from scratch and trained
jointly.  The gated-fusion gate initialises to α = 0.5 for every patient at
epoch-0 (zero-weight init in GatedFusion), then adapts freely during training.

Training
--------
  - 80 / 20 stratified split, random_state=42
  - Adam, lr=1e-4, weight_decay=1e-4
  - 20 epochs, BCEWithLogitsLoss
  - Best checkpoint (highest val ROC-AUC) → checkpoints/best_model_v2_e2e.pt

Evaluation (held-out val cohort)
---------------------------------
  - ROC-AUC, PR-AUC, Brier Score
  - Per-patient α: mean, std, min, max
  - Count of α < 0.5 vs α > 0.5
  - Borderline-case α for TCGA-BH-A0HK and TCGA-BH-A0HW

Results saved to reports/gated_fusion_e2e_results.json

Usage
-----
    python src/utils/train_gated_fusion_e2e.py
"""

import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
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
RESULTS_PATH  = ROOT / "reports" / "gated_fusion_e2e_results.json"

BORDERLINE_IDS = {"TCGA-BH-A0HK", "TCGA-BH-A0HW"}

LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 20
BATCH_SIZE   = 4


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        patch_emb  = batch["patch_embeddings"].to(device)
        genomic    = batch["genomic_counts"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels     = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, _, _ = model(patch_emb, genomic, patch_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    mean_loss = total_loss / len(loader)
    print(f"  [Epoch {epoch:02d}] train_loss={mean_loss:.4f}")
    return mean_loss


def val_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    epoch: int,
) -> float:
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
    auc_val = auc.item() if isinstance(auc, torch.Tensor) else float(auc)
    print(f"  [Epoch {epoch:02d}] val_auc={auc_val:.4f}")
    return auc_val


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
) -> tuple[list[float], list[float], float]:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "best_model_v2_e2e.pt"

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc  = -1.0

    history_loss: list[float] = []
    history_auc:  list[float] = []

    print(f"\nTraining Gated-Fusion v2 (e2e, scratch) on {device}")
    print(f"  epochs={EPOCHS}, lr={LR}, weight_decay={WEIGHT_DECAY}, batch={BATCH_SIZE}")
    print("─" * 60)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        auc  = val_epoch(model, val_loader, device, epoch)

        history_loss.append(loss)
        history_auc.append(auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_auc": auc},
                ckpt_path,
            )
            print(f"  ✓ Checkpoint saved  (val_auc={auc:.4f} → {ckpt_path.name})")

    print("─" * 60)
    print(f"Training complete. Best val_auc={best_auc:.4f}\n")
    return history_loss, history_auc, best_auc


# ── Held-out evaluation ───────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    val_data: list[dict],
    device: torch.device,
) -> dict:
    """
    Per-patient inference on the val cohort; returns all metrics + α stats.
    Runs on CPU to avoid MPS need_weights assertion for MultiheadAttention.
    """
    model = model.to("cpu")
    model.eval()

    probs_out, labels_out, alphas_out, pids_out = [], [], [], []

    for patient in val_data:
        pe = torch.tensor(
            patient["patch_embeddings"], dtype=torch.float32
        ).unsqueeze(0)  # (1, N, 768)
        gc = torch.tensor(
            patient["genomic_counts"], dtype=torch.float32
        ).unsqueeze(0)  # (1, G)
        pm = torch.ones(1, pe.shape[1], dtype=torch.bool)

        with torch.no_grad():
            logit, _, alpha = model(pe, gc, pm)

        probs_out.append(torch.sigmoid(logit).item())
        labels_out.append(float(patient["label"]))
        alphas_out.append(float(alpha.squeeze().item()))
        pids_out.append(patient.get("patient_id", f"patient_{len(probs_out)}"))

    probs  = np.array(probs_out)
    labels = np.array(labels_out)
    alphas = np.array(alphas_out)

    roc_auc  = float(roc_auc_score(labels, probs))
    pr_auc   = float(average_precision_score(labels, probs))
    brier    = float(brier_score_loss(labels, probs))

    alpha_lt = int((alphas < 0.5).sum())
    alpha_gt = int((alphas >= 0.5).sum())

    borderline = {}
    for pid, alpha_val, prob, label in zip(pids_out, alphas, probs, labels):
        if pid in BORDERLINE_IDS:
            borderline[pid] = {
                "alpha": float(alpha_val),
                "prob_pr_pos": float(prob),
                "label": int(label),
            }

    return {
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc,
        "brier_score":  brier,
        "alpha_stats": {
            "mean": float(alphas.mean()),
            "std":  float(alphas.std()),
            "min":  float(alphas.min()),
            "max":  float(alphas.max()),
        },
        "alpha_lt_0_5": alpha_lt,
        "alpha_ge_0_5": alpha_gt,
        "borderline_cases": borderline,
        "n_val_patients": len(val_data),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — End-to-End Training ═══════════════════════\n")

    # ── Config + data ─────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    pos = sum(d["label"] == 1.0 for d in patient_data)
    neg = sum(d["label"] == 0.0 for d in patient_data)
    print(f"Total patients after QC: {len(patient_data)}  (PR+={pos}, PR−={neg})")

    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data,
        test_size=0.2,
        random_state=42,
        stratify=labels_for_split,
    )
    print(f"Train: {len(train_data)} patients    Val (held-out): {len(val_data)} patients\n")

    train_loader = build_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model — initialised from scratch ──────────────────────────────────────
    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")  # MPS need_weights restriction; CPU is fine here
    model       = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_gate   = sum(p.numel() for p in model.gated_fusion.parameters())
    print(f"Model: PathoGenomicFusionModel  total_params={n_params:,}")
    print(f"  of which GatedFusion (Wg, bg): {n_gate:,} params\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    history_loss, history_auc, best_train_auc = train(
        model, train_loader, val_loader, device
    )

    # ── Load best checkpoint for evaluation ───────────────────────────────────
    ckpt_path = CKPT_DIR / "best_model_v2_e2e.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best checkpoint: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}\n")

    # ── Evaluate on held-out cohort ───────────────────────────────────────────
    print("── Held-Out Cohort Evaluation ────────────────────────────────────")
    results = evaluate(model, val_data, device)

    w = 62
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print("  " + "─" * 30)
    print(f"  {'ROC-AUC':<20} {results['roc_auc']:>10.4f}")
    print(f"  {'PR-AUC':<20} {results['pr_auc']:>10.4f}")
    print(f"  {'Brier Score':<20} {results['brier_score']:>10.4f}")

    a = results["alpha_stats"]
    print(f"\n  GatedFusion α distribution (N={results['n_val_patients']} patients):")
    print(f"    mean={a['mean']:.4f}  std={a['std']:.4f}  "
          f"min={a['min']:.4f}  max={a['max']:.4f}")
    print(f"    α < 0.5 : {results['alpha_lt_0_5']} patients")
    print(f"    α ≥ 0.5 : {results['alpha_ge_0_5']} patients")

    print(f"\n  Borderline cases:")
    if results["borderline_cases"]:
        for pid, info in sorted(results["borderline_cases"].items()):
            label_str = "PR+" if info["label"] == 1 else "PR−"
            print(f"    {pid}  α={info['alpha']:.4f}  "
                  f"P(PR+)={info['prob_pr_pos']:.3f}  label={label_str}")
    else:
        print("    (Patients not present in this validation split)")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "model":    "GatedFusion-v2-e2e",
        "training": {
            "epochs":            EPOCHS,
            "lr":                LR,
            "weight_decay":      WEIGHT_DECAY,
            "batch_size":        BATCH_SIZE,
            "random_state":      42,
            "n_train_patients":  len(train_data),
            "n_val_patients":    len(val_data),
            "best_checkpoint":   str(ckpt_path),
            "best_epoch":        int(ckpt["epoch"]),
            "best_val_auc":      float(ckpt["val_auc"]),
            "history_loss":      history_loss,
            "history_val_auc":   history_auc,
        },
        "evaluation": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n  Results saved → {RESULTS_PATH}")
    print("\n══ Done ════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
