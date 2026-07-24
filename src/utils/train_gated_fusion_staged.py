"""
train_gated_fusion_staged.py
============================
Two-stage training of Gated-Fusion v2.

Stage 1 (15 epochs, lr=1e-4)
  Gate parameters frozen; cross-attention encoder, genomic projector,
  post-attention projection, and classification head trained freely.
  Best checkpoint (val ROC-AUC) saved to checkpoints/best_model_v2_stage1.pt.

Stage 2 (10 epochs, lr=1e-5)
  Loads best Stage 1 checkpoint; all parameters unfrozen including Wg/bg.
  Lower lr prevents destabilising the encoder representations already learned.
  Best checkpoint saved to checkpoints/best_model_v2_staged.pt.

Evaluation (held-out cohort, best Stage 2 checkpoint)
  ROC-AUC, PR-AUC, Brier Score
  Per-patient α: mean, std, min, max
  α < 0.5 vs α ≥ 0.5 patient counts
  Borderline-case α for TCGA-BH-A0HK and TCGA-BH-A0HW

Results saved to reports/gated_fusion_staged_results.json

Usage
-----
    python src/utils/train_gated_fusion_staged.py
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
RESULTS_PATH  = ROOT / "reports" / "gated_fusion_staged_results.json"

BORDERLINE_IDS = {"TCGA-BH-A0HK", "TCGA-BH-A0HW"}

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 10
BATCH_SIZE    = 4
WEIGHT_DECAY  = 1e-4
LR_STAGE1     = 1e-4
LR_STAGE2     = 1e-5

STAGE1_CKPT = CKPT_DIR / "best_model_v2_stage1.pt"
STAGE2_CKPT = CKPT_DIR / "best_model_v2_staged.pt"


# ── Training helpers ──────────────────────────────────────────────────────────

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


def run_stage(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    ckpt_path: Path,
    stage_label: str,
) -> tuple[list[float], list[float], float]:
    """Generic training loop; saves best checkpoint to ckpt_path."""
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc  = -1.0

    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n── {stage_label} ({'─' * (54 - len(stage_label))})")
    print(f"  epochs={epochs}, lr={lr}, weight_decay={WEIGHT_DECAY}")
    print(f"  trainable params: {trainable:,}")
    if frozen:
        print(f"  frozen: {', '.join(frozen)}")
    print("─" * 60)

    history_loss: list[float] = []
    history_auc:  list[float] = []

    for epoch in range(1, epochs + 1):
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
    print(f"{stage_label} complete. Best val_auc={best_auc:.4f}\n")
    return history_loss, history_auc, best_auc


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, val_data: list[dict]) -> dict:
    model = model.to("cpu")
    model.eval()

    probs_out, labels_out, alphas_out, pids_out = [], [], [], []

    for patient in val_data:
        pe = torch.tensor(
            patient["patch_embeddings"], dtype=torch.float32
        ).unsqueeze(0)
        gc = torch.tensor(
            patient["genomic_counts"], dtype=torch.float32
        ).unsqueeze(0)
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

    borderline = {}
    for pid, a, prob, label in zip(pids_out, alphas, probs, labels):
        if pid in BORDERLINE_IDS:
            borderline[pid] = {
                "alpha": float(a),
                "prob_pr_pos": float(prob),
                "label": int(label),
            }

    return {
        "roc_auc":     float(roc_auc_score(labels, probs)),
        "pr_auc":      float(average_precision_score(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "alpha_stats": {
            "mean": float(alphas.mean()),
            "std":  float(alphas.std()),
            "min":  float(alphas.min()),
            "max":  float(alphas.max()),
        },
        "alpha_lt_0_5": int((alphas < 0.5).sum()),
        "alpha_ge_0_5": int((alphas >= 0.5).sum()),
        "borderline_cases": borderline,
        "n_val_patients": len(val_data),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n══ Gated-Fusion v2 — Two-Stage Training ════════════════════════\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    pos = sum(d["label"] == 1.0 for d in patient_data)
    neg = sum(d["label"] == 0.0 for d in patient_data)
    print(f"Total patients after QC: {len(patient_data)}  (PR+={pos}, PR−={neg})")

    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split,
    )
    print(f"Train: {len(train_data)} patients    Val (held-out): {len(val_data)} patients\n")

    train_loader = build_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")

    # ── Stage 1: freeze gate ──────────────────────────────────────────────────
    model = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim).to(device)

    for param in model.gated_fusion.parameters():
        param.requires_grad = False

    s1_loss, s1_auc, s1_best = run_stage(
        model, train_loader, val_loader, device,
        epochs=STAGE1_EPOCHS, lr=LR_STAGE1,
        ckpt_path=STAGE1_CKPT, stage_label="Stage 1 (gate frozen)",
    )

    # ── Stage 2: unfreeze all, lower lr ───────────────────────────────────────
    ckpt1 = torch.load(STAGE1_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt1["model_state"])
    print(f"Loaded Stage 1 checkpoint: epoch={ckpt1['epoch']}  val_auc={ckpt1['val_auc']:.4f}")

    for param in model.parameters():
        param.requires_grad = True

    s2_loss, s2_auc, s2_best = run_stage(
        model, train_loader, val_loader, device,
        epochs=STAGE2_EPOCHS, lr=LR_STAGE2,
        ckpt_path=STAGE2_CKPT, stage_label="Stage 2 (all params, fine-tune gate)",
    )

    # ── Evaluate from best Stage 2 checkpoint ─────────────────────────────────
    ckpt2 = torch.load(STAGE2_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt2["model_state"])
    print(f"Loaded Stage 2 checkpoint: epoch={ckpt2['epoch']}  val_auc={ckpt2['val_auc']:.4f}\n")

    print("── Held-Out Cohort Evaluation ────────────────────────────────────")
    results = evaluate(model, val_data)

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print("  " + "─" * 30)
    print(f"  {'ROC-AUC':<20} {results['roc_auc']:>10.4f}")
    print(f"  {'PR-AUC':<20} {results['pr_auc']:>10.4f}")
    print(f"  {'Brier Score':<20} {results['brier_score']:>10.4f}")

    a = results["alpha_stats"]
    print(f"\n  GatedFusion α  (N={results['n_val_patients']} patients):")
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

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "model": "GatedFusion-v2-staged",
        "stage1": {
            "epochs":         STAGE1_EPOCHS,
            "lr":             LR_STAGE1,
            "weight_decay":   WEIGHT_DECAY,
            "gate_frozen":    True,
            "best_epoch":     int(ckpt1["epoch"]),
            "best_val_auc":   float(ckpt1["val_auc"]),
            "history_loss":   s1_loss,
            "history_val_auc": s1_auc,
        },
        "stage2": {
            "epochs":         STAGE2_EPOCHS,
            "lr":             LR_STAGE2,
            "weight_decay":   WEIGHT_DECAY,
            "gate_frozen":    False,
            "best_epoch":     int(ckpt2["epoch"]),
            "best_val_auc":   float(ckpt2["val_auc"]),
            "history_loss":   s2_loss,
            "history_val_auc": s2_auc,
            "checkpoint":     str(STAGE2_CKPT),
        },
        "training_shared": {
            "batch_size":        BATCH_SIZE,
            "random_state":      42,
            "n_train_patients":  len(train_data),
            "n_val_patients":    len(val_data),
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
