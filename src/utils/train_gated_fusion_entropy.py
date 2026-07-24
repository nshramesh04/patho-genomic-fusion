"""
train_gated_fusion_entropy.py
==============================
End-to-end Gated-Fusion v2 with binary-entropy regularisation on the gate
output α to prevent collapse toward zero.

Loss formulation
----------------
  entropy_loss = -mean(α·log(α+ε) + (1-α)·log(1-α+ε))   # mean H(Bernoulli(α))
  total_loss   = bce_loss - λ · entropy_loss

Minimising total_loss → maximising H(α) → pushes α toward 0.5.
This prevents the gate from collapsing to the all-suppressed regime observed
in the plain e2e run (α ≈ 0.17 mean) or the near-neutral staged run (α ≈ 0.51).

Experiments
-----------
  λ ∈ {0.05, 0.1, 0.2}  — each trained from scratch, 20 epochs
  Results saved individually to reports/gated_fusion_entropy_{lambda}.json

Shared config (all experiments)
--------------------------------
  80/20 stratified split, random_state=42 → 760 train / 191 held-out
  Adam lr=1e-4, weight_decay=1e-4, BCEWithLogitsLoss + entropy reg.
  Best checkpoint (val ROC-AUC) → checkpoints/best_model_v2_entropy_{lambda}.pt

Usage
-----
  python src/utils/train_gated_fusion_entropy.py
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
REPORTS_DIR   = ROOT / "reports"

BORDERLINE_IDS = {"TCGA-BH-A0HK", "TCGA-BH-A0HW"}
LAMBDAS        = [0.05, 0.1, 0.2]

EPOCHS       = 20
BATCH_SIZE   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPS          = 1e-8


# ── Entropy regularisation ────────────────────────────────────────────────────

def gate_entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    Mean binary entropy of the gate scalar α ∈ (0,1).
    H = -mean( α·log(α+ε) + (1-α)·log(1-α+ε) )
    Maximum at α=0.5 (H=log2≈0.693), zero at α∈{0,1}.
    """
    return -torch.mean(
        alpha * torch.log(alpha + EPS)
        + (1.0 - alpha) * torch.log(1.0 - alpha + EPS)
    )


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    lambda_ent: float,
) -> tuple[float, float, float]:
    """Returns (mean_total_loss, mean_bce_loss, mean_entropy_loss)."""
    model.train()
    total_loss_sum = bce_sum = ent_sum = 0.0

    for batch in loader:
        patch_emb  = batch["patch_embeddings"].to(device)
        genomic    = batch["genomic_counts"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels     = batch["label"].float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, _, alpha = model(patch_emb, genomic, patch_mask)

        bce  = criterion(logits, labels)
        ent  = gate_entropy_loss(alpha)
        loss = bce - lambda_ent * ent

        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        bce_sum        += bce.item()
        ent_sum        += ent.item()

    n = len(loader)
    mean_total = total_loss_sum / n
    mean_bce   = bce_sum / n
    mean_ent   = ent_sum / n
    print(f"  [Epoch {epoch:02d}] total={mean_total:.4f}  "
          f"bce={mean_bce:.4f}  entropy={mean_ent:.4f}")
    return mean_total, mean_bce, mean_ent


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


def train_one_experiment(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    lambda_ent: float,
    ckpt_path: Path,
) -> tuple[dict, dict]:
    """
    Trains model from its current (freshly initialised) state.
    Returns (history dict, best-checkpoint info dict).
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc  = -1.0
    best_ckpt_info: dict = {}

    hist = {"total_loss": [], "bce_loss": [], "entropy_loss": [], "val_auc": []}

    label = f"λ={lambda_ent}"
    print(f"\n{'═'*60}")
    print(f"  Experiment: entropy regularisation  {label}")
    print(f"  epochs={EPOCHS}  lr={LR}  weight_decay={WEIGHT_DECAY}")
    print(f"{'─'*60}")

    for epoch in range(1, EPOCHS + 1):
        tl, bce, ent = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, lambda_ent
        )
        auc = val_epoch(model, val_loader, device, epoch)

        hist["total_loss"].append(tl)
        hist["bce_loss"].append(bce)
        hist["entropy_loss"].append(ent)
        hist["val_auc"].append(auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_auc": auc},
                ckpt_path,
            )
            best_ckpt_info = {"epoch": epoch, "val_auc": auc}
            print(f"  ✓ Checkpoint saved  (val_auc={auc:.4f} → {ckpt_path.name})")

    print(f"{'─'*60}")
    print(f"  {label} complete. Best val_auc={best_auc:.4f}\n")
    return hist, best_ckpt_info


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
        "roc_auc":      float(roc_auc_score(labels, probs)),
        "pr_auc":       float(average_precision_score(labels, probs)),
        "brier_score":  float(brier_score_loss(labels, probs)),
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
    print("\n══ Gated-Fusion v2 — Entropy Regularisation Sweep ══════════════\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    # Load data once; reused across all three experiments.
    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    pos = sum(d["label"] == 1.0 for d in patient_data)
    neg = sum(d["label"] == 0.0 for d in patient_data)
    print(f"Patients after QC: {len(patient_data)}  (PR+={pos}, PR−={neg})")

    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split,
    )
    print(f"Train: {len(train_data)}    Val (held-out): {len(val_data)}\n")

    train_loader = build_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")

    # ── Run each λ ────────────────────────────────────────────────────────────
    for lambda_ent in LAMBDAS:
        lambda_str = str(lambda_ent).replace(".", "p")   # "0.05" → "0p05"
        ckpt_path  = CKPT_DIR / f"best_model_v2_entropy_{lambda_str}.pt"
        out_path   = REPORTS_DIR / f"gated_fusion_entropy_{lambda_ent}.json"

        # Fresh model for each experiment.
        model = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim).to(device)

        hist, best_info = train_one_experiment(
            model, train_loader, val_loader, device, lambda_ent, ckpt_path
        )

        # Load best checkpoint for evaluation.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

        results = evaluate(model, val_data)

        # Print summary.
        print(f"\n── λ={lambda_ent} Evaluation ──────────────────────────────────")
        print(f"  ROC-AUC    : {results['roc_auc']:.4f}")
        print(f"  PR-AUC     : {results['pr_auc']:.4f}")
        print(f"  Brier Score: {results['brier_score']:.4f}")
        a = results["alpha_stats"]
        print(f"  α  mean={a['mean']:.4f}  std={a['std']:.4f}  "
              f"min={a['min']:.4f}  max={a['max']:.4f}")
        print(f"  α < 0.5: {results['alpha_lt_0_5']}   α ≥ 0.5: {results['alpha_ge_0_5']}")
        for pid, info in sorted(results["borderline_cases"].items()):
            label_str = "PR+" if info["label"] == 1 else "PR−"
            print(f"  {pid}  α={info['alpha']:.4f}  "
                  f"P(PR+)={info['prob_pr_pos']:.3f}  {label_str}")

        output = {
            "model":      "GatedFusion-v2-entropy",
            "lambda_ent": lambda_ent,
            "training": {
                "epochs":           EPOCHS,
                "lr":               LR,
                "weight_decay":     WEIGHT_DECAY,
                "batch_size":       BATCH_SIZE,
                "random_state":     42,
                "n_train_patients": len(train_data),
                "n_val_patients":   len(val_data),
                "best_epoch":       int(best_info["epoch"]),
                "best_val_auc":     float(best_info["val_auc"]),
                "checkpoint":       str(ckpt_path),
                "history":          hist,
            },
            "evaluation": results,
        }

        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  Results saved → {out_path.name}")

    print("\n══ All experiments complete ═════════════════════════════════════\n")


if __name__ == "__main__":
    main()
