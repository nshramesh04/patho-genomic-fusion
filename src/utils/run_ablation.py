"""
run_ablation.py
================
Trains all five fusion-architecture conditions on the identical stratified
80/20 split (random_state=42), hyperparameters, and random seed, then
reports held-out performance for each:

    crossattention  PathoGenomicFusionModel — genomic query, cross-attention
    earlyFusion     EarlyFusionBaseline     — concat genomic onto every patch
    randomQuery     RandomQueryBaseline     — cross-attention, fixed random query
    abmil           ABMIL (Ilse et al. 2018) — pathology-only, no genomic input
    latefusion      LateFusionModel          — independent arms, concat fusion

Note on "held-out test set": this repo's existing scripts (trainer.py,
benchmark_fusion_topologies.py, run_mil_baselines.py) all evaluate on the
80/20 stratified validation split rather than a separate third split — there
is no pre-defined train/val/test split anywhere in the codebase. To keep
these numbers comparable to those existing results, this script reuses that
same convention: the 20% split is the held-out evaluation set referenced
below.

Note on "same batch size": abmil's existing implementation
(run_mil_baselines.py) trains one bag (patient) at a time — it has no
batched forward path. That is inherent to the existing baseline and is left
unchanged here; every other hyperparameter (lr, weight_decay, epochs, split,
seed) is identical across all five conditions.

Results saved to reports/ablation_results.csv.

Usage
-----
    python src/utils/run_ablation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloader, load_and_qc_patients
from src.trainer import FUSION_TYPE, build_model, forward_logits, load_config
from src.utils.run_mil_baselines import ABMIL, _train_epoch_abmil

warnings.filterwarnings("ignore", category=UserWarning, module="monai")

COUNTS_PATH   = ROOT / "data" / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data" / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data" / "raw" / "clinical_metadata.csv"
RESULTS_PATH  = ROOT / "reports" / "ablation_results.csv"

BATCHED_MODELS = ["crossattention", "earlyFusion", "randomQuery", "latefusion"]
ALL_MODELS     = BATCHED_MODELS + ["abmil"]

SEED       = 42
BATCH_SIZE = 4
WD         = 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# DeLong (1988) AUC confidence interval — Sun & Xu (2014) O(N log N) algorithm
# ══════════════════════════════════════════════════════════════════════════════

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    # np.cov collapses to a 0-d scalar when k == 1 (our case: one prediction
    # vector) instead of a (1, 1) matrix — force 2D so delongcov[0, 0] works.
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.95) -> tuple[float, float, float]:
    """DeLong AUC point estimate + (alpha) CI. Returns (auc, ci_lower, ci_upper)."""
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    order          = np.argsort(-y_true, kind="mergesort")   # positives (label=1) first
    y_true_sorted  = y_true[order]
    y_score_sorted = y_score[order]
    label_1_count  = int(y_true_sorted.sum())

    predictions_sorted_transposed = y_score_sorted.reshape(1, -1)
    aucs, delongcov = _fast_delong(predictions_sorted_transposed, label_1_count)

    auc = float(aucs[0])
    se  = float(np.sqrt(delongcov[0, 0]))
    z   = norm.ppf(1 - (1 - alpha) / 2)
    ci_lower = max(0.0, auc - z * se)
    ci_upper = min(1.0, auc + z * se)
    return auc, ci_lower, ci_upper


def _metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict:
    auc, ci_lo, ci_hi = delong_roc_ci(labels, probs)
    return {
        "roc_auc":          auc,
        "roc_auc_ci_lower": ci_lo,
        "roc_auc_ci_upper": ci_hi,
        "pr_auc":           float(average_precision_score(labels, probs)),
        "brier_score":      float(brier_score_loss(labels, probs)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Batched conditions (crossattention, earlyFusion, randomQuery, latefusion)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_eval_batched(
    model_name: str,
    config: dict,
    genomic_dim: int,
    train_data: list,
    holdout_data: list,
    device: torch.device,
) -> dict:
    torch.manual_seed(SEED)
    model = build_model(model_name, config, genomic_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader   = build_dataloader(train_data,   batch_size=BATCH_SIZE, shuffle=True)
    holdout_loader = build_dataloader(holdout_data, batch_size=BATCH_SIZE, shuffle=False)

    lr        = config["training"]["learning_rate"]
    epochs    = config["training"]["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n── Training {model_name}  (params={n_params:,}) ─────────────────────")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels     = batch["label"].float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits, _ = forward_logits(model_name, model, patch_emb, genomic, patch_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch:>2}/{epochs}  train_loss={total_loss / len(train_loader):.4f}")

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in holdout_loader:
            patch_emb  = batch["patch_embeddings"].to(device)
            genomic    = batch["genomic_counts"].to(device)
            patch_mask = batch["patch_mask"].to(device)

            logits, _ = forward_logits(model_name, model, patch_emb, genomic, patch_mask)
            probs.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
            labels.extend(batch["label"].float().cpu().tolist())

    metrics = _metrics_from_probs(np.array(labels), np.array(probs))
    print(f"  → ROC-AUC={metrics['roc_auc']:.4f} "
          f"[{metrics['roc_auc_ci_lower']:.4f}, {metrics['roc_auc_ci_upper']:.4f}]  "
          f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier_score']:.4f}")
    return {"model": model_name, **metrics}


# ══════════════════════════════════════════════════════════════════════════════
# abmil — existing per-patient training loop (run_mil_baselines.py)
# ══════════════════════════════════════════════════════════════════════════════

def train_and_eval_abmil(config: dict, train_data: list, holdout_data: list) -> dict:
    torch.manual_seed(SEED)
    model = ABMIL()
    n_params = sum(p.numel() for p in model.parameters())

    lr        = config["training"]["learning_rate"]
    epochs    = config["training"]["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)

    print(f"\n── Training abmil  (params={n_params:,}) "
          f"[per-patient loop — no batching, per existing implementation] ───")
    for epoch in range(1, epochs + 1):
        loss = _train_epoch_abmil(model, train_data, optimizer)
        print(f"  Epoch {epoch:>2}/{epochs}  train_loss={loss:.4f}")

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for pt in holdout_data:
            h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
            logit, _ = model(h)
            probs.append(torch.sigmoid(logit).item())
            labels.append(float(pt["label"]))

    metrics = _metrics_from_probs(np.array(labels), np.array(probs))
    print(f"  → ROC-AUC={metrics['roc_auc']:.4f} "
          f"[{metrics['roc_auc_ci_lower']:.4f}, {metrics['roc_auc_ci_upper']:.4f}]  "
          f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier_score']:.4f}")
    return {"model": "abmil", **metrics}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("══ Fusion-Mechanism Ablation — 5 Conditions ════════════════════════\n")

    config      = load_config(ROOT / "configs" / "model_config.yaml")
    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    labels_for_split = [d["label"] for d in patient_data]
    train_data, holdout_data = train_test_split(
        patient_data, test_size=0.2, random_state=SEED, stratify=labels_for_split,
    )
    print(f"  Train   : {len(train_data)} patients")
    print(f"  Holdout : {len(holdout_data)} patients  "
          f"(pos={sum(d['label'] == 1 for d in holdout_data)}, "
          f"neg={sum(d['label'] == 0 for d in holdout_data)})")
    print(f"  lr={config['training']['learning_rate']}  wd={WD}  "
          f"batch_size={BATCH_SIZE}  epochs={config['training']['epochs']}  seed={SEED}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device  : {device}")

    results = []
    for model_name in BATCHED_MODELS:
        results.append(
            train_and_eval_batched(model_name, config, genomic_dim, train_data, holdout_data, device)
        )
    results.append(train_and_eval_abmil(config, train_data, holdout_data))

    df = pd.DataFrame(results, columns=[
        "model", "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper", "pr_auc", "brier_score",
    ])
    # Descriptive fusion-mechanism label per condition (see FUSION_TYPE in trainer.py):
    #   crossattention → query_guided_early   earlyFusion → naive_early
    #   randomQuery    → query_guided_early_no_content
    #   latefusion     → late                 abmil       → unimodal
    df.insert(1, "fusion_type", df["model"].map(FUSION_TYPE))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Results saved → {RESULTS_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 118
    print(f"\n{'─' * w}")
    print(f"{'Ablation Results — Held-out Split (random_state=42)':^{w}}")
    print(f"{'─' * w}")
    print(f"  {'Model':<16} {'Fusion Type':<32} {'ROC-AUC':>10}  {'95% CI (DeLong)':>20}  "
          f"{'PR-AUC':>10}  {'Brier':>10}")
    print(f"  {'─'*14:<16} {'─'*30:<32} {'─'*8:>10}  {'─'*18:>20}  {'─'*8:>10}  {'─'*8:>10}")
    for r in results:
        ci = f"[{r['roc_auc_ci_lower']:.4f}, {r['roc_auc_ci_upper']:.4f}]"
        print(f"  {r['model']:<16} {FUSION_TYPE[r['model']]:<32} {r['roc_auc']:>10.4f}  {ci:>20}  "
              f"{r['pr_auc']:>10.4f}  {r['brier_score']:>10.4f}")
    print(f"{'─' * w}\n")


if __name__ == "__main__":
    main()
