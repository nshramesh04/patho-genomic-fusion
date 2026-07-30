"""
generate_roc_pr_figure.py
==========================
Reproducibly generates reports/figures/roc_pr_curves.png from the
held-out cohort predictions in
reports/gate_diagnostics_anchored_calibrated.csv.

Usage
-----
    python src/utils/generate_roc_pr_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_CSV = REPO_ROOT / "reports" / "gate_diagnostics_anchored_calibrated.csv"
OUTPUT_PATH = REPO_ROOT / "reports" / "figures" / "roc_pr_curves.png"

C_POS = "#0077BB"

EXPECTED_ROC_AUC = 0.8482
EXPECTED_PR_AUC = 0.9054
TOLERANCE = 0.001


def generate_roc_pr_figure():
    df = pd.read_csv(PREDICTIONS_CSV)
    labels = df["true_label"].values
    probs = df["prob_pr_pos"].values

    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    no_skill = labels.mean()

    print(f"ROC-AUC: {roc_auc:.4f} (expected {EXPECTED_ROC_AUC} ± {TOLERANCE})")
    print(f"PR-AUC:  {pr_auc:.4f} (expected {EXPECTED_PR_AUC} ± {TOLERANCE})")
    assert abs(roc_auc - EXPECTED_ROC_AUC) <= TOLERANCE, (
        f"ROC-AUC {roc_auc:.4f} outside expected {EXPECTED_ROC_AUC} ± {TOLERANCE}"
    )
    assert abs(pr_auc - EXPECTED_PR_AUC) <= TOLERANCE, (
        f"PR-AUC {pr_auc:.4f} outside expected {EXPECTED_PR_AUC} ± {TOLERANCE}"
    )

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # ── ROC ──────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    ax_roc.plot(fpr, tpr, color=C_POS, ls="-", lw=2.2,
                label=f"Cross-Attention + Gate  (AUC = {roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="0.65", ls=":", lw=1.1,
                label="Random classifier  (AUC = 0.50)")
    ax_roc.fill_between([0, 1], [0, 1], alpha=0.03, color="0.5")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=9, loc="lower right")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(True, ls="--", lw=0.5, alpha=0.4)
    ax_roc.tick_params(labelsize=9)
    ax_roc.set_facecolor("white")

    # ── Precision-Recall ─────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(labels, probs)
    ax_pr.plot(rec, prec, color=C_POS, ls="-", lw=2.2,
               label=f"Cross-Attention + Gate  (PR-AUC = {pr_auc:.4f})")
    ax_pr.axhline(no_skill, color="0.65", ls=":", lw=1.1,
                  label=f"No-skill baseline  ({no_skill:.3f})")
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.legend(fontsize=9, loc="lower left")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, ls="--", lw=0.5, alpha=0.4)
    ax_pr.tick_params(labelsize=9)
    ax_pr.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, facecolor="white")
    plt.close(fig)

    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_roc_pr_figure()
