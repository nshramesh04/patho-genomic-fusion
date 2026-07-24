"""
gate_alpha_comparison.py
=========================
Side-by-side comparison of the alpha distribution (split by PR status)
between the end-to-end and staged Gated-Fusion training regimes.

Reads the two CSVs produced by gate_diagnostics.py and saves a two-panel
figure to reports/figures/gate_alpha_comparison.png.

Usage
-----
    python src/utils/gate_alpha_comparison.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

E2E_CSV    = ROOT / "reports" / "gate_diagnostics_e2e.csv"
STAGED_CSV = ROOT / "reports" / "gate_diagnostics_staged.csv"
FIG_PATH   = ROOT / "reports" / "figures" / "gate_alpha_comparison.png"

C_POS = "#0077BB"   # blue — PR+
C_NEG = "#CC3311"   # red  — PR-


def load_alphas(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    pos = np.array([float(r["alpha"]) for r in rows if int(r["true_label"]) == 1])
    neg = np.array([float(r["alpha"]) for r in rows if int(r["true_label"]) == 0])
    return pos, neg


def plot_panel(ax, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    lo = min(pos.min(), neg.min())
    hi = max(pos.max(), neg.max())
    pad = max((hi - lo) * 0.1, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 30)

    ax.hist(neg, bins=bins, color=C_NEG, alpha=0.55,
            label=f"PR$-$  (n={len(neg)})", edgecolor="white", linewidth=0.4)
    ax.hist(pos, bins=bins, color=C_POS, alpha=0.55,
            label=f"PR$+$  (n={len(pos)})", edgecolor="white", linewidth=0.4)
    if lo - pad <= 0.5 <= hi + pad:
        ax.axvline(0.5, color="0.3", ls="--", lw=1.2, label="flag threshold (α = 0.5)")

    ax.set_xlabel("Gate value α", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4, zorder=0)


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        300,
    })

    e2e_pos, e2e_neg       = load_alphas(E2E_CSV)
    staged_pos, staged_neg = load_alphas(STAGED_CSV)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_panel(ax1, e2e_pos, e2e_neg, "End-to-End Retrained\n(best_model_v2_e2e.pt)")
    plot_panel(ax2, staged_pos, staged_neg, "Staged Retrained\n(best_model_v2_staged.pt)")
    ax1.set_ylabel("Patient count", fontsize=11)

    fig.suptitle(
        "Gate α Distribution by PR Status — Training Regime Comparison\n"
        "Held-out cohort, N=191",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {FIG_PATH}")
