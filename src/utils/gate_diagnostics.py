"""
gate_diagnostics.py
====================
Per-patient diagnostics for the Gated-Fusion module on the held-out
validation cohort.

Loads a checkpoint (default: checkpoints/best_model.pt), re-creates the exact
train/val split used during training (stratified 80/20, random_state=42),
runs per-patient inference on the 191-patient validation cohort, and outputs:

    <csv>
    ─────
    patient_id, alpha, prob_pr_pos, true_label, cosine_sim_v_g

    <fig>
    ─────
    Alpha distribution split by PR+ / PR- status.

cosine_sim_v_g is the cosine similarity between the cross-attention output
(visual stream, v) and the genomic query token (genomic stream, g) — the
same two vectors GatedFusion concatenates to produce alpha. It measures raw
geometric agreement between the two modalities' 512-dim representations,
independent of what the gate itself learned to do with that agreement.

Console output includes alpha summary stats, a Welch t-test on alpha between
PR+ and PR- groups, alpha for the two borderline IHC patients
(TCGA-BH-A0HK, TCGA-BH-A0HW), and cosine-similarity stats by PR status.

Usage
-----
    python src/utils/gate_diagnostics.py \\
        --ckpt checkpoints/best_model.pt \\
        --csv  reports/gate_diagnostics.csv \\
        --fig  reports/figures/gate_alpha_distribution.png
"""

import sys
import csv
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset        import load_and_qc_patients
from src.models.fusion_model import PathoGenomicFusionModel

CONFIG_PATH   = ROOT / "configs" / "model_config.yaml"
COUNTS_PATH   = ROOT / "data"   / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data"   / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data"   / "raw" / "clinical_metadata.csv"

BORDERLINE_PATIENTS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]

# Colorblind-safe (Wong 2011) — matches run_validation_analysis.py
C_POS = "#0077BB"   # blue — PR+
C_NEG = "#CC3311"   # red  — PR-

warnings.filterwarnings("ignore")


def get_val_data() -> list[dict]:
    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    labels       = [d["label"] for d in patient_data]
    _, val_data  = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels,
    )
    return val_data


def run_inference(val_data: list[dict], ckpt_path: Path) -> list[dict]:
    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]
    model       = PathoGenomicFusionModel({"fusion_bottleneck": yaml.safe_load(
        open(CONFIG_PATH))["fusion_bottleneck"]}, genomic_input_dim=genomic_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # strict=False: some checkpoints (e.g. the base model) predate the gate
    # parameters; others (e2e / staged) contain them.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    rows = []
    for patient in val_data:
        pe = torch.tensor(patient["patch_embeddings"], dtype=torch.float32).unsqueeze(0)
        gc = torch.tensor(patient["genomic_counts"],   dtype=torch.float32).unsqueeze(0)
        pm = torch.ones(1, pe.shape[1], dtype=torch.bool)

        with torch.no_grad():
            # Replicate PathoGenomicFusionModel.forward() step by step to
            # recover the intermediate visual (v) and genomic (g) vectors
            # that GatedFusion consumes — the public forward() only returns
            # logits, attn_weights, alpha.
            g = model.genomic_projector(gc)                  # (1, 512) — genomic query token
            query_seq = g.unsqueeze(1)                        # (1, 1, 512)
            attn_out, _ = model.cross_attention(
                query=query_seq, key=pe, value=pe,
                key_padding_mask=~pm, need_weights=True,
            )
            v = attn_out.squeeze(1)                            # (1, 512) — cross-attention output
            fused, alpha = model.gated_fusion(v, g)
            fused  = model.post_attn(fused)
            logit  = model.head(fused)

            cos_sim = F.cosine_similarity(v, g, dim=-1).item()

        rows.append({
            "patient_id":     patient.get("patient_id", f"patient_{len(rows)}"),
            "alpha":          float(alpha.squeeze().item()),
            "prob_pr_pos":    torch.sigmoid(logit).item(),
            "true_label":     int(patient["label"]),
            "cosine_sim_v_g": cos_sim,
        })

    return rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["patient_id", "alpha", "prob_pr_pos", "true_label", "cosine_sim_v_g"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved   → {csv_path}  ({len(rows)} patients)")


def plot_alpha_distribution(rows: list[dict], fig_path: Path, ckpt_name: str) -> None:
    alpha_pos = np.array([r["alpha"] for r in rows if r["true_label"] == 1])
    alpha_neg = np.array([r["alpha"] for r in rows if r["true_label"] == 0])

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        300,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    lo = min(alpha_pos.min(), alpha_neg.min())
    hi = max(alpha_pos.max(), alpha_neg.max())
    bins = np.linspace(lo, hi, 40) if hi > lo else np.linspace(lo - 0.01, hi + 0.01, 40)

    ax.hist(alpha_neg, bins=bins, color=C_NEG, alpha=0.55,
            label=f"PR$-$  (n={len(alpha_neg)})", edgecolor="white", linewidth=0.4)
    ax.hist(alpha_pos, bins=bins, color=C_POS, alpha=0.55,
            label=f"PR$+$  (n={len(alpha_pos)})", edgecolor="white", linewidth=0.4)
    if lo <= 0.5 <= hi:
        ax.axvline(0.5, color="0.3", ls="--", lw=1.2, label="flag threshold (α = 0.5)")

    ax.set_xlabel("Gate value α", fontsize=11)
    ax.set_ylabel("Patient count", fontsize=11)
    ax.set_title(
        "Gate α Distribution by PR Status\n"
        f"checkpoint: {ckpt_name}  (held-out cohort, N={len(rows)})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(True, axis="y", ls="--", lw=0.5, alpha=0.4, zorder=0)
    fig.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")


def print_stats(rows: list[dict], ckpt_name: str) -> None:
    alphas    = np.array([r["alpha"] for r in rows])
    alpha_pos = np.array([r["alpha"] for r in rows if r["true_label"] == 1])
    alpha_neg = np.array([r["alpha"] for r in rows if r["true_label"] == 0])
    cos_pos   = np.array([r["cosine_sim_v_g"] for r in rows if r["true_label"] == 1])
    cos_neg   = np.array([r["cosine_sim_v_g"] for r in rows if r["true_label"] == 0])

    t_stat, p_val = stats.ttest_ind(alpha_pos, alpha_neg, equal_var=False)

    w = 60
    print("\n" + "─" * w)
    print(f"{ckpt_name:^{w}}")
    print("─" * w)
    print(f"  alpha mean/std      : {alphas.mean():.4f} / {alphas.std():.4f}")
    print(f"  alpha min/max       : {alphas.min():.4f} / {alphas.max():.4f}")
    print(f"  alpha < 0.5         : {(alphas < 0.5).sum()} / {len(alphas)}")
    print(f"\n  Welch t-test (alpha, PR+ vs PR-):")
    print(f"    t-statistic       : {t_stat:+.4f}")
    print(f"    p-value           : {p_val:.4e}")

    print(f"\n  Borderline patients:")
    by_pid = {r["patient_id"]: r for r in rows}
    for pid in BORDERLINE_PATIENTS:
        if pid in by_pid:
            r = by_pid[pid]
            print(f"    {pid}: alpha={r['alpha']:.4f}  P(PR+)={r['prob_pr_pos']:.4f}  "
                  f"true_label={r['true_label']}  cos_sim={r['cosine_sim_v_g']:.4f}")
        else:
            print(f"    {pid}: not found in held-out cohort")

    print(f"\n  cosine_sim_v_g mean/std by PR status:")
    print(f"    PR+  : {cos_pos.mean():.4f} / {cos_pos.std():.4f}  (n={len(cos_pos)})")
    print(f"    PR-  : {cos_neg.mean():.4f} / {cos_neg.std():.4f}  (n={len(cos_neg)})")
    print("─" * w)


def main() -> list[dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--csv",  type=Path, default=ROOT / "reports" / "gate_diagnostics.csv")
    parser.add_argument("--fig",  type=Path, default=ROOT / "reports" / "figures" / "gate_alpha_distribution.png")
    args = parser.parse_args()

    print(f"Loading held-out validation cohort (191 patients, random_state=42)...")
    val_data = get_val_data()
    print(f"  {len(val_data)} patients loaded\n")

    print(f"Running inference with checkpoint: {args.ckpt}")
    rows = run_inference(val_data, args.ckpt)

    print("\nWriting outputs...")
    write_csv(rows, args.csv)
    plot_alpha_distribution(rows, args.fig, args.ckpt.name)

    print_stats(rows, args.ckpt.name)
    return rows


if __name__ == "__main__":
    main()
