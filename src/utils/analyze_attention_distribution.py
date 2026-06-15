"""
Formal statistical analysis of attention-entropy distributions across the
stratified PR+ / PR− validation cohort.

Workflow
--------
1. Reproduce the exact val split (random_state=42, test_size=0.2).
2. Run per-patient inference to collect masked attention vectors.
3. Compute top_1_percent_mass per patient: sum of the top 1% highest-weighted
   patches / total attention mass — a slide-size-invariant focus metric.
4. Two-sample Welch t-test on top_1_percent_mass (PR+ vs PR−).
   Also retains the original entropy t-test for comparison.
5. Report the 3 sharpest PR+ patients (lowest entropy)
   and the 3 most diffuse PR− patients (highest entropy / variance)
   as TCGA barcodes + exact metrics — ready for GDC portal verification.

Usage
-----
    python src/utils/analyze_attention_distribution.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.data.dataset        import load_and_qc_patients
from src.models.fusion_model import PathoGenomicFusionModel
from src.utils.attention_analysis import run_inference


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _print_rule(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_extremes_table(
    label_str:  str,
    patients:   list[dict],
    metric_key: str,
    ascending:  bool,
) -> None:
    direction = "lowest" if ascending else "highest"
    print(f"\n  {label_str} — {direction} {metric_key}:")
    print(f"  {'Barcode':<14}  {'Label':>5}  {'Prob':>6}  "
          f"{'Entropy':>8}  {'Variance':>12}  {'N patches':>10}  Top-10 patch indices")
    print("  " + "─" * 90)
    for r in patients:
        var = float(np.var(r["attn"]))
        top = r["top10_indices"]
        print(
            f"  {r['patient_id']:<14}  {r['label']:>5}  {r['prob']:>6.3f}  "
            f"{r['entropy']:>8.3f}  {var:>12.3e}  {len(r['attn']):>10,}  {top}"
        )


def main() -> None:
    config        = _load_config(root / "configs" / "model_config.yaml")
    counts_path   = root / "data" / "raw" / "counts.csv"
    emb_dir       = root / "data" / "processed" / "image_embeddings"
    clinical_path = root / "data" / "raw" / "clinical_metadata.csv"
    ckpt_path     = root / "checkpoints" / "best_model.pt"

    # ── Reproduce exact val split ─────────────────────────────────────────────
    print("\nLoading QC patient list …")
    patient_data = load_and_qc_patients(counts_path, emb_dir, clinical_path)
    labels_for_split = [d["label"] for d in patient_data]
    _, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split
    )
    print(f"  Val cohort : {len(val_data)} patients  "
          f"(pos={sum(d['label']==1 for d in val_data)}, "
          f"neg={sum(d['label']==0 for d in val_data)})")

    # ── Load model checkpoint ─────────────────────────────────────────────────
    genomic_dim = pd.read_csv(counts_path, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("cpu")   # MPS has a Metal bug with need_weights=True + batch_size=1
    model       = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)
    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Checkpoint : epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running per-patient inference …")
    results = run_inference(model, val_data, device)

    pos = [r for r in results if r["label"] == 1]
    neg = [r for r in results if r["label"] == 0]

    # ── Compute top_1_percent_mass (slide-size-invariant focus metric) ────────
    def _top1mass(r: dict) -> float:
        a = r["attn"]
        n_top = max(1, int(len(a) * 0.01))
        sorted_a = np.sort(a)[::-1]
        return float(sorted_a[:n_top].sum() / (sorted_a.sum() + 1e-12))

    top1mass_pos = np.array([_top1mass(r) for r in pos])
    top1mass_neg = np.array([_top1mass(r) for r in neg])

    entropy_pos = np.array([r["entropy"] for r in pos])
    entropy_neg = np.array([r["entropy"] for r in neg])

    # ── Welch two-sample t-test on top_1_percent_mass ─────────────────────────
    t_stat, p_value = stats.ttest_ind(top1mass_pos, top1mass_neg, equal_var=False)

    _print_rule("═")
    print("  TOP-1%-MASS — SLIDE-SIZE-INVARIANT ATTENTION FOCUS ANALYSIS")
    _print_rule("═")

    print(f"\n  Metric: top_1_percent_mass = sum(top 1% patches) / sum(all patches)")
    print(f"\n  Group          N    Mean top-1%-mass    Std top-1%-mass")
    print(f"  {'─'*58}")
    print(f"  PR+  (label=1) {len(top1mass_pos):>4}   {top1mass_pos.mean():>16.4f}    {top1mass_pos.std():>15.4f}")
    print(f"  PR−  (label=0) {len(top1mass_neg):>4}   {top1mass_neg.mean():>16.4f}    {top1mass_neg.std():>15.4f}")

    print(f"\n  Welch two-sample t-test  (H₀: equal means)")
    print(f"  t-statistic : {t_stat:>10.4f}")
    print(f"  p-value     : {p_value:>10.4e}", end="")
    if p_value < 0.001:
        print("  *** (p < 0.001 — highly significant)")
    elif p_value < 0.01:
        print("  **  (p < 0.01  — significant)")
    elif p_value < 0.05:
        print("  *   (p < 0.05  — significant)")
    else:
        print("  ns  (p ≥ 0.05  — not significant)")

    direction = "PR+ patients concentrate attention MORE densely" if top1mass_pos.mean() > top1mass_neg.mean() \
                else "PR− patients concentrate attention MORE densely"
    print(f"\n  Interpretation: {direction}  (higher top-1%-mass = sharper focus)")

    # ── Entropy t-test retained for comparison ────────────────────────────────
    t_ent, p_ent = stats.ttest_ind(entropy_pos, entropy_neg, equal_var=False)
    print(f"\n  [Reference] Entropy Welch t-test: t={t_ent:.4f}  p={p_ent:.4e}"
          f"  (confounded by slide size)")

    # ── Extreme patient extraction ────────────────────────────────────────────
    _print_rule()
    print("  EXTREME PATIENT EXTRACTION — TARGET LIST FOR PORTAL VERIFICATION")
    _print_rule()

    # Top 3 PR+ by lowest entropy (sharpest, most focused)
    top3_pos_sharp = sorted(pos, key=lambda r: r["entropy"])[:3]

    # Top 3 PR− by highest entropy (most diffuse, most heterogeneous)
    top3_neg_diffuse = sorted(neg, key=lambda r: r["entropy"], reverse=True)[:3]

    print_extremes_table(
        "Top 3 PR+  [ sharpest attention — lowest entropy ]",
        top3_pos_sharp,
        "entropy", ascending=True,
    )
    print_extremes_table(
        "Top 3 PR−  [ most diffuse attention — highest entropy ]",
        top3_neg_diffuse,
        "entropy", ascending=False,
    )

    _print_rule()
    print("\n  GDC Portal verification targets (12-char TCGA barcodes):\n")
    print("    Sharpest PR+")
    for r in top3_pos_sharp:
        print(f"      {r['patient_id']}  entropy={r['entropy']:.4f}  prob={r['prob']:.4f}")
    print("    Most diffuse PR−")
    for r in top3_neg_diffuse:
        print(f"      {r['patient_id']}  entropy={r['entropy']:.4f}  prob={r['prob']:.4f}")
    _print_rule()
    print()


if __name__ == "__main__":
    main()
