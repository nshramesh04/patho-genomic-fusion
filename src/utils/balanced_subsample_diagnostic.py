"""
balanced_subsample_diagnostic.py
==================================
Checks whether the alpha / cosine-similarity findings on
reports/gate_diagnostics_staged.csv (best_model_v2_staged.pt) hold up under
class balance, given the held-out cohort's 128 PR+ / 63 PR- (2:1) imbalance.

Procedure
----------
  1. Split the 191-patient cohort into PR+ / PR- by true_label.
  2. Randomly subsample PR+ down to len(PR-) patients (random_state=42).
  3. Compute alpha mean/std/min/max, Welch t-test (alpha, PR+ vs PR-), and
     cosine_sim_v_g mean/std by PR status — on both the full cohort and the
     balanced subsample — for direct comparison.
  4. Print alpha for TCGA-BH-A0HK / TCGA-BH-A0HW from this checkpoint.

Outputs
-------
  reports/balanced_subsample_ids.csv — patient_ids in the balanced subsample

Usage
-----
    python src/utils/balanced_subsample_diagnostic.py
"""

import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = ROOT / "reports" / "gate_diagnostics_staged.csv"
DEFAULT_OUT_PATH = ROOT / "reports" / "balanced_subsample_ids.csv"

BORDERLINE_IDS = ["TCGA-BH-A0HK", "TCGA-BH-A0HW"]
RANDOM_STATE = 42


def compute_metrics(df: pd.DataFrame) -> dict:
    pos = df[df["true_label"] == 1]
    neg = df[df["true_label"] == 0]

    t_stat, p_val = stats.ttest_ind(pos["alpha"], neg["alpha"], equal_var=False)

    return {
        "n_pos":          len(pos),
        "n_neg":          len(neg),
        "alpha_mean":     df["alpha"].mean(),
        "alpha_std":      df["alpha"].std(),
        "alpha_min":      df["alpha"].min(),
        "alpha_max":      df["alpha"].max(),
        "welch_t":        t_stat,
        "welch_p":        p_val,
        "cos_pos_mean":   pos["cosine_sim_v_g"].mean(),
        "cos_pos_std":    pos["cosine_sim_v_g"].std(),
        "cos_neg_mean":   neg["cosine_sim_v_g"].mean(),
        "cos_neg_std":    neg["cosine_sim_v_g"].std(),
    }


def run(csv_path: Path, out_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path.name}: {len(df)} patients\n")

    pos_df = df[df["true_label"] == 1]
    neg_df = df[df["true_label"] == 0]
    print(f"Full cohort: PR+ n={len(pos_df)}  PR- n={len(neg_df)}")

    # ── Balanced subsample: downsample PR+ to match PR- count ─────────────────
    rng = np.random.RandomState(RANDOM_STATE)
    pos_sub = pos_df.sample(n=len(neg_df), random_state=rng)
    balanced_df = pd.concat([pos_sub, neg_df]).sort_index()
    print(f"Balanced subsample: PR+ n={len(pos_sub)}  PR- n={len(neg_df)}  "
          f"(random_state={RANDOM_STATE})\n")

    full_m = compute_metrics(df)
    bal_m  = compute_metrics(balanced_df)

    # ── Side-by-side table ─────────────────────────────────────────────────────
    rows = [
        ("alpha mean",        "alpha_mean",   "{:.4f}"),
        ("alpha std",         "alpha_std",    "{:.4f}"),
        ("alpha min",         "alpha_min",    "{:.4f}"),
        ("alpha max",         "alpha_max",    "{:.4f}"),
        ("Welch t",           "welch_t",      "{:+.4f}"),
        ("Welch p",           "welch_p",      "{:.4e}"),
        ("cos(v,g) PR+ mean", "cos_pos_mean", "{:.4f}"),
        ("cos(v,g) PR- mean", "cos_neg_mean", "{:.4f}"),
    ]

    print(f"  {'Metric':<20} {'Full cohort':>15} {'Balanced subsample':>20}")
    print("  " + "-" * 57)
    for label, key, fmt in rows:
        full_val = fmt.format(full_m[key])
        bal_val  = fmt.format(bal_m[key])
        print(f"  {label:<20} {full_val:>15} {bal_val:>20}")

    print(f"\n  (Full cohort: n_pos={full_m['n_pos']} n_neg={full_m['n_neg']}   "
          f"Balanced: n_pos={bal_m['n_pos']} n_neg={bal_m['n_neg']})")

    # ── Borderline patients ─────────────────────────────────────────────────────
    print(f"\n  Borderline patients ({csv_path.name}):")
    for pid in BORDERLINE_IDS:
        row = df[df["patient_id"] == pid]
        if len(row):
            r = row.iloc[0]
            print(f"    {pid}: alpha={r['alpha']:.4f}  P(PR+)={r['prob_pr_pos']:.4f}  "
                  f"true_label={int(r['true_label'])}  cos_sim={r['cosine_sim_v_g']:.4f}")
        else:
            print(f"    {pid}: not found")

    # ── Save subsample IDs ────────────────────────────────────────────────────
    # The subsample depends only on true_label (identical across checkpoints,
    # since patient order/labels come from the same fixed 80/20 split), so this
    # is idempotent across CSVs — safe to (re)write each time.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "true_label"])
        for _, r in balanced_df.sort_values("patient_id").iterrows():
            writer.writerow([r["patient_id"], int(r["true_label"])])
    print(f"\n  Balanced subsample IDs saved → {out_path}  ({len(balanced_df)} patients)")

    return {"full": full_m, "balanced": bal_m}


def verdict(full_p: float, bal_p: float) -> str:
    if full_p < 0.05 and bal_p >= 0.05:
        return "fragile"
    if full_p < 0.05 and bal_p < 0.05:
        return "robust"
    return "not significant"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, action="append", default=None,
                         help="CSV path(s); repeatable. Defaults to staged/e2e/staged_var.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    csv_paths = args.csv or [
        ROOT / "reports" / "gate_diagnostics_staged.csv",
        ROOT / "reports" / "gate_diagnostics_e2e.csv",
        ROOT / "reports" / "gate_diagnostics_staged_var.csv",
    ]

    results = {}
    for csv_path in csv_paths:
        print("═" * 60)
        print(f"  {csv_path.stem}")
        print("═" * 60)
        results[csv_path.stem] = run(csv_path, args.out)
        print()

    print("═" * 60)
    print("  Combined summary")
    print("═" * 60)
    print(f"  {'Checkpoint':<20} {'Full p':>12} {'Balanced p':>12}   {'Verdict':<12}")
    print("  " + "-" * 60)
    for name, m in results.items():
        full_p = m["full"]["welch_p"]
        bal_p  = m["balanced"]["welch_p"]
        print(f"  {name:<20} {full_p:>12.4e} {bal_p:>12.4e}   {verdict(full_p, bal_p):<12}")


if __name__ == "__main__":
    main()
