"""
Attention-weight analysis over the stratified validation cohort.

Loads the best model checkpoint, runs inference on the exact same val split used
during training (random_state=42, test_size=0.2), and for each patient extracts:
  - The masked attention weight vector  (N_real_patches,)
  - Shannon entropy of the distribution  (low = sharp focus, high = diffuse)
  - Top-10 highest-weighted patch indices

Outputs
-------
reports/figures/attention_distributions.png
  Panel A : Sorted attention curves for the 5 highest-confidence true-positive
            patients — shows how peaked or diffuse the model's focus is.
  Panel B : Mean attention variance (±1 SD) for PR+ vs PR− patients.

Usage
-----
    python src/utils/attention_analysis.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.data.dataset        import build_dataloader, load_and_qc_patients
from src.models.fusion_model import PathoGenomicFusionModel


# ── Config / paths ────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model:      PathoGenomicFusionModel,
    val_data:   list[dict],
    device:     torch.device,
) -> list[dict]:
    """
    Run the model over val_data one patient at a time (batch_size=1) so that
    attn_weights are never padded across patients.

    Returns a list of result dicts with keys:
        patient_id, label, prob, attn (1-D np.ndarray, real patches only),
        entropy, top10_indices
    """
    loader = build_dataloader(val_data, batch_size=1, shuffle=False, cache_rate=1.0)
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            patch_emb  = batch["patch_embeddings"].to(device)   # (1, N_pad, 768)
            genomic    = batch["genomic_counts"].to(device)     # (1, G)
            patch_mask = batch["patch_mask"].to(device)         # (1, N_pad)

            logits, attn_weights = model(patch_emb, genomic, patch_mask)
            prob = torch.sigmoid(logits).item()

            # attn_weights: (1, 1, N_pad) — squeeze to 1-D, keep real patches only
            attn_full  = attn_weights[0, 0].cpu().numpy()       # (N_pad,)
            n_real     = int(patch_mask[0].sum().item())
            attn       = attn_full[:n_real]                     # (N_real,)

            # Re-normalise over real patches (softmax may have leaked mass to pads)
            attn = attn / (attn.sum() + 1e-12)

            entropy     = float(-np.sum(attn * np.log(attn + 1e-12)))
            top10       = np.argsort(attn)[::-1][:10].tolist()

            pid   = batch["patient_id"][0]
            label = int(batch["label"].item())

            results.append({
                "patient_id":    pid,
                "label":         label,
                "prob":          prob,
                "attn":          attn,
                "entropy":       entropy,
                "top10_indices": top10,
            })

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_attention_distributions(results: list[dict], output_path: Path) -> None:
    pos = [r for r in results if r["label"] == 1]
    neg = [r for r in results if r["label"] == 0]

    # Pick the 5 highest-confidence true-positive patients for Panel A
    tp_confident = sorted(pos, key=lambda r: r["prob"], reverse=True)[:5]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: sorted attention curves ─────────────────────────────────────
    cmap   = plt.get_cmap("Blues")
    colors = [cmap(0.45 + 0.11 * i) for i in range(len(tp_confident))]

    for i, r in enumerate(tp_confident):
        sorted_attn = np.sort(r["attn"])[::-1]
        x = np.linspace(0, 100, len(sorted_attn))
        ax_a.plot(
            x, sorted_attn,
            color=colors[i], linewidth=1.8,
            label=f"{r['patient_id']}  (p={r['prob']:.2f}, H={r['entropy']:.2f})",
        )

    ax_a.set_title("Panel A — Sorted Attention Coefficients\n(Top-5 Confident PR+ Patients)",
                   fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Patch rank (percentile)")
    ax_a.set_ylabel("Attention weight")
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.grid(True, linestyle="--", alpha=0.4)
    ax_a.set_xlim(0, 100)

    # ── Panel B: mean attention variance by class ─────────────────────────────
    def _variances(group: list[dict]) -> np.ndarray:
        return np.array([np.var(r["attn"]) for r in group])

    var_pos = _variances(pos)
    var_neg = _variances(neg)

    means  = [var_pos.mean(), var_neg.mean()]
    stds   = [var_pos.std(),  var_neg.std()]
    labels = [f"PR+  (n={len(pos)})", f"PR−  (n={len(neg)})"]
    colors_b = ["#16A34A", "#DC2626"]

    bars = ax_b.bar(labels, means, yerr=stds, capsize=6,
                    color=colors_b, alpha=0.82, edgecolor="white", linewidth=1.2,
                    error_kw={"elinewidth": 1.5, "ecolor": "black"})

    for bar, mean in zip(bars, means):
        ax_b.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.05,
                  f"{mean:.2e}", ha="center", va="bottom", fontsize=9)

    ax_b.set_title("Panel B — Mean Attention Variance by Class\n(higher = broader patch focus)",
                   fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Attention variance (mean ± 1 SD)")
    ax_b.set_ylim(0, max(means) + max(stds) * 2.2)
    ax_b.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("PathoGenomic Fusion — Validation Attention Analysis",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {output_path}")


# ── Console report ────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    pos = [r for r in results if r["label"] == 1]
    neg = [r for r in results if r["label"] == 0]

    print(f"\n── Attention analysis: {len(results)} val patients ──────────────────────")
    print(f"  PR+ (label=1) : {len(pos)}   PR− (label=0) : {len(neg)}")

    print(f"\n  {'Patient':<14} {'Label':>5}  {'Prob':>6}  {'Entropy':>8}  {'N patches':>10}  Top-10 indices")
    print(f"  {'─'*14} {'─'*5}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*30}")
    for r in sorted(results, key=lambda x: (-x["label"], -x["prob"]))[:20]:
        top = str(r["top10_indices"])
        print(f"  {r['patient_id']:<14} {r['label']:>5}  {r['prob']:>6.3f}  "
              f"{r['entropy']:>8.3f}  {len(r['attn']):>10,}  {top}")

    print(f"\n  Mean entropy  —  PR+: {np.mean([r['entropy'] for r in pos]):.3f}"
          f"   PR−: {np.mean([r['entropy'] for r in neg]):.3f}")
    print(f"  Mean variance —  PR+: {np.mean([np.var(r['attn']) for r in pos]):.2e}"
          f"   PR−: {np.mean([np.var(r['attn']) for r in neg]):.2e}")
    print(f"{'─'*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config        = _load_config(root / "configs" / "model_config.yaml")
    counts_path   = root / "data" / "raw" / "counts.csv"
    emb_dir       = root / "data" / "processed" / "image_embeddings"
    clinical_path = root / "data" / "raw" / "clinical_metadata.csv"
    ckpt_path     = root / "checkpoints" / "best_model.pt"
    output_path   = root / "reports" / "figures" / "attention_distributions.png"

    print("\nLoading QC patient list …")
    patient_data = load_and_qc_patients(counts_path, emb_dir, clinical_path)

    # Reproduce the exact same 80/20 stratified split used during training
    labels_for_split = [d["label"] for d in patient_data]
    _, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=labels_for_split
    )
    print(f"  Val cohort : {len(val_data)} patients  "
          f"(pos={sum(d['label']==1 for d in val_data)}, "
          f"neg={sum(d['label']==0 for d in val_data)})")

    # Load model
    genomic_dim = pd.read_csv(counts_path, index_col="patient_id", nrows=0).shape[1]
    # Force CPU: MPS has a Metal kernel bug with need_weights=True at batch_size=1
    # (single query token hits an MPSNDArray slice assertion). CPU is fast enough
    # for 191-patient inference.
    device      = torch.device("cpu")
    model       = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"  Checkpoint : epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")
    print(f"  Device     : {device}\n")

    print("Running inference …")
    results = run_inference(model, val_data, device)

    print_report(results)
    plot_attention_distributions(results, output_path)
    print("Attention analysis complete.\n")


if __name__ == "__main__":
    main()
