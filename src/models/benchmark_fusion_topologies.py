"""
Benchmark: Cross-Attention Fusion vs. Late Fusion Baseline
===========================================================
Cohort   : TCGA-BRCA, N=951 (638 PR+, 313 PR−)
Split    : stratified 80/20, random_state=42
Training : 10 epochs, lr=1e-4, weight_decay=1e-4, batch_size=4
Metrics  : Val AUC · Macro Precision · Macro Recall · BCE loss trajectory
Focus    : Top-1%-Mass Concentration (cross-attention attention weights)
           vs. Gradient×Input Top-1%-Mass (late fusion genomic attribution)

Usage
-----
    python src/models/benchmark_fusion_topologies.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from src.data.dataset        import build_dataloader, load_and_qc_patients
from src.models.fusion_model import PathoGenomicFusionModel

warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 10
BATCH_SIZE   = 4
LR           = 1e-4
WD           = 1e-4
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Late Fusion Model
# ─────────────────────────────────────────────────────────────────────────────

class MeanMaxPool(nn.Module):
    """
    Permutation-invariant pooling over variable-length patch sequences.

    For each slide: compute masked mean and masked max independently across
    all real patches, concatenate the two 768-D vectors, then project to
    out_dim. Padding positions are excluded from both operations.
    """
    def __init__(self, patch_dim: int = 768, out_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(patch_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        patch_emb:  torch.Tensor,   # (B, N, 768)
        patch_mask: torch.Tensor,   # (B, N)  True = real patch
    ) -> torch.Tensor:              # (B, out_dim)
        mask_f = patch_mask.float().unsqueeze(-1)              # (B, N, 1)

        # Masked mean  — divide by actual patch count per slide
        mean_emb = (patch_emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)

        # Masked max  — fill pad positions with −∞ before reduction
        max_emb = patch_emb.masked_fill(
            ~patch_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values

        return self.proj(torch.cat([mean_emb, max_emb], dim=-1))


class LateFusionModel(nn.Module):
    """
    Two independent encoding arms fused by concatenation.

    Genomic arm  : 20 513 → 512 → 256  (linear + ReLU + dropout)
    Pathology arm: (N, 768) → MeanMaxPool → 256
    Fusion head  : concat 512 → linear → 1 logit
    """
    def __init__(self, genomic_input_dim: int = 20513) -> None:
        super().__init__()
        self.genomic_arm = nn.Sequential(
            nn.Linear(genomic_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.pathology_arm = MeanMaxPool(patch_dim=768, out_dim=256)
        self.fusion_head   = nn.Linear(512, 1)

    def forward(
        self,
        patch_emb:      torch.Tensor,   # (B, N, 768)
        genomic_counts: torch.Tensor,   # (B, G)
        patch_mask:     torch.Tensor,   # (B, N)  True = real
    ) -> torch.Tensor:                  # (B, 1)
        gen  = self.genomic_arm(genomic_counts)
        path = self.pathology_arm(patch_emb, patch_mask)
        return self.fusion_head(torch.cat([gen, path], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Shared training / evaluation primitives
# ─────────────────────────────────────────────────────────────────────────────

def _print_rule(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _forward(
    model,
    batch:          dict,
    device:         torch.device,
    is_cross_attn:  bool,
):
    """Unified forward; returns (logits (B,1), attn_or_None)."""
    pe  = batch["patch_embeddings"].to(device)
    gc  = batch["genomic_counts"].to(device)
    pm  = batch["patch_mask"].to(device)
    if is_cross_attn:
        logits, attn = model(pe, gc, pm)
        return logits, attn
    return model(pe, gc, pm), None


def train_epoch(
    model, loader, optimizer, criterion, device, is_cross_attn: bool
) -> float:
    model.train()
    running = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits, _ = _forward(model, batch, device, is_cross_attn)
        labels = batch["label"].float().to(device)
        loss = criterion(logits.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def val_epoch(
    model, loader, criterion, device, is_cross_attn: bool
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    running = 0.0

    with torch.no_grad():
        for batch in loader:
            logits, _ = _forward(model, batch, device, is_cross_attn)
            labels = batch["label"].float().to(device)
            running += criterion(logits.squeeze(1), labels).item()
            all_probs.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return {
        "auc":       roc_auc_score(labels, probs),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "val_loss":  running / len(loader),
        "probs":     probs,
        "labels":    labels,
    }


def train_model(
    model,
    train_data: list,
    val_data:   list,
    device:     torch.device,
    is_cross_attn: bool,
    label:      str,
) -> dict:
    """
    Full training run; returns best-checkpoint metrics plus full history.
    """
    train_loader = build_dataloader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,  cache_rate=1.0
    )
    val_loader = build_dataloader(
        val_data,   batch_size=BATCH_SIZE, shuffle=False, cache_rate=1.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    history    = {"train_loss": [], "val_auc": [], "val_loss": []}
    best_auc   = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_val_m : dict = {}

    print(f"\n  [{label}]  training for {EPOCHS} epochs …")
    _print_rule("─", 60)
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val AUC':>8}  "
          f"{'Precision':>9}  {'Recall':>7}  {'Val Loss':>9}")
    _print_rule("─", 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, is_cross_attn)
        val_m      = val_epoch(model, val_loader, criterion, device, is_cross_attn)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_m["auc"])
        history["val_loss"].append(val_m["val_loss"])

        marker = ""
        if val_m["auc"] > best_auc:
            best_auc   = val_m["auc"]
            best_val_m = val_m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker     = " *"

        print(f"  {epoch:>5}  {train_loss:>11.4f}  {val_m['auc']:>8.4f}  "
              f"{val_m['precision']:>9.4f}  {val_m['recall']:>7.4f}  "
              f"{val_m['val_loss']:>9.4f}{marker}")

    _print_rule("─", 60)
    model.load_state_dict(best_state)

    best_val_m["history"]    = history
    best_val_m["best_epoch"] = history["val_auc"].index(best_auc) + 1
    return best_val_m


# ─────────────────────────────────────────────────────────────────────────────
# Convergence metric
# ─────────────────────────────────────────────────────────────────────────────

def convergence_epoch(train_losses: list) -> int:
    """
    First epoch where training loss drops to ≤ 50% of the epoch-1 loss.
    Proxy for 'the model has left the high-loss regime'.
    Falls back to the epoch of minimum loss if threshold is never crossed.
    """
    threshold = train_losses[0] * 0.5
    for i, loss in enumerate(train_losses, start=1):
        if loss <= threshold:
            return i
    return int(np.argmin(train_losses)) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_top1mass_attn(
    model:    PathoGenomicFusionModel,
    val_data: list,
) -> list:
    """
    Per-patient Top-1%-Mass Concentration from raw cross-attention weights.

    Runs batch_size=1 inference on CPU to avoid the MPS Metal assertion
    triggered by need_weights=True with a single query token.
    """
    cpu = torch.device("cpu")
    model = model.to(cpu).eval()
    loader = build_dataloader(val_data, batch_size=1, shuffle=False, cache_rate=1.0)
    results = []

    with torch.no_grad():
        for batch in loader:
            pe  = batch["patch_embeddings"].to(cpu)
            gc  = batch["genomic_counts"].to(cpu)
            pm  = batch["patch_mask"].to(cpu)

            logits, attn = model(pe, gc, pm)     # attn: (1, 1, N_pad)
            prob = torch.sigmoid(logits).item()

            attn_full = attn[0, 0].cpu().numpy()
            n_real    = int(pm[0].sum().item())
            attn_real = attn_full[:n_real]
            attn_real = attn_real / (attn_real.sum() + 1e-12)

            k     = max(1, int(n_real * 0.01))
            top1m = float(np.sort(attn_real)[::-1][:k].sum())

            results.append({
                "patient_id": batch["patient_id"][0],
                "label":      int(batch["label"].item()),
                "prob":       prob,
                "top1mass":   top1m,
            })

    return results


def extract_top1mass_gxi(
    model:    LateFusionModel,
    val_data: list,
) -> list:
    """
    Per-patient Top-1%-Mass Concentration via Gradient × Input attribution.

    Computes ∂P(PR+)/∂x_genomic · x_genomic for each patient, then applies
    the same top-1% mass fraction formula used for the attention metric.
    Run on CPU for gradient stability.
    """
    cpu = torch.device("cpu")
    model = model.to(cpu).eval()
    loader = build_dataloader(val_data, batch_size=1, shuffle=False, cache_rate=1.0)
    results = []

    for batch in loader:
        pe  = batch["patch_embeddings"].to(cpu).detach()
        pm  = batch["patch_mask"].to(cpu).detach()

        # Leaf genomic tensor with gradient tracking
        gc = batch["genomic_counts"].to(cpu).detach().clone().requires_grad_(True)

        model.zero_grad()
        logit = model(pe, gc, pm)
        prob  = torch.sigmoid(logit.squeeze())
        prob.backward()

        gxi = (gc.grad.detach() * gc.detach()).abs().squeeze(0).cpu().numpy()
        k   = max(1, int(gxi.shape[0] * 0.01))     # top 1% of 20,513 genes = 205 genes
        sorted_gxi = np.sort(gxi)[::-1]
        top1m = float(sorted_gxi[:k].sum() / (sorted_gxi.sum() + 1e-12))

        results.append({
            "patient_id": batch["patient_id"][0],
            "label":      int(batch["label"].item()),
            "prob":       prob.item(),
            "top1mass":   top1m,
        })

    return results


def _focus_stats(results: list) -> dict:
    """Group-level Top-1%-Mass means, SEMs, and Welch t-test."""
    pos = [r["top1mass"] for r in results if r["label"] == 1]
    neg = [r["top1mass"] for r in results if r["label"] == 0]
    arr_pos, arr_neg = np.array(pos), np.array(neg)
    t, p = stats.ttest_ind(arr_pos, arr_neg, equal_var=False)
    return {
        "pos_mean": arr_pos.mean(), "pos_sem": arr_pos.std() / np.sqrt(len(arr_pos)),
        "neg_mean": arr_neg.mean(), "neg_sem": arr_neg.std() / np.sqrt(len(arr_neg)),
        "t": t, "p": p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    ca_metrics: dict,
    lf_metrics: dict,
    ca_focus:   dict,
    lf_focus:   dict,
    ca_conv:    int,
    lf_conv:    int,
    n_params_ca: int,
    n_params_lf: int,
) -> None:

    def _sig(p: float) -> str:
        if p < 0.001: return "p < 0.001 ***"
        if p < 0.01:  return "p < 0.01  **"
        if p < 0.05:  return f"p = {p:.3f} *"
        return         f"p = {p:.3f} ns"

    W = 74
    _print_rule("═", W)
    print("  ARCHITECTURE COMPARISON — TCGA-BRCA VALIDATION COHORT  (N = 951)")
    _print_rule("═", W)
    header = f"  {'Metric':<42} {'Cross-Attn':>12}  {'Late Fusion':>12}"
    print(header)
    _print_rule("─", W)

    rows = [
        ("Parameters",
         f"{n_params_ca:>12,}", f"{n_params_lf:>12,}"),

        ("── Performance ──────────────────────────", "", ""),
        ("Best Val AUC",
         f"{ca_metrics['auc']:>12.4f}", f"{lf_metrics['auc']:>12.4f}"),
        ("Macro Precision (threshold 0.5)",
         f"{ca_metrics['precision']:>12.4f}", f"{lf_metrics['precision']:>12.4f}"),
        ("Macro Recall    (threshold 0.5)",
         f"{ca_metrics['recall']:>12.4f}", f"{lf_metrics['recall']:>12.4f}"),
        ("Best Checkpoint Epoch",
         f"{ca_metrics['best_epoch']:>12}", f"{lf_metrics['best_epoch']:>12}"),

        ("── Convergence ──────────────────────────", "", ""),
        ("Epoch loss ≤ 50% of initial loss",
         f"{ca_conv:>12}", f"{lf_conv:>12}"),
        ("Initial train loss (epoch 1)",
         f"{ca_metrics['history']['train_loss'][0]:>12.4f}",
         f"{lf_metrics['history']['train_loss'][0]:>12.4f}"),
        ("Final train loss  (epoch 10)",
         f"{ca_metrics['history']['train_loss'][-1]:>12.4f}",
         f"{lf_metrics['history']['train_loss'][-1]:>12.4f}"),

        ("── Focus Metric (Top-1%-Mass) ───────────", "", ""),
        ("PR+  mean (SEM)",
         f"{ca_focus['pos_mean']:.4f} ({ca_focus['pos_sem']:.4f})",
         f"{lf_focus['pos_mean']:.4f} ({lf_focus['pos_sem']:.4f})"),
        ("PR−  mean (SEM)",
         f"{ca_focus['neg_mean']:.4f} ({ca_focus['neg_sem']:.4f})",
         f"{lf_focus['neg_mean']:.4f} ({lf_focus['neg_sem']:.4f})"),
        ("Welch t-statistic",
         f"{ca_focus['t']:>12.3f}", f"{lf_focus['t']:>12.3f}"),
        ("Welch t-test",
         f"{_sig(ca_focus['p']):>12}", f"{_sig(lf_focus['p']):>12}"),
        ("Focus source",
         f"{'Attn weights':>12}", f"{'Grad × Input':>12}"),
    ]

    for row in rows:
        label, ca_val, lf_val = row
        if label.startswith("──"):
            _print_rule("─", W)
            print(f"  {label}")
            _print_rule("─", W)
        else:
            print(f"  {label:<42} {ca_val:>12}  {lf_val:>12}")

    _print_rule("═", W)

    winner = "Cross-Attention" if ca_metrics["auc"] > lf_metrics["auc"] else "Late Fusion"
    delta  = abs(ca_metrics["auc"] - lf_metrics["auc"])
    print(f"\n  Verdict: {winner} achieves superior AUC  (ΔAUC = {delta:.4f})")

    ca_sig = ca_focus["p"] < 0.05
    lf_sig = lf_focus["p"] < 0.05
    if ca_sig and not lf_sig:
        print("  Focus signal: Cross-Attention shows significant phenotypic focus "
              "differentiation;\n"
              "                Late Fusion genomic attribution does not.")
    elif lf_sig and not ca_sig:
        print("  Focus signal: Late Fusion attribution shows significant phenotypic "
              "focus differentiation;\n"
              "                Cross-Attention attention does not.")
    elif ca_sig and lf_sig:
        print("  Focus signal: Both architectures show significant phenotypic "
              "focus differentiation.")
    else:
        print("  Focus signal: Neither architecture produces a significant "
              "phenotypic focus difference.")
    _print_rule("═", W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config_path   = root / "configs" / "model_config.yaml"
    counts_path   = root / "data" / "raw" / "counts.csv"
    emb_dir       = root / "data" / "processed" / "image_embeddings"
    clinical_path = root / "data" / "raw" / "clinical_metadata.csv"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Shared data split ─────────────────────────────────────────────────────
    print("\nLoading and QC-filtering patient data …")
    patient_data = load_and_qc_patients(counts_path, emb_dir, clinical_path)
    genomic_dim  = pd.read_csv(counts_path, index_col="patient_id", nrows=0).shape[1]

    labels       = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"\n  Split  → train={len(train_data)} · val={len(val_data)} "
          f"(pos={sum(d['label']==1 for d in val_data)}, "
          f"neg={sum(d['label']==0 for d in val_data)})")
    print(f"  Genomic features : {genomic_dim:,}")

    # Device for training (MPS if available, else CPU)
    if torch.backends.mps.is_available():
        train_device = torch.device("mps")
    elif torch.cuda.is_available():
        train_device = torch.device("cuda")
    else:
        train_device = torch.device("cpu")
    print(f"  Training device  : {train_device}\n")

    # ── Model A: Cross-Attention ──────────────────────────────────────────────
    _print_rule("═")
    print("  MODEL A — Asymmetric Multi-Head Cross-Attention Fusion")
    _print_rule("═")
    ca_model = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)
    ca_model.to(train_device)
    n_params_ca = sum(p.numel() for p in ca_model.parameters())
    print(f"  Parameters : {n_params_ca:,}")

    ca_metrics = train_model(
        ca_model, train_data, val_data,
        train_device, is_cross_attn=True, label="Cross-Attention"
    )

    # ── Model B: Late Fusion ──────────────────────────────────────────────────
    _print_rule("═")
    print("  MODEL B — Late Fusion Baseline  (MeanMax Pooling + Concatenation)")
    _print_rule("═")
    lf_model = LateFusionModel(genomic_input_dim=genomic_dim)
    lf_model.to(train_device)
    n_params_lf = sum(p.numel() for p in lf_model.parameters())
    print(f"  Parameters : {n_params_lf:,}")

    lf_metrics = train_model(
        lf_model, train_data, val_data,
        train_device, is_cross_attn=False, label="Late Fusion"
    )

    # ── Convergence epochs ────────────────────────────────────────────────────
    ca_conv = convergence_epoch(ca_metrics["history"]["train_loss"])
    lf_conv = convergence_epoch(lf_metrics["history"]["train_loss"])

    # ── Interpretability extraction ───────────────────────────────────────────
    _print_rule("═")
    print("  INTERPRETABILITY EXTRACTION  (CPU, batch_size=1)")
    _print_rule("═")

    print("\n  Cross-Attention → Top-1%-Mass from attention weight vectors …")
    ca_focus_results = extract_top1mass_attn(ca_model, val_data)
    ca_focus = _focus_stats(ca_focus_results)
    print(f"  Done.  PR+ mean={ca_focus['pos_mean']:.4f}  "
          f"PR− mean={ca_focus['neg_mean']:.4f}  "
          f"t={ca_focus['t']:.3f}  p={ca_focus['p']:.4f}")

    print("\n  Late Fusion → Top-1%-Mass from Gradient × Input attribution …")
    lf_focus_results = extract_top1mass_gxi(lf_model, val_data)
    lf_focus = _focus_stats(lf_focus_results)
    print(f"  Done.  PR+ mean={lf_focus['pos_mean']:.4f}  "
          f"PR− mean={lf_focus['neg_mean']:.4f}  "
          f"t={lf_focus['t']:.3f}  p={lf_focus['p']:.4f}")

    # ── Final comparison table ────────────────────────────────────────────────
    _print_rule("═")
    print("  LOSS TRAJECTORY SUMMARY")
    _print_rule("═")
    print(f"\n  {'Epoch':>5}  {'CA Train':>10}  {'CA Val AUC':>10}  "
          f"{'LF Train':>10}  {'LF Val AUC':>10}")
    _print_rule("─", 56)
    for ep in range(EPOCHS):
        print(f"  {ep+1:>5}  "
              f"{ca_metrics['history']['train_loss'][ep]:>10.4f}  "
              f"{ca_metrics['history']['val_auc'][ep]:>10.4f}  "
              f"{lf_metrics['history']['train_loss'][ep]:>10.4f}  "
              f"{lf_metrics['history']['val_auc'][ep]:>10.4f}")
    _print_rule("─", 56)

    print_comparison_table(
        ca_metrics, lf_metrics,
        ca_focus, lf_focus,
        ca_conv, lf_conv,
        n_params_ca, n_params_lf,
    )


if __name__ == "__main__":
    main()
