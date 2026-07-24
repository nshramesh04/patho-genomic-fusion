"""
run_mil_baselines.py
====================
ABMIL (Ilse et al. 2018) and CLAM-SB (Lu et al. 2021) MIL baselines.

Uses the same patient-level labels and data loading pipeline as the main model,
and the identical 80/20 stratified split (random_state=42):
    760 patients for training, 191 for held-out evaluation.

Input: CHIEF ViT-B patch embeddings (dim=768), one bag per patient.
Output: binary classification (PR+ / PR-).

Results saved to reports/baseline_results.json.

Usage
-----
    python src/utils/run_mil_baselines.py
"""

import sys
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_and_qc_patients

COUNTS_PATH   = ROOT / "data" / "raw" / "counts.csv"
EMB_DIR       = ROOT / "data" / "processed" / "image_embeddings"
CLINICAL_PATH = ROOT / "data" / "raw" / "clinical_metadata.csv"
RESULTS_PATH  = ROOT / "reports" / "baseline_results.json"

EMBED_DIM = 768
HIDDEN    = 512
ATT_DIM   = 128
LR        = 1e-4
WD        = 1e-4
EPOCHS    = 20
SEED      = 42
K_SAMPLE  = 8       # CLAM-SB: top-k / bottom-k instance pseudo-labels
INST_W    = 0.3     # CLAM-SB: instance loss weight


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ABMIL — Ilse et al. 2018
# ══════════════════════════════════════════════════════════════════════════════

class ABMIL(nn.Module):
    """
    Gated attention MIL.

    Architecture
    ────────────
    h_i  →  encoder  →  h̃_i  (N × HIDDEN)
    a_i  =  w^T (tanh(V h̃_i) ⊙ σ(U h̃_i))      gated attention score
    z    =  Σ softmax(a) · h̃                     attention-pooled bag embedding
    ŷ    =  classifier(z)
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.att_V = nn.Linear(HIDDEN, ATT_DIM)
        self.att_U = nn.Linear(HIDDEN, ATT_DIM)
        self.att_w = nn.Linear(ATT_DIM, 1)
        self.classifier = nn.Linear(HIDDEN, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(h)                                      # (N, HIDDEN)
        a = self.att_w(torch.tanh(self.att_V(h)) *
                       torch.sigmoid(self.att_U(h)))             # (N, 1)
        a = F.softmax(a, dim=0)                                  # (N, 1)
        z = (a * h).sum(dim=0)                                   # (HIDDEN,)
        return self.classifier(z).squeeze(), a.squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLAM-SB — Lu et al. 2021
# ══════════════════════════════════════════════════════════════════════════════

class CLAM_SB(nn.Module):
    """
    Single-branch CLAM with instance-level pseudo-labelling.

    Architecture
    ────────────
    Same encoder and gated attention as ABMIL.
    During training, top-K and bottom-K attended patches receive
    pseudo-labels (matching / opposing the bag label) that drive
    an auxiliary instance cross-entropy loss:

        L = L_bag  +  inst_weight * L_instance
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.att_V = nn.Linear(HIDDEN, ATT_DIM)
        self.att_U = nn.Linear(HIDDEN, ATT_DIM)
        self.att_w = nn.Linear(ATT_DIM, 1)
        self.instance_clf = nn.Linear(HIDDEN, 2)   # pseudo-label head
        self.bag_clf      = nn.Linear(HIDDEN, 1)

    def _attention(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_raw = self.att_w(torch.tanh(self.att_V(h)) *
                           torch.sigmoid(self.att_U(h)))          # (N, 1)
        return F.softmax(a_raw, dim=0), a_raw                     # soft, raw

    def forward(
        self,
        h: torch.Tensor,
        label: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h       = self.encoder(h)                                  # (N, HIDDEN)
        a_soft, a_raw = self._attention(h)                         # (N,1), (N,1)
        z       = (a_soft * h).sum(dim=0)                          # (HIDDEN,)
        bag_logit = self.bag_clf(z).squeeze()                      # scalar

        inst_loss = torch.tensor(0.0, device=h.device)
        if label is not None and self.training:
            k         = min(K_SAMPLE, h.shape[0])
            top_idx   = a_raw.squeeze().topk(k).indices
            bot_idx   = (-a_raw.squeeze()).topk(k).indices
            y_top     = torch.full((k,), int(label),     dtype=torch.long, device=h.device)
            y_bot     = torch.full((k,), 1 - int(label), dtype=torch.long, device=h.device)
            inst_loss = (
                F.cross_entropy(self.instance_clf(h[top_idx]), y_top) +
                F.cross_entropy(self.instance_clf(h[bot_idx]), y_bot)
            ) / 2

        return bag_logit, a_soft.squeeze(), inst_loss


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Training and evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_abmil(model: ABMIL,
                       data: list[dict],
                       optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total = 0.0
    for pt in data:
        h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
        y = torch.tensor(pt["label"],            dtype=torch.float32)
        optimizer.zero_grad()
        logit, _ = model(h)
        F.binary_cross_entropy_with_logits(logit, y).backward()
        optimizer.step()
        total += F.binary_cross_entropy_with_logits(logit.detach(), y).item()
    return total / len(data)


def _train_epoch_clam(model: CLAM_SB,
                      data: list[dict],
                      optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total = 0.0
    for pt in data:
        h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
        y = torch.tensor(pt["label"],            dtype=torch.float32)
        optimizer.zero_grad()
        logit, _, inst_loss = model(h, label=pt["label"])
        loss = F.binary_cross_entropy_with_logits(logit, y) + INST_W * inst_loss
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(data)


@torch.no_grad()
def evaluate(model: nn.Module,
             data: list[dict],
             is_clam: bool = False) -> dict:
    model.eval()
    probs, labels = [], []
    for pt in data:
        h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
        if is_clam:
            logit, _, _ = model(h)
        else:
            logit, _ = model(h)
        probs.append(torch.sigmoid(logit).item())
        labels.append(float(pt["label"]))
    probs  = np.array(probs)
    labels = np.array(labels)
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc":  float(average_precision_score(labels, probs)),
        "brier":   float(brier_score_loss(labels, probs)),
    }


def run_abmil(train_data: list[dict], val_data: list[dict]) -> dict:
    torch.manual_seed(SEED)
    model     = ABMIL()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    print("\n── Training ABMIL ───────────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        loss = _train_epoch_abmil(model, train_data, optimizer)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2}/{EPOCHS}  train_loss={loss:.4f}")

    metrics = evaluate(model, val_data, is_clam=False)
    print(f"  → ROC-AUC={metrics['roc_auc']:.4f}  "
          f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}")
    return metrics


def run_clam_sb(train_data: list[dict], val_data: list[dict]) -> dict:
    torch.manual_seed(SEED)
    model     = CLAM_SB()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    print("\n── Training CLAM-SB ─────────────────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        loss = _train_epoch_clam(model, train_data, optimizer)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2}/{EPOCHS}  train_loss={loss:.4f}")

    metrics = evaluate(model, val_data, is_clam=True)
    print(f"  → ROC-AUC={metrics['roc_auc']:.4f}  "
          f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("══ MIL Baselines — ABMIL & CLAM-SB ════════════════════════════\n")

    patient_data = load_and_qc_patients(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    all_labels   = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data, test_size=0.2, random_state=42, stratify=all_labels,
    )

    pos_tr = int(sum(d["label"] == 1 for d in train_data))
    neg_tr = int(sum(d["label"] == 0 for d in train_data))
    pos_va = int(sum(d["label"] == 1 for d in val_data))
    neg_va = int(sum(d["label"] == 0 for d in val_data))
    print(f"  Train : {len(train_data)} patients  (PR+ = {pos_tr}, PR− = {neg_tr})")
    print(f"  Val   : {len(val_data)} patients  (PR+ = {pos_va}, PR− = {neg_va})")
    print(f"  Epochs: {EPOCHS}  |  lr={LR}  |  wd={WD}")

    abmil_m  = run_abmil(train_data, val_data)
    clam_m   = run_clam_sb(train_data, val_data)

    results = {
        "split": {
            "train_n":      len(train_data),
            "val_n":        len(val_data),
            "random_state": 42,
        },
        "hyperparameters": {
            "epochs":       EPOCHS,
            "lr":           LR,
            "weight_decay": WD,
            "embed_dim":    EMBED_DIM,
            "hidden_dim":   HIDDEN,
            "att_dim":      ATT_DIM,
            "clam_k":       K_SAMPLE,
            "clam_inst_w":  INST_W,
        },
        "ABMIL":   abmil_m,
        "CLAM_SB": clam_m,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 68
    print(f"\n{'─' * w}")
    print(f"{'MIL Baseline Results — Held-out Cohort (N = 191)':^{w}}")
    print(f"{'─' * w}")
    print(f"  {'Model':<14} {'ROC-AUC':>10}  {'PR-AUC':>10}  {'Brier Score':>12}")
    print(f"  {'─'*12:<14} {'─'*8:>10}  {'─'*8:>10}  {'─'*10:>12}")
    print(f"  {'ABMIL':<14} {abmil_m['roc_auc']:>10.4f}  "
          f"{abmil_m['pr_auc']:>10.4f}  {abmil_m['brier']:>12.4f}")
    print(f"  {'CLAM-SB':<14} {clam_m['roc_auc']:>10.4f}  "
          f"{clam_m['pr_auc']:>10.4f}  {clam_m['brier']:>12.4f}")
    print(f"{'─' * w}\n")


if __name__ == "__main__":
    main()
