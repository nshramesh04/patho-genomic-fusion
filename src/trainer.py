import sys
import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from monai.metrics import ROCAUCMetric
from sklearn.model_selection import train_test_split

# ── Model conditions ────────────────────────────────────────────────────────
# All five share identical training hyperparameters, data split, and seed
# (see run_ablation.py). "abmil" does not share the batched
# (patch_embeddings, genomic_counts, patch_mask) forward signature used by
# the other four — see build_model()'s docstring — so Trainer below only
# supports the other four; abmil is trained via its own existing per-patient
# loop in src/utils/run_mil_baselines.py.
#
# Descriptive fusion_type labels for these five conditions (used in
# run_ablation.py's ablation_results.csv) live in FUSION_TYPE below.
MODEL_CHOICES = ["crossattention", "earlyFusion", "randomQuery", "abmil", "latefusion"]

# model (CLI/dispatch key) → fusion_type (descriptive label for reporting)
FUSION_TYPE = {
    "crossattention": "query_guided_early",
    "earlyFusion":     "naive_early",
    "randomQuery":     "query_guided_early_no_content",
    "latefusion":      "late",
    "abmil":           "unimodal",
}


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(model_name: str, config: dict, genomic_input_dim: int) -> nn.Module:
    """
    Factory for the five ablation conditions. "abmil" is pathology-only
    (no genomic input) and uses a per-patient — not batched — forward
    signature; it is returned here for completeness but is not compatible
    with Trainer / forward_logits below. Use run_mil_baselines.py's
    existing training loop for it instead.
    """
    from src.models.fusion_model import (
        PathoGenomicFusionModel,
        EarlyFusionBaseline,
        RandomQueryBaseline,
    )

    if model_name == "crossattention":
        return PathoGenomicFusionModel(config, genomic_input_dim=genomic_input_dim)
    if model_name == "earlyFusion":
        return EarlyFusionBaseline(config, genomic_input_dim=genomic_input_dim)
    if model_name == "randomQuery":
        return RandomQueryBaseline(config, genomic_input_dim=genomic_input_dim)
    if model_name == "latefusion":
        from src.models.benchmark_fusion_topologies import LateFusionModel
        return LateFusionModel(genomic_input_dim=genomic_input_dim)
    if model_name == "abmil":
        from src.utils.run_mil_baselines import ABMIL
        return ABMIL()
    raise ValueError(f"Unknown model: {model_name!r}  (choices: {MODEL_CHOICES})")


def forward_logits(
    model_type: str,
    model: nn.Module,
    patch_emb: torch.Tensor,
    genomic: torch.Tensor,
    patch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Unified forward across the batched conditions (crossattention,
    earlyFusion, randomQuery, latefusion). Returns (logits, attn_weights)
    — attn_weights is None for architectures that don't produce one.
    """
    if model_type == "crossattention":
        logits, attn_weights, _ = model(patch_emb, genomic, patch_mask)
        return logits, attn_weights
    if model_type in ("earlyFusion", "randomQuery", "latefusion"):
        return model(patch_emb, genomic, patch_mask), None
    raise ValueError(
        f"forward_logits does not support model_type={model_type!r}; "
        "abmil uses its own per-patient loop in run_mil_baselines.py"
    )


def plot_training_curves(
    losses: list[float],
    aucs:   list[float],
    output_path: Path,
) -> None:
    epochs = range(1, len(losses) + 1)
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, losses, marker="o", linewidth=2, color="#2563EB")
    ax_loss.set_title("Training Loss (BCE)", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.set_xticks(list(epochs))
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    ax_auc.plot(epochs, aucs, marker="o", linewidth=2, color="#16A34A")
    ax_auc.set_title("Stratified Validation AUC", fontsize=13, fontweight="bold")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_xticks(list(epochs))
    ax_auc.set_ylim(0.0, 1.05)
    ax_auc.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax_auc.legend(fontsize=9)
    ax_auc.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("PathoGenomic Fusion — 10-Epoch Training Run", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {output_path}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: Path,
        device: torch.device,
        model_type: str = "crossattention",
    ) -> None:
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.model_type     = model_type
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        lr = config["training"]["learning_rate"]
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion  = nn.BCEWithLogitsLoss()
        self.auc_metric = ROCAUCMetric()
        self.best_auc   = -1.0

    # ── Single training epoch ─────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            patch_emb  = batch["patch_embeddings"].to(self.device)
            genomic    = batch["genomic_counts"].to(self.device)
            patch_mask = batch["patch_mask"].to(self.device)
            labels     = batch["label"].float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            logits, _ = forward_logits(self.model_type, self.model, patch_emb, genomic, patch_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(self.train_loader)
        print(f"  [Epoch {epoch}] train_loss={mean_loss:.4f}")
        return mean_loss

    # ── Validation epoch ──────────────────────────────────────────────────────
    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        self.auc_metric.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                patch_emb  = batch["patch_embeddings"].to(self.device)
                genomic    = batch["genomic_counts"].to(self.device)
                patch_mask = batch["patch_mask"].to(self.device)
                labels     = batch["label"].float().unsqueeze(1).to(self.device)

                logits, attn_weights = forward_logits(self.model_type, self.model, patch_emb, genomic, patch_mask)
                probs  = torch.sigmoid(logits)
                self.auc_metric(y_pred=probs, y=labels)

        if attn_weights is not None:
            print(f"  attn_weights shape: {tuple(attn_weights.shape)}")  # (B, 1, N_patches)

        auc = self.auc_metric.aggregate()
        # aggregate() returns a tensor in MONAI >=1.0 — extract scalar safely
        auc_val = auc.item() if isinstance(auc, torch.Tensor) else float(auc)
        print(f"  [Epoch {epoch}] val_auc={auc_val:.4f}")
        return auc_val

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def _maybe_checkpoint(self, epoch: int, auc: float) -> None:
        if auc > self.best_auc:
            self.best_auc = auc
            ckpt_path = self.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "val_auc":     auc,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved  (val_auc={auc:.4f} -> {ckpt_path})")

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(self, epochs: int) -> tuple[list[float], list[float]]:
        history_loss: list[float] = []
        history_auc:  list[float] = []

        print(f"\nTraining for {epochs} epoch(s) on {self.device}\n{'─'*52}")
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            auc  = self.val_epoch(epoch)
            self._maybe_checkpoint(epoch, auc)
            history_loss.append(loss)
            history_auc.append(auc)

        print(f"{'─'*52}\nTraining complete. Best val_auc={self.best_auc:.4f}")
        return history_loss, history_auc


# ── Execution block ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.dataset import build_dataloader, load_and_qc_patients

    parser = argparse.ArgumentParser(description="Train a single fusion-architecture condition.")
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="crossattention",
        help="Fusion condition to train (default: crossattention). "
             "'abmil' is not supported here — it uses its own per-patient "
             "training loop; run src/utils/run_mil_baselines.py or "
             "src/utils/run_ablation.py instead.",
    )
    args = parser.parse_args()

    if args.model == "abmil":
        parser.error(
            "--model abmil is not supported by trainer.py (different, unbatched "
            "forward signature — no genomic input). Use "
            "src/utils/run_mil_baselines.py directly, or src/utils/run_ablation.py "
            "to run all five conditions together."
        )

    root          = Path(__file__).resolve().parents[1]
    config        = load_config(root / "configs" / "model_config.yaml")
    counts_path   = root / "data" / "raw" / "counts.csv"
    emb_dir       = root / "data" / "processed" / "image_embeddings"
    clinical_path = root / "data" / "raw" / "clinical_metadata.csv"

    # ── QC: strict 3-way intersection, drop mismatched patients ──────────────
    patient_data = load_and_qc_patients(counts_path, emb_dir, clinical_path)

    pos = sum(d["label"] == 1.0 for d in patient_data)
    neg = sum(d["label"] == 0.0 for d in patient_data)
    print(f"Patients after QC: {len(patient_data)}  (pos={pos}, neg={neg})\n")

    # ── Stratified 80/20 train / val split ───────────────────────────────────
    labels_for_split = [d["label"] for d in patient_data]
    train_data, val_data = train_test_split(
        patient_data,
        test_size=0.2,
        random_state=42,
        stratify=labels_for_split,
    )

    train_loader = build_dataloader(train_data, batch_size=4, shuffle=True)
    val_loader   = build_dataloader(val_data,   batch_size=4, shuffle=False)

    # Confirm dynamic padding across splits
    sample_batch = next(iter(train_loader))
    print(f"Train batch shapes:")
    print(f"  patch_embeddings : {tuple(sample_batch['patch_embeddings'].shape)}")
    print(f"  patch_mask       : {tuple(sample_batch['patch_mask'].shape)}")
    print(f"  genomic_counts   : {tuple(sample_batch['genomic_counts'].shape)}")
    print(f"  valid patches    : {sample_batch['patch_mask'].sum(dim=1).tolist()}\n")

    # ── Build model ───────────────────────────────────────────────────────────
    import pandas as pd
    genomic_dim = pd.read_csv(counts_path, index_col="patient_id", nrows=0).shape[1]
    device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)
    model       = build_model(args.model, config, genomic_dim)

    print(f"Model  : {args.model}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Device : {device}\n")

    # ── 10-epoch training run ─────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        config         = config,
        checkpoint_dir = root / "checkpoints",
        device         = device,
        model_type     = args.model,
    )
    history_loss, history_auc = trainer.fit(epochs=config["training"]["epochs"])

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(
        losses      = history_loss,
        aucs        = history_auc,
        output_path = root / "reports" / "figures" / f"training_curves_{args.model}.png",
    )

    # ── Verify checkpoint was written ─────────────────────────────────────────
    ckpt_path = root / "checkpoints" / "best_model.pt"
    assert ckpt_path.exists(), "Checkpoint file not found!"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Checkpoint verified : epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

    # ── Per-epoch summary table ───────────────────────────────────────────────
    print(f"\n{'─'*54}")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Δ Loss':>8}  {'Val AUC':>9}")
    print(f"{'─'*54}")
    for i, (loss, auc) in enumerate(zip(history_loss, history_auc), 1):
        delta = f"{history_loss[i-2] - loss:+.4f}" if i > 1 else "    —   "
        print(f"  {i:>5}  {loss:>11.4f}  {delta:>8}  {auc:>9.4f}")
    print(f"{'─'*54}")
