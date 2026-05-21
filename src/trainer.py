import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from monai.metrics import ROCAUCMetric


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: Path,
        device: torch.device,
    ) -> None:
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        lr = config["training"]["learning_rate"]
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
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
            logits = self.model(patch_emb, genomic, patch_mask)
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

                logits = self.model(patch_emb, genomic, patch_mask)
                probs  = torch.sigmoid(logits)
                self.auc_metric(y_pred=probs, y=labels)

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
    def fit(self, epochs: int) -> None:
        print(f"\nTraining for {epochs} epoch(s) on {self.device}\n{'─'*52}")
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            auc = self.val_epoch(epoch)
            self._maybe_checkpoint(epoch, auc)
        print(f"{'─'*52}\nTraining complete. Best val_auc={self.best_auc:.4f}")


# ── Execution block ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.dataset        import build_dataloader
    from src.models.fusion_model import PathoGenomicFusionModel

    root        = Path(__file__).resolve().parents[1]
    config      = load_config(root / "configs" / "model_config.yaml")
    emb_dir     = root / "data" / "processed" / "image_embeddings"
    counts_path = root / "data" / "raw" / "counts.csv"

    # ── Load synthetic patient data ───────────────────────────────────────────
    print("Loading synthetic data...")
    counts_df    = pd.read_csv(counts_path, index_col="patient_id")
    patient_data = []
    for pt_file in sorted(emb_dir.glob("*.pt")):
        pid = pt_file.stem
        if pid not in counts_df.index:
            continue
        patient_data.append({
            "patient_id":       pid,
            "patch_embeddings": torch.load(pt_file, weights_only=True).numpy(),
            "genomic_counts":   counts_df.loc[pid].to_numpy(),
        })

    # Alternating binary labels ensures both classes present in every split
    for i, d in enumerate(patient_data):
        d["label"] = float(i % 2)

    pos = sum(d["label"] == 1.0 for d in patient_data)
    neg = sum(d["label"] == 0.0 for d in patient_data)
    print(f"  Patients: {len(patient_data)}  (pos={pos}, neg={neg})")

    # ── Train / val split (16 / 4) ────────────────────────────────────────────
    train_loader = build_dataloader(patient_data[:16], batch_size=4, shuffle=True)
    val_loader   = build_dataloader(patient_data[16:], batch_size=4, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    genomic_dim = counts_df.shape[1]
    device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model       = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)

    print(f"  Device : {device}")
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")

    # ── 2-epoch smoke test ────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        config         = config,
        checkpoint_dir = root / "checkpoints",
        device         = device,
    )
    trainer.fit(epochs=2)

    # ── Verify checkpoint was written ─────────────────────────────────────────
    ckpt_path = root / "checkpoints" / "best_model.pt"
    assert ckpt_path.exists(), "Checkpoint file not found!"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"\nCheckpoint verified: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")
