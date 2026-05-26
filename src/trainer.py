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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data.dataset        import build_dataloader, load_and_qc_patients
    from src.models.fusion_model import PathoGenomicFusionModel

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

    # ── Stratified train / val split (12 / 3) ────────────────────────────────
    # Separate by class so val always contains at least one sample of each label.
    pos_patients = [d for d in patient_data if d["label"] == 1.0]
    neg_patients = [d for d in patient_data if d["label"] == 0.0]
    val_data   = pos_patients[:2] + neg_patients[:1]   # 2 pos + 1 neg
    train_data = pos_patients[2:] + neg_patients[1:]   # remaining 12

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
    model       = PathoGenomicFusionModel(config, genomic_input_dim=genomic_dim)

    print(f"Model  : PathoGenomicFusionModel  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Device : {device}\n")

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
    print("Smoke test passed.")
