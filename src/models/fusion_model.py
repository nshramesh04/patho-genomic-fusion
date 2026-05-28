import yaml
import torch
import torch.nn as nn
from pathlib import Path


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class PathoGenomicFusionModel(nn.Module):
    """
    Dual-stream fusion model.

    Genomic counts are projected into a query vector (Q).
    Pathology patch embeddings serve as keys (K) and values (V).
    Multi-Head Cross-Attention attends to the morphological patches that are
    most predictive given each patient's genomic profile.
    """

    def __init__(
        self,
        config: dict,
        genomic_input_dim: int = 20513,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        fb         = config["fusion_bottleneck"]
        query_dim  = fb["query_dim"]      # 512
        kv_dim     = fb["key_value_dim"]  # 768
        num_heads  = fb["num_heads"]      # 8
        dropout    = fb["dropout"]        # 0.1
        hidden_dim = fb["hidden_dim"]     # 512

        # ── Genomic stream ────────────────────────────────────────────────────
        # Projects high-dimensional counts (G,) → compact query token (query_dim,)
        self.genomic_projector = nn.Sequential(
            nn.Linear(genomic_input_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
        )

        # ── Cross-attention fusion ────────────────────────────────────────────
        # Q: genomic embedding  (B, 1, query_dim=512)
        # K: patch embeddings   (B, N, kv_dim=768)
        # V: patch embeddings   (B, N, kv_dim=768)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=dropout,
            batch_first=True,
        )

        # ── Post-attention projection ─────────────────────────────────────────
        self.post_attn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Task head ─────────────────────────────────────────────────────────
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        patch_embeddings: torch.Tensor,  # (B, N, 768)  zero-padded to batch max N
        genomic_counts:   torch.Tensor,  # (B, G)
        patch_mask:       torch.Tensor,  # (B, N)  bool — True = real token, False = pad
    ) -> torch.Tensor:                   # (B, num_classes)
        """
        patch_mask convention (matches dataset._collate output):
          True  → real tissue patch — attend to this token
          False → zero-padded slot  — must be masked out

        PyTorch MultiheadAttention key_padding_mask is the INVERSE:
          True  → IGNORE this key position
        So we pass ~patch_mask, ensuring padded slots never contribute
        to the attention distribution.
        """
        # Project genomic counts → query token: (B, 1, query_dim)
        query = self.genomic_projector(genomic_counts).unsqueeze(1)

        # Invert mask: our True=valid → PyTorch wants True=ignore
        attn_out, attn_weights = self.cross_attention(
            query=query,
            key=patch_embeddings,
            value=patch_embeddings,
            key_padding_mask=~patch_mask,
            need_weights=True,
        )

        # Remove query-sequence dim: (B, 1, query_dim) → (B, query_dim)
        fused = attn_out.squeeze(1)

        fused = self.post_attn(fused)
        return self.head(fused), attn_weights   # (B, num_classes), (B, 1, N)


if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.dataset import build_dataloader

    root        = Path(__file__).resolve().parents[2]
    config      = _load_config(root / "configs" / "model_config.yaml")
    emb_dir     = root / "data" / "processed" / "image_embeddings"
    counts_path = root / "data" / "raw" / "counts.csv"

    # ── Load synthetic data ───────────────────────────────────────────────────
    print("Loading synthetic data...")
    counts_df   = pd.read_csv(counts_path, index_col="patient_id")
    patient_data = []
    for pt_file in sorted(emb_dir.glob("*.pt")):
        patient_id = pt_file.stem
        if patient_id not in counts_df.index:
            continue
        patient_data.append({
            "patient_id":       patient_id,
            "patch_embeddings": torch.load(pt_file, weights_only=True).numpy(),
            "genomic_counts":   counts_df.loc[patient_id].to_numpy(),
        })

    loader = build_dataloader(patient_data, batch_size=4, shuffle=False)
    batch  = next(iter(loader))

    patch_emb  = batch["patch_embeddings"]   # (B, N, 768)
    genomic    = batch["genomic_counts"]     # (B, 20513)
    patch_mask = batch["patch_mask"]         # (B, N)

    print(f"  patch_embeddings : {tuple(patch_emb.shape)}  dtype={patch_emb.dtype}")
    print(f"  genomic_counts   : {tuple(genomic.shape)}  dtype={genomic.dtype}")
    print(f"  patch_mask       : {tuple(patch_mask.shape)}  dtype={patch_mask.dtype}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    genomic_input_dim = genomic.shape[1]   # 20513
    model  = PathoGenomicFusionModel(config, genomic_input_dim=genomic_input_dim)
    model.eval()

    print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        logits, attn_weights = model(patch_emb, genomic, patch_mask)

    print(f"\n── Forward pass output ──────────────────────────────")
    print(f"  logits shape      : {tuple(logits.shape)}  dtype={logits.dtype}")
    print(f"  logits            : {logits.squeeze().tolist()}")
    print(f"  attn_weights shape: {tuple(attn_weights.shape)}  dtype={attn_weights.dtype}")
    print(f"────────────────────────────────────────────────────")
