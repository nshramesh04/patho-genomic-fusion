import yaml
import torch
import torch.nn as nn
from pathlib import Path


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class GatedFusion(nn.Module):
    """
    Confidence-aware sigmoid gate for dual-stream fusion.

    Concatenates the 512-dim cross-attention output (visual stream) with the
    512-dim genomic projector output (genomic stream) → 1024-dim input, then
    produces a scalar gate α ∈ (0, 1) via a single linear projection + Sigmoid.

    The gate modulates the visual representation:
        gated = α * visual

    High cross-modal coherence → α → 1 (full visual contribution retained).
    Low coherence or high ambiguity → α < 0.5 (visual contribution suppressed),
    stabilising output for borderline cases near the IHC positivity threshold.

    O(N) complexity is preserved: the gate operates on the single 512-dim token
    produced after O(N) cross-attention, not on any patch-level representation.

    Both weight and bias are initialised to 0, so α = 0.5 exactly for every
    patient at the start of training regardless of input — a fully neutral gate.
    This prevents gradient collapse and ensures reproducible pre-training α values
    when the checkpoint is loaded with strict=False.
    """

    def __init__(self, visual_dim: int = 512, genomic_dim: int = 512) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + genomic_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[0].weight, 0.0)
        nn.init.constant_(self.gate[0].bias, 0.0)

    def forward(
        self,
        visual:  torch.Tensor,   # (B, visual_dim)  — cross-attention output
        genomic: torch.Tensor,   # (B, genomic_dim) — genomic projector token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        gated : (B, visual_dim)   α-modulated visual representation
        alpha : (B, 1)            gate scalar  [logged for interpretability]
        """
        gate_input = torch.cat([visual, genomic], dim=-1)  # (B, 1024)
        alpha      = self.gate(gate_input)                 # (B, 1)
        gated      = alpha * visual                        # broadcast: (B, 512)
        return gated, alpha


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

        # ── Gated fusion ──────────────────────────────────────────────────────
        # Produces α ∈ (0,1) from [visual ‖ genomic] (1024-dim) to modulate
        # the cross-attention output before the post-attention projection.
        self.gated_fusion = GatedFusion(
            visual_dim=query_dim, genomic_dim=query_dim
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

    def encode(
        self,
        patch_embeddings: torch.Tensor,  # (B, N, 768)  zero-padded to batch max N
        genomic_counts:   torch.Tensor,  # (B, G)
        patch_mask:       torch.Tensor,  # (B, N)  bool — True = real token, False = pad
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared computation up to (but not including) the task head(s):
        genomic projection → cross-attention → gated fusion → post-attention
        projection. Factored out of forward() so that subclasses (e.g. a
        survival head branching off the same `fused` representation) can
        reuse this path without duplicating it or breaking forward()'s
        existing 3-tuple return signature relied on throughout this repo.

        Returns
        -------
        fused       : (B, hidden_dim)    post-attention fused representation
        attn_weights: (B, 1, N)          per-patch attention weights
        alpha       : (B, 1)             gate scalar from GatedFusion

        patch_mask convention (matches dataset._collate output):
          True  → real tissue patch — attend to this token
          False → zero-padded slot  — must be masked out

        PyTorch MultiheadAttention key_padding_mask is the INVERSE:
          True  → IGNORE this key position
        So we pass ~patch_mask, ensuring padded slots never contribute
        to the attention distribution.
        """
        # Project genomic counts → query: (B, query_dim)
        query_1d  = self.genomic_projector(genomic_counts)      # (B, 512)
        query_seq = query_1d.unsqueeze(1)                        # (B, 1, 512)

        # Cross-attention: O(N) — 1×N attention map
        attn_out, attn_weights = self.cross_attention(
            query=query_seq,
            key=patch_embeddings,
            value=patch_embeddings,
            key_padding_mask=~patch_mask,
            need_weights=True,
        )
        visual = attn_out.squeeze(1)                             # (B, 512)

        # Gated fusion: α ∈ (0,1) from [visual ‖ genomic] (1024-dim)
        fused, alpha = self.gated_fusion(visual, query_1d)       # (B,512), (B,1)

        fused = self.post_attn(fused)
        return fused, attn_weights, alpha

    def forward(
        self,
        patch_embeddings: torch.Tensor,  # (B, N, 768)  zero-padded to batch max N
        genomic_counts:   torch.Tensor,  # (B, G)
        patch_mask:       torch.Tensor,  # (B, N)  bool — True = real token, False = pad
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits      : (B, num_classes)
        attn_weights: (B, 1, N)          per-patch attention weights
        alpha       : (B, 1)             gate scalar from GatedFusion
        """
        fused, attn_weights, alpha = self.encode(patch_embeddings, genomic_counts, patch_mask)
        return self.head(fused), attn_weights, alpha


class EarlyFusionBaseline(nn.Module):
    """
    Condition 3 ablation — naive early fusion, no cross-attention.

    The RNA-Seq projection is concatenated onto every projected patch
    embedding (rather than serving as a cross-attention query over the
    patches), fused per-patch by a linear layer, then aggregated by
    standard ABMIL-style gated attention pooling (Ilse et al. 2018;
    see src/utils/run_mil_baselines.py:ABMIL). No gate / reliability
    estimator is included — the goal is to isolate the fusion mechanism
    (concatenation vs. cross-attention) only.
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
        dropout    = fb["dropout"]        # 0.1
        hidden_dim = fb["hidden_dim"]     # 512
        att_dim    = 128                  # matches ABMIL's ATT_DIM

        # ── Genomic stream — identical projector to PathoGenomicFusionModel ────
        self.genomic_projector = nn.Sequential(
            nn.Linear(genomic_input_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
        )

        # ── Patch stream — same projector shape/style (768 → 512) ──────────────
        self.patch_projector = nn.Sequential(
            nn.Linear(kv_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
        )

        # ── Early fusion: concat [genomic ‖ patch] (1024) → linear → 512 ───────
        self.fusion = nn.Sequential(
            nn.Linear(query_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Standard (ABMIL-style) gated attention pooling ──────────────────────
        self.att_V = nn.Linear(hidden_dim, att_dim)
        self.att_U = nn.Linear(hidden_dim, att_dim)
        self.att_w = nn.Linear(att_dim, 1)

        # ── Post-attention projection — identical to PathoGenomicFusionModel ───
        self.post_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Task head — identical to PathoGenomicFusionModel ────────────────────
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        patch_embeddings: torch.Tensor,  # (B, N, 768)  zero-padded to batch max N
        genomic_counts:   torch.Tensor,  # (B, G)
        patch_mask:       torch.Tensor,  # (B, N)  bool — True = real token, False = pad
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, num_classes)
        """
        query_1d = self.genomic_projector(genomic_counts)                  # (B, 512)
        patches  = self.patch_projector(patch_embeddings)                  # (B, N, 512)

        n = patches.shape[1]
        genomic_expanded = query_1d.unsqueeze(1).expand(-1, n, -1)         # (B, N, 512)
        concat        = torch.cat([genomic_expanded, patches], dim=-1)     # (B, N, 1024)
        fused_patches = self.fusion(concat)                                # (B, N, 512)

        a = self.att_w(torch.tanh(self.att_V(fused_patches)) *
                        torch.sigmoid(self.att_U(fused_patches)))          # (B, N, 1)
        a = a.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(a, dim=1)                                        # (B, N, 1)

        z = (a * fused_patches).sum(dim=1)                                 # (B, 512)
        z = self.post_attn(z)
        return self.head(z)


class RandomQueryBaseline(PathoGenomicFusionModel):
    """
    Condition — random query token ablation.

    Identical cross-attention architecture to PathoGenomicFusionModel
    (same genomic projector, cross-attention module, gated fusion, and
    task head — same weight init, dropout, and regularization by virtue
    of inheritance). The ONLY change: the query fed into cross-attention
    is a fixed random vector sampled once from N(0, 1) at construction
    time, instead of the RNA-Seq projection. The genomic projection is
    still computed and still feeds the gate (GatedFusion), matching the
    existing model everywhere except the cross-attention query itself.

    Tests whether transcriptomic CONTENT in the query drives the gain,
    or whether the attention mechanism alone accounts for it.
    """

    def __init__(
        self,
        config: dict,
        genomic_input_dim: int = 20513,
        num_classes: int = 1,
    ) -> None:
        super().__init__(config, genomic_input_dim=genomic_input_dim, num_classes=num_classes)
        query_dim = config["fusion_bottleneck"]["query_dim"]
        # Fixed at construction time — sampled once, reused for every patient.
        self.register_buffer("random_query", torch.randn(1, query_dim))

    def encode(
        self,
        patch_embeddings: torch.Tensor,
        genomic_counts:   torch.Tensor,
        patch_mask:       torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_1d  = self.genomic_projector(genomic_counts)         # (B, 512) — still feeds the gate
        bsz       = genomic_counts.shape[0]
        query_seq = self.random_query.expand(bsz, -1).unsqueeze(1)  # (B, 1, 512) — fixed random query

        attn_out, attn_weights = self.cross_attention(
            query=query_seq,
            key=patch_embeddings,
            value=patch_embeddings,
            key_padding_mask=~patch_mask,
            need_weights=True,
        )
        visual = attn_out.squeeze(1)                                # (B, 512)

        fused, alpha = self.gated_fusion(visual, query_1d)
        fused = self.post_attn(fused)
        return fused, attn_weights, alpha

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        genomic_counts:   torch.Tensor,
        patch_mask:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, num_classes)
        """
        fused, _, _ = self.encode(patch_embeddings, genomic_counts, patch_mask)
        return self.head(fused)


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
        logits, attn_weights, alpha = model(patch_emb, genomic, patch_mask)

    print(f"\n── Forward pass output ──────────────────────────────")
    print(f"  logits shape      : {tuple(logits.shape)}  dtype={logits.dtype}")
    print(f"  logits            : {logits.squeeze().tolist()}")
    print(f"  attn_weights shape: {tuple(attn_weights.shape)}  dtype={attn_weights.dtype}")
    print(f"  alpha (gate) shape: {tuple(alpha.shape)}  dtype={alpha.dtype}")
    print(f"  alpha values      : {alpha.squeeze().tolist()}")
    print(f"────────────────────────────────────────────────────")

    # ── Dry-run: single patient, N=100 dummy patches ─────────────────────────
    print("\n── Dry-run (B=1, N=100) ─────────────────────────────")
    dummy_patches  = torch.randn(1, 100, 768)
    dummy_genomic  = torch.randn(1, genomic_input_dim)
    dummy_mask     = torch.ones(1, 100, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        d_logits, d_attn, d_alpha = model(dummy_patches, dummy_genomic, dummy_mask)

    print(f"  patch_embeddings : (1, 100, 768)")
    print(f"  logits           : {tuple(d_logits.shape)}")
    print(f"  attn_weights     : {tuple(d_attn.shape)}   ← 1×N attention map (O(N))")
    print(f"  alpha (gate)     : {tuple(d_alpha.shape)}   value = {d_alpha.item():.4f}")
    assert d_attn.shape  == (1, 1, 100), "attn_weights shape mismatch"
    assert d_alpha.shape == (1, 1),      "alpha shape mismatch"
    print("  Shape assertions passed ✓")
    print(f"────────────────────────────────────────────────────")
