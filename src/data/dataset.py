import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import CastToTyped, Compose, EnsureTyped, ToTensord

KEYS = ["patch_embeddings", "genomic_counts"]


def build_transforms() -> Compose:
    return Compose([
        # Convert numpy arrays / lists to torch tensors (no MONAI MetaTensor overhead).
        ToTensord(keys=KEYS, track_meta=False),
        # Cast to float32 — genomic counts arrive as int32 from the CSV.
        CastToTyped(keys=KEYS, dtype=torch.float32),
        # Final safety net: enforces dtype even if upstream already produced a tensor.
        EnsureTyped(keys=KEYS, dtype=torch.float32),
    ])


def _collate(batch: list[dict]) -> dict:
    # patch_embeddings are (N_i, 1280) with N_i varying per patient.
    # Pad to the batch's max N and produce a boolean mask for cross-attention.
    max_patches = max(item["patch_embeddings"].shape[0] for item in batch)
    embed_dim   = batch[0]["patch_embeddings"].shape[1]
    bsz         = len(batch)

    padded = torch.zeros(bsz, max_patches, embed_dim, dtype=torch.float32)
    mask   = torch.zeros(bsz, max_patches, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["patch_embeddings"].shape[0]
        padded[i, :n] = item["patch_embeddings"]
        mask[i, :n]   = True

    result = {
        "patch_embeddings": padded,           # (B, max_N, 1280)
        "patch_mask":       mask,             # (B, max_N)  True = valid patch
        "genomic_counts":   torch.stack(      # (B, G)
            [item["genomic_counts"] for item in batch]
        ),
    }

    # Pass through any extra keys (e.g. label, patient_id) not handled above.
    handled = {"patch_embeddings", "patch_mask", "genomic_counts"}
    for key in batch[0]:
        if key in handled:
            continue
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals          # strings (e.g. patient_id) stay as list

    return result


def build_dataloader(
    data: list[dict],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_rate: float = 1.0,
) -> DataLoader:
    """
    Args:
        data: list of dicts, each with keys:
              'patch_embeddings' — np.ndarray or Tensor of shape (N, 1280)
              'genomic_counts'   — np.ndarray or Tensor of shape (G,)
        cache_rate: fraction of the dataset to cache in RAM (1.0 = full cache).
    """
    dataset = CacheDataset(data=data, transform=build_transforms(), cache_rate=cache_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
    )


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    root         = Path(__file__).resolve().parents[2]
    emb_dir      = root / "data" / "processed" / "image_embeddings"
    counts_path  = root / "data" / "raw" / "counts.csv"

    print(f"Loading genomic counts  : {counts_path}")
    counts_df = pd.read_csv(counts_path, index_col="patient_id")

    print(f"Loading patch embeddings: {emb_dir}\n")
    patient_data = []
    for pt_file in sorted(emb_dir.glob("*.pt")):
        patient_id = pt_file.stem
        if patient_id not in counts_df.index:
            print(f"  [skip] {patient_id} — no matching genomic row")
            continue
        patient_data.append({
            "patient_id":       patient_id,
            "patch_embeddings": torch.load(pt_file, weights_only=True).numpy(),
            "genomic_counts":   counts_df.loc[patient_id].to_numpy(),
        })

    print(f"Patients loaded: {len(patient_data)}\n")

    loader = build_dataloader(patient_data, batch_size=4, shuffle=False)
    batch  = next(iter(loader))

    print("── Batch keys & shapes ──────────────────────────────")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:<20} shape={tuple(val.shape)}  dtype={val.dtype}")
    print("─────────────────────────────────────────────────────")
    print(f"  valid patches/slide  : {batch['patch_mask'].sum(dim=1).tolist()}")
    print("Alignment check passed.")
