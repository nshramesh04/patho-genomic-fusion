from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import CastToTyped, Compose, EnsureTyped, ToTensord

KEYS = ["patch_embeddings", "genomic_counts"]


def load_and_qc_patients(
    counts_path: Path,
    emb_dir: Path,
    clinical_path: Path,
) -> list[dict]:
    """
    Loads all three data sources, computes the strict patient-ID intersection,
    and returns a filtered patient list ready for build_dataloader().

    Any patient absent from even one source is dropped; a summary is printed
    explaining exactly which patients were removed and why.

    Returns:
        List of dicts with keys: patient_id, patch_embeddings (np.ndarray,
        shape N×768), genomic_counts (np.ndarray, shape G), label (float).
    """
    counts_df   = pd.read_csv(counts_path, index_col="patient_id")
    clinical_df = pd.read_csv(clinical_path)
    pt_files    = {p.stem: p for p in sorted(emb_dir.glob("*.pt"))}

    genomic_ids  = set(counts_df.index)
    imaging_ids  = set(pt_files)
    clinical_ids = set(clinical_df["patient_id"])
    valid_ids    = genomic_ids & imaging_ids & clinical_ids
    all_ids      = genomic_ids | imaging_ids | clinical_ids

    # ── Drop report ───────────────────────────────────────────────────────────
    dropped: dict[str, list[str]] = {}
    for pid in sorted(all_ids - valid_ids):
        reasons = []
        if pid not in genomic_ids:
            reasons.append("no genomic counts")
        if pid not in imaging_ids:
            reasons.append("no patch embeddings")
        if pid not in clinical_ids:
            reasons.append("no clinical label")
        dropped[pid] = reasons

    print("── QC: Modality Intersection ────────────────────────────────")
    print(f"  genomic patients   : {len(genomic_ids)}")
    print(f"  imaging patients   : {len(imaging_ids)}")
    print(f"  clinical patients  : {len(clinical_ids)}")
    print(f"  valid (all three)  : {len(valid_ids)}")
    print(f"  dropped            : {len(dropped)}")

    if dropped:
        by_reason: dict[str, list[str]] = defaultdict(list)
        for pid, reasons in dropped.items():
            by_reason[" + ".join(sorted(reasons))].append(pid)
        for reason, pids in sorted(by_reason.items()):
            print(f"    [{reason}]  →  {', '.join(pids)}")

    print("─────────────────────────────────────────────────────────────\n")

    # ── Build patient dicts ───────────────────────────────────────────────────
    label_series = clinical_df.set_index("patient_id")["label"]
    patient_data = [
        {
            "patient_id":       pid,
            "patch_embeddings": torch.load(pt_files[pid], weights_only=True).numpy(),
            "genomic_counts":   counts_df.loc[pid].to_numpy(),
            "label":            float(label_series[pid]),
        }
        for pid in sorted(valid_ids)
    ]

    return patient_data


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
    # patch_embeddings are (N_i, 768) with N_i varying per patient.
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
        "patch_embeddings": padded,           # (B, max_N, 768)
        "patch_mask":       mask,             # (B, max_N)  True = valid patch
        "genomic_counts":   torch.stack(      # (B, G)
            [item["genomic_counts"] for item in batch]
        ),
    }

    # Pass through any extra keys (e.g. label, patient_id) not handled above.
    # Skip MONAI's internal *_transforms metadata keys — not needed downstream.
    handled = {"patch_embeddings", "patch_mask", "genomic_counts"}
    for key in batch[0]:
        if key in handled or key.endswith("_transforms"):
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
              'patch_embeddings' — np.ndarray or Tensor of shape (N, 768)
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
    root            = Path(__file__).resolve().parents[2]
    counts_path     = root / "data" / "raw" / "counts.csv"
    emb_dir         = root / "data" / "processed" / "image_embeddings"
    clinical_path   = root / "data" / "raw" / "clinical_metadata.csv"

    patient_data = load_and_qc_patients(counts_path, emb_dir, clinical_path)
    print(f"Patients after QC: {len(patient_data)}\n")

    loader = build_dataloader(patient_data, batch_size=4, shuffle=False)
    batch  = next(iter(loader))

    print("── Batch keys & shapes ──────────────────────────────────────")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:<22} shape={tuple(val.shape)}  dtype={val.dtype}")
        else:
            print(f"  {key:<22} {val}")
    print("─────────────────────────────────────────────────────────────")
    print(f"  valid patches/slide : {batch['patch_mask'].sum(dim=1).tolist()}")
    print("  patch_mask encodes  : 1 = real tissue token, 0 = zero-padded token")
    print("QC + alignment check passed.")
