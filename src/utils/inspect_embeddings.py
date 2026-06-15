"""
Inspect the on-disk structure of a CHIEF patch-embedding .pt file.

Usage:
    python src/utils/inspect_embeddings.py
    python src/utils/inspect_embeddings.py --path data/processed/image_embeddings/TCGA-A2-A0CK.pt
"""

import argparse
from pathlib import Path

import torch


SPATIAL_KEYS = {"coords", "positions", "coordinates", "patch_coords", "metadata"}


def inspect(path: Path) -> None:
    print(f"\n── Embedding inspector ───────────────────────────────────────")
    print(f"  File : {path}")
    print(f"  Size : {path.stat().st_size / 1_048_576:.1f} MB")

    obj = torch.load(path, weights_only=True)

    print(f"\n  Top-level type : {type(obj).__name__}")

    if isinstance(obj, torch.Tensor):
        print(f"\n  Raw tensor detected — no spatial coordinate wrapper.")
        print(f"  shape  : {tuple(obj.shape)}")
        print(f"  dtype  : {obj.dtype}")
        print(f"  device : {obj.device}")
        n_patches, embed_dim = obj.shape
        print(f"\n  Interpretation:")
        print(f"    N patches  : {n_patches:,}")
        print(f"    Embed dim  : {embed_dim}  (CHIEF ViT-B backbone)")
        print(f"    Min value  : {obj.min().item():.4f}")
        print(f"    Max value  : {obj.max().item():.4f}")
        print(f"    Mean/Std   : {obj.mean().item():.4f} / {obj.std().item():.4f}")
        print(f"\n  Spatial coords : NOT PRESENT — embeddings are patch-order only.")
        print(f"                   No 'coords', 'positions', or 'metadata' keys.")

    elif isinstance(obj, dict):
        print(f"\n  Dict detected — {len(obj)} keys:")
        for key, val in obj.items():
            if isinstance(val, torch.Tensor):
                print(f"    '{key}' : Tensor  shape={tuple(val.shape)}  dtype={val.dtype}")
            else:
                print(f"    '{key}' : {type(val).__name__}  value={val!r}")

        found_spatial = {k: obj[k] for k in SPATIAL_KEYS if k in obj}
        if found_spatial:
            print(f"\n  Spatial keys found: {list(found_spatial)}")
            for k, v in found_spatial.items():
                print(f"    '{k}' : shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"\n  Spatial coords : NOT PRESENT")
            print(f"  None of {sorted(SPATIAL_KEYS)} found in dict keys.")

    else:
        print(f"\n  Unexpected type: {type(obj)}  —  raw repr:\n  {obj!r}")

    print(f"\n── Done ──────────────────────────────────────────────────────\n")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default = root / "data" / "processed" / "image_embeddings" / "TCGA-A1-A0SD.pt"

    parser = argparse.ArgumentParser(description="Inspect a CHIEF patch-embedding .pt file.")
    parser.add_argument(
        "--path",
        type=Path,
        default=default,
        help=f"Path to a .pt embedding file (default: {default.name})",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"File not found: {args.path}")

    inspect(args.path)


if __name__ == "__main__":
    main()
