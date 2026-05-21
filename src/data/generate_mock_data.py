"""
Generate synthetic data to validate both processing streams:
  - data/raw/counts.csv          : RNA-Seq count matrix (patients x genes)
  - data/processed/image_embeddings/<patient_id>.pt : patch embedding tensors (N x 1280)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


N_PATIENTS = 20
N_GENES = 5_000       # realistic gene count; variance filter will select top 2000
EMBED_DIM = 1280      # ViT-L/14 output dimension (Virchow backbone)
MIN_PATCHES = 50      # minimum tissue patches per slide
MAX_PATCHES = 300     # maximum tissue patches per slide
SEED = 42


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw":        root / "data" / "raw",
        "img_emb":    root / "data" / "processed" / "image_embeddings",
        "gen_emb":    root / "data" / "processed" / "genomic_embeddings",
        "interim":    root / "data" / "interim" / "patches",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_counts_csv(output_path: Path, n_patients: int, n_genes: int, rng: np.random.Generator) -> None:
    patient_ids = [f"TCGA-{i:04d}" for i in range(n_patients)]
    gene_ids    = [f"ENSG{i:011d}" for i in range(n_genes)]

    # Negative-binomial counts approximate real RNA-Seq count distributions.
    # mean ~50 counts, dispersion r=0.5 → overdispersed like bulk RNA-Seq.
    p = 0.5 / (0.5 + 50)
    counts = rng.negative_binomial(n=0.5, p=p, size=(n_patients, n_genes)).astype(np.int32)

    df = pd.DataFrame(counts, index=patient_ids, columns=gene_ids)
    df.index.name = "patient_id"
    df.to_csv(output_path)
    print(f"  counts.csv         → {output_path}  shape={df.shape}")


def generate_patch_embeddings(output_dir: Path, n_patients: int, rng: np.random.Generator) -> None:
    torch.manual_seed(SEED)
    patch_counts = []
    for i in range(n_patients):
        patient_id = f"TCGA-{i:04d}"
        n_patches  = random.randint(MIN_PATCHES, MAX_PATCHES)

        # Unit-normalised embeddings mimic the output of a frozen ViT backbone.
        emb = torch.randn(n_patches, EMBED_DIM)
        emb = emb / emb.norm(dim=1, keepdim=True)

        out_path = output_dir / f"{patient_id}.pt"
        torch.save(emb, out_path)
        patch_counts.append(n_patches)

    print(
        f"  image_embeddings/  → {output_dir}  "
        f"n_files={n_patients}  patches/slide={min(patch_counts)}–{max(patch_counts)}  "
        f"embed_dim={EMBED_DIM}"
    )


def main(root: Path) -> None:
    print(f"\nGenerating mock data under: {root}\n")
    rng  = np.random.default_rng(SEED)
    dirs = make_dirs(root)

    generate_counts_csv(
        output_path=dirs["raw"] / "counts.csv",
        n_patients=N_PATIENTS,
        n_genes=N_GENES,
        rng=rng,
    )
    generate_patch_embeddings(
        output_dir=dirs["img_emb"],
        n_patients=N_PATIENTS,
        rng=rng,
    )
    print("\nDone. Both processing streams have synthetic inputs ready.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for pipeline validation.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory (default: repo root).",
    )
    args = parser.parse_args()
    main(args.root)
