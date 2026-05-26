"""
Generate a 'dirty' synthetic dataset that mirrors real-world data flaws:

  data/raw/counts.csv                        — RNA-Seq count matrix
                                               (genomic-only patients included;
                                                imaging-only patients absent)
  data/processed/image_embeddings/<id>.pt    — Patch embedding tensors
                                               (imaging-only patients included;
                                                genomic-only patients absent)
  data/raw/clinical_metadata.csv             — Binary treatment-response labels
                                               for a subset of patients only

Cohort layout (N=30):
  TCGA-0000 … TCGA-0019  →  both modalities present  (20 patients)
  TCGA-0020 … TCGA-0024  →  genomic only             ( 5 patients, no .pt)
  TCGA-0025 … TCGA-0029  →  imaging only             ( 5 patients, not in counts.csv)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ── Cohort constants ──────────────────────────────────────────────────────────
N_TOTAL          = 30
N_BOTH           = 20   # TCGA-0000 … TCGA-0019
N_GENOMIC_ONLY   = 5    # TCGA-0020 … TCGA-0024  (no imaging)
N_IMAGING_ONLY   = 5    # TCGA-0025 … TCGA-0029  (no genomics)

N_GENES          = 5_000
EMBED_DIM        = 1_280   # ViT-L/14 (Virchow backbone)
MIN_PATCHES      = 20
MAX_PATCHES      = 500
CLINICAL_FRAC    = 0.75    # fraction of fully-paired patients with a label
SEED             = 42


def _patient_id(i: int) -> str:
    return f"TCGA-{i:04d}"


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "raw":     root / "data" / "raw",
        "img_emb": root / "data" / "processed" / "image_embeddings",
        "gen_emb": root / "data" / "processed" / "genomic_embeddings",
        "interim": root / "data" / "interim" / "patches",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_counts_csv(
    output_path: Path,
    patient_ids: list[str],
    n_genes: int,
    rng: np.random.Generator,
) -> None:
    """Write counts.csv for both-modality + genomic-only patients."""
    gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]

    # Negative-binomial counts approximate real bulk RNA-Seq distributions.
    p      = 0.5 / (0.5 + 50)
    counts = rng.negative_binomial(n=0.5, p=p, size=(len(patient_ids), n_genes)).astype(np.int32)

    df = pd.DataFrame(counts, index=patient_ids, columns=gene_ids)
    df.index.name = "patient_id"
    df.to_csv(output_path)
    print(f"  counts.csv          → {output_path}")
    print(f"                         shape={df.shape}  "
          f"({N_BOTH} paired + {N_GENOMIC_ONLY} genomic-only)")


def generate_patch_embeddings(
    output_dir: Path,
    patient_ids: list[str],
    rng: np.random.Generator,
) -> None:
    """Write one .pt file per patient for both-modality + imaging-only patients."""
    torch.manual_seed(SEED)
    patch_counts = []

    for pid in patient_ids:
        # True biopsy variance: uniform draw over the full 20-500 range.
        n_patches = rng.integers(MIN_PATCHES, MAX_PATCHES + 1)

        emb = torch.randn(int(n_patches), EMBED_DIM)
        emb = emb / emb.norm(dim=1, keepdim=True)

        torch.save(emb, output_dir / f"{pid}.pt")
        patch_counts.append(int(n_patches))

    print(f"  image_embeddings/   → {output_dir}")
    print(f"                         n_files={len(patient_ids)}  "
          f"patches/slide={min(patch_counts)}–{max(patch_counts)}  "
          f"({N_BOTH} paired + {N_IMAGING_ONLY} imaging-only)")


def generate_clinical_metadata(
    output_path: Path,
    paired_ids: list[str],
    rng: np.random.Generator,
) -> None:
    """Write clinical_metadata.csv with binary labels for ~75% of paired patients."""
    n_labelled = max(1, int(len(paired_ids) * CLINICAL_FRAC))
    labelled   = rng.choice(paired_ids, size=n_labelled, replace=False).tolist()
    labels     = rng.integers(0, 2, size=n_labelled).tolist()

    df = pd.DataFrame({"patient_id": labelled, "treatment_response": labels})
    df.to_csv(output_path, index=False)
    print(f"  clinical_metadata.csv → {output_path}")
    print(f"                          {n_labelled}/{len(paired_ids)} paired patients labelled  "
          f"(pos={sum(labels)}, neg={n_labelled - sum(labels)})")


def main(root: Path) -> None:
    print(f"\nGenerating dirty mock dataset under: {root}\n")
    rng  = np.random.default_rng(SEED)
    dirs = make_dirs(root)

    # Build patient ID lists per group
    paired_ids       = [_patient_id(i) for i in range(N_BOTH)]
    genomic_only_ids = [_patient_id(i) for i in range(N_BOTH, N_BOTH + N_GENOMIC_ONLY)]
    imaging_only_ids = [_patient_id(i) for i in range(N_BOTH + N_GENOMIC_ONLY, N_TOTAL)]

    print(f"Cohort  : {N_TOTAL} patients total")
    print(f"  Paired         : {len(paired_ids)}   ({paired_ids[0]}–{paired_ids[-1]})")
    print(f"  Genomic-only   : {len(genomic_only_ids)}    ({genomic_only_ids[0]}–{genomic_only_ids[-1]}, no .pt)")
    print(f"  Imaging-only   : {len(imaging_only_ids)}    ({imaging_only_ids[0]}–{imaging_only_ids[-1]}, not in counts.csv)")
    print()

    # counts.csv: paired + genomic-only (imaging-only deliberately absent)
    generate_counts_csv(
        output_path = dirs["raw"] / "counts.csv",
        patient_ids = paired_ids + genomic_only_ids,
        n_genes     = N_GENES,
        rng         = rng,
    )

    # .pt files: paired + imaging-only (genomic-only deliberately absent)
    generate_patch_embeddings(
        output_dir  = dirs["img_emb"],
        patient_ids = paired_ids + imaging_only_ids,
        rng         = rng,
    )

    # clinical metadata: subset of paired patients only
    generate_clinical_metadata(
        output_path = dirs["raw"] / "clinical_metadata.csv",
        paired_ids  = paired_ids,
        rng         = rng,
    )

    print("\nDone. Dirty simulation dataset ready.\n")
    print("Modality alignment summary:")
    print(f"  Patients in counts.csv      : {N_BOTH + N_GENOMIC_ONLY}")
    print(f"  Patients with .pt files     : {N_BOTH + N_IMAGING_ONLY}")
    print(f"  Intersection (paired)       : {N_BOTH}")
    print(f"  Union (all patients)        : {N_TOTAL}")
    print(f"  Patients with clinical label: ~{int(N_BOTH * CLINICAL_FRAC)}/{N_BOTH} paired")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dirty synthetic data for pipeline robustness testing."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory (default: repo root).",
    )
    args = parser.parse_args()
    main(args.root)
