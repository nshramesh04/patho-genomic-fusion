"""
Download pre-extracted patch embedding tensors from HuggingFace,
filtered to patients present in data/raw/counts.csv.

Repository : Dearcat/CPathPatchFeature
Structure  : {cohort}/{model}/pt_files/{TCGA-XX-XXXX-01Z-00-DX{n}.UUID}.pt

For patients with more than one slide (DX1, DX2 …) only the first
alphabetically-sorted file is downloaded, since our model expects a
single embedding tensor per patient.

Outputs:
  data/interim/hf_downloads/   — raw HuggingFace cache (original names)
  data/processed/image_embeddings/{TCGA-XX-XXXX}.pt  — renamed, pipeline-ready
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


REPO_ID   = "Dearcat/CPathPatchFeature"
REPO_TYPE = "dataset"


# ── Counts cohort loader ──────────────────────────────────────────────────────

def load_counts_ids(counts_path: Path) -> set[str]:
    """Return the set of 12-char patient barcodes from counts.csv.

    Reads only the patient_id column to avoid loading the full 20k-gene matrix.
    """
    if not counts_path.exists():
        sys.exit(f"[ERROR] counts.csv not found: {counts_path}\n"
                 "Run src/data/format_cbioportal.py first.")
    ids = pd.read_csv(counts_path, usecols=["patient_id"])["patient_id"]
    return set(ids)


def filter_to_counts(
    patient_map: dict[str, str],
    counts_ids:  set[str],
) -> dict[str, str]:
    """Keep only patients present in both the HF manifest and counts.csv."""
    return {pid: path for pid, path in patient_map.items() if pid in counts_ids}


# ── Manifest helpers ──────────────────────────────────────────────────────────

def list_pt_files(cohort: str, model: str) -> list[str]:
    """Return all .pt repo paths under {cohort}/{model}/pt_files/."""
    api    = HfApi()
    prefix = f"{cohort}/{model}/pt_files/"
    files  = [
        f for f in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    return sorted(files)


def select_one_per_patient(repo_paths: list[str]) -> dict[str, str]:
    """
    Map each 12-character patient barcode to exactly one repo path.
    When a patient has multiple slides (DX1, DX2 …) keep the first
    alphabetically — consistent, deterministic, no random seed needed.
    """
    chosen: dict[str, str] = {}
    for path in sorted(repo_paths):
        fname      = Path(path).stem                  # e.g. TCGA-3C-AALI-01Z-00-DX1.UUID
        patient_id = fname[:12]                        # TCGA-3C-AALI
        if patient_id not in chosen:
            chosen[patient_id] = path
    return chosen


# ── Download ──────────────────────────────────────────────────────────────────

def download_embeddings(
    patient_map:   dict[str, str],
    interim_dir:   Path,
    output_dir:    Path,
    token:         str | None,
    max_patients:  int | None,
    dry_run:       bool,
) -> None:
    """Download and rename embedding files for each patient in patient_map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    items = list(patient_map.items())
    if max_patients is not None:
        items = items[:max_patients]

    total     = len(items)
    skipped   = 0
    downloaded = 0

    print(f"  Patients to fetch  : {total}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Interim cache      : {interim_dir}")
    if dry_run:
        print("  [DRY RUN — no files will be written]\n")
    print()

    for idx, (patient_id, repo_path) in enumerate(items, 1):
        dest = output_dir / f"{patient_id}.pt"

        if dest.exists():
            skipped += 1
            print(f"  [{idx:>4}/{total}] skip   {patient_id}  (already exists)")
            continue

        if dry_run:
            print(f"  [{idx:>4}/{total}] would  {patient_id}  ←  {repo_path}")
            continue

        try:
            cached = hf_hub_download(
                repo_id   = REPO_ID,
                repo_type = REPO_TYPE,
                filename  = repo_path,
                local_dir = str(interim_dir),
                token     = token,
            )
            shutil.copy2(cached, dest)
            downloaded += 1
            print(f"  [{idx:>4}/{total}] done   {patient_id}")

        except Exception as exc:
            print(f"  [{idx:>4}/{total}] ERROR  {patient_id}: {exc}", file=sys.stderr)

    if not dry_run:
        print(f"\n  Downloaded : {downloaded}")
        print(f"  Skipped    : {skipped}  (already present)")
        print(f"  Total in output dir: {len(list(output_dir.glob('*.pt')))}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root   = Path(__file__).resolve().parents[2]
    interim_dir = repo_root / "data" / "interim" / "hf_downloads"
    output_dir  = repo_root / "data" / "processed" / "image_embeddings"

    parser = argparse.ArgumentParser(
        description="Download TCGA patch embeddings from Dearcat/CPathPatchFeature on HuggingFace."
    )
    parser.add_argument(
        "--cohort",
        default="brca",
        help="Cohort sub-directory in the repo (default: brca).",
    )
    parser.add_argument(
        "--model",
        default="chief",
        choices=["chief", "gigap"],
        help="Feature-extractor backbone to use (default: chief).",
    )
    parser.add_argument(
        "--counts_path",
        type=Path,
        default=repo_root / "data" / "raw" / "counts.csv",
        help="Path to counts.csv; only patients present here will be downloaded.",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=50,
        metavar="N",
        help="Download at most N intersecting patients (default: 50).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=output_dir,
        help=f"Destination for renamed .pt files (default: {output_dir}).",
    )
    parser.add_argument(
        "--interim_dir",
        type=Path,
        default=interim_dir,
        help=f"Raw HuggingFace cache directory (default: {interim_dir}).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token. Falls back to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded without writing any files.",
    )
    args = parser.parse_args()

    print(f"\n── HuggingFace embedding downloader ─────────────────────────────")
    print(f"  Repo    : {REPO_ID}")
    print(f"  Cohort  : {args.cohort}  /  Model: {args.model}\n")

    print(f"Loading patient IDs from counts.csv …")
    counts_ids  = load_counts_ids(args.counts_path)
    print(f"  Patients in counts.csv : {len(counts_ids)}\n")

    print("Fetching HuggingFace file manifest …")
    repo_paths  = list_pt_files(args.cohort, args.model)
    patient_map = select_one_per_patient(repo_paths)
    multi_slide = len(repo_paths) - len(patient_map)
    print(f"  .pt files in manifest  : {len(repo_paths)}")
    print(f"  Unique HF patients     : {len(patient_map)}")
    print(f"  Extra slides dropped   : {multi_slide}  (kept first slide per patient)")

    matched = filter_to_counts(patient_map, counts_ids)
    hf_only = len(patient_map) - len(matched)
    csv_only = len(counts_ids) - len(matched)
    print(f"\n  Intersection (both)    : {len(matched)}")
    print(f"  HF only (no counts)    : {hf_only}")
    print(f"  counts only (no .pt)   : {csv_only}")
    print(f"  Fetching first         : {min(args.max_patients, len(matched))}\n")

    download_embeddings(
        patient_map  = matched,
        interim_dir  = args.interim_dir,
        output_dir   = args.output_dir,
        token        = args.token,
        max_patients = args.max_patients,
        dry_run      = args.dry_run,
    )

    print("── Done ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
