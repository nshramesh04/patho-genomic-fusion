"""
Format raw cBioPortal download files into the pipeline-ready layout:

  data_mrna_seq_v2_rsem.txt     →  data/raw/counts.csv
  data_clinical_patient.txt     →  data/raw/clinical_metadata.csv

cBioPortal file conventions handled:
  - mRNA file: Hugo_Symbol + Entrez_Gene_Id as first two columns; patients
    as remaining columns; genes as rows.  Needs transposing.
  - Clinical file: first N lines start with '#' (metadata / display headers);
    first non-'#' line is the true column header row.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ── mRNA formatter ────────────────────────────────────────────────────────────

def format_mrna(input_path: Path, output_path: Path) -> None:
    """
    Reads cBioPortal RSEM matrix (genes × patients) and writes a
    patient × gene counts.csv.

    Steps:
      1. Index on Hugo_Symbol; drop Entrez_Gene_Id.
      2. Collapse duplicate gene symbols by mean (different Entrez IDs,
         same Hugo name).
      3. Transpose → patients × genes.
      4. Truncate sample barcodes to 12 chars  (TCGA-XX-XXXX).
      5. Drop duplicate patient indices, keeping the first occurrence.
    """
    if not input_path.exists():
        sys.exit(f"[ERROR] mRNA file not found: {input_path}")

    print(f"Reading mRNA file  : {input_path}")
    df = pd.read_csv(input_path, sep="\t", index_col="Hugo_Symbol", low_memory=False)
    df = df.drop(columns=["Entrez_Gene_Id"], errors="ignore")

    # Collapse duplicate gene symbols
    n_genes_raw = len(df)
    df = df.groupby(df.index).mean()
    n_collapsed = n_genes_raw - len(df)

    # Transpose: patients as rows, gene symbols as columns
    df = df.T
    df.index.name = "patient_id"

    # Truncate TCGA barcodes: TCGA-AB-1234-01 → TCGA-AB-1234
    df.index = df.index.str[:12]

    # Drop duplicate patients (e.g. multiple tumour portions from same case)
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dup_dropped = n_before - len(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    print(f"  genes raw          : {n_genes_raw:>6}  →  after dedup symbol: {len(df.columns)}  "
          f"(collapsed {n_collapsed})")
    print(f"  patients           : {n_before:>6}  →  after dedup barcode: {len(df)}  "
          f"(dropped {n_dup_dropped})")
    print(f"  output shape       : {df.shape}")
    print(f"  saved →  {output_path}\n")


# ── Clinical formatter ────────────────────────────────────────────────────────

def format_clinical(
    input_path: Path,
    output_path: Path,
    target_col: str = "PR_STATUS_BY_IHC",
) -> None:
    """
    Reads a cBioPortal clinical patient file and writes clinical_metadata.csv
    with columns [patient_id, label].

    cBioPortal clinical files have 4-5 leading '#' comment lines (display
    names, descriptions, datatypes, priority) before the true header row.
    pandas comment='#' skips them automatically.

    Label mapping  (all other values → dropped):
      Positive  →  1
      Negative  →  0
    """
    if not input_path.exists():
        sys.exit(f"[ERROR] Clinical file not found: {input_path}")

    print(f"Reading clinical file: {input_path}")
    df = pd.read_csv(input_path, sep="\t", comment="#", low_memory=False)

    # Normalise patient ID column name (varies slightly across cBioPortal studies)
    id_aliases = ["PATIENT_ID", "Patient Identifier", "patient_id"]
    id_col = next((c for c in id_aliases if c in df.columns), None)
    if id_col is None:
        sys.exit(f"[ERROR] Cannot find patient ID column. Columns present: {df.columns.tolist()}")
    df = df.rename(columns={id_col: "patient_id"}).set_index("patient_id")

    if target_col not in df.columns:
        sys.exit(
            f"[ERROR] Target column '{target_col}' not found.\n"
            f"        Available columns: {df.columns.tolist()}"
        )

    label_map = {"Positive": 1, "Negative": 0}
    df["label"] = df[target_col].map(label_map)

    n_before = len(df)
    unmapped = df["label"].isna()
    if unmapped.any():
        unmapped_vals = df.loc[unmapped, target_col].value_counts().to_dict()
        print(f"  dropping {unmapped.sum()} patients with unmapped '{target_col}' values: "
              f"{unmapped_vals}")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    out = df[["label"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path)

    print(f"  patients raw       : {n_before:>6}  →  after label filter: {len(out)}")
    print(f"  label distribution : pos={out['label'].sum()}  "
          f"neg={(out['label'] == 0).sum()}")
    print(f"  saved →  {output_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir   = repo_root / "data" / "raw"

    parser = argparse.ArgumentParser(
        description="Format cBioPortal download files for the patho-genomic pipeline."
    )
    parser.add_argument(
        "--mrna",
        type=Path,
        default=raw_dir / "data_mrna_seq_v2_rsem.txt",
        help="Path to cBioPortal mRNA RSEM file.",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=raw_dir / "data_clinical_patient.txt",
        help="Path to cBioPortal clinical patient file.",
    )
    parser.add_argument(
        "--counts_out",
        type=Path,
        default=raw_dir / "counts.csv",
        help="Output path for the patient × gene counts matrix.",
    )
    parser.add_argument(
        "--clinical_out",
        type=Path,
        default=raw_dir / "clinical_metadata.csv",
        help="Output path for the patient × label clinical file.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="PR_STATUS_BY_IHC",
        help="cBioPortal column to use as the binary label.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip a formatter silently if its input file does not exist "
             "(useful when only one file is available).",
    )
    args = parser.parse_args()

    print("\n── cBioPortal formatter ─────────────────────────────────────────\n")

    mrna_missing     = not args.mrna.exists()
    clinical_missing = not args.clinical.exists()

    if mrna_missing and clinical_missing:
        sys.exit(
            "[ERROR] Neither input file found.\n"
            f"  mRNA    : {args.mrna}\n"
            f"  Clinical: {args.clinical}\n"
            "Place the cBioPortal download files in data/raw/ and re-run."
        )

    if mrna_missing and not args.skip_missing:
        sys.exit(f"[ERROR] mRNA file not found: {args.mrna}\n"
                 "Use --skip_missing to skip unavailable files.")
    if clinical_missing and not args.skip_missing:
        sys.exit(f"[ERROR] Clinical file not found: {args.clinical}\n"
                 "Use --skip_missing to skip unavailable files.")

    if not mrna_missing:
        format_mrna(args.mrna, args.counts_out)
    if not clinical_missing:
        format_clinical(args.clinical, args.clinical_out, target_col=args.target_col)

    print("── Done ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
