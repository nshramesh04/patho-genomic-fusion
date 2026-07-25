# Asymmetric Cross-Attention Networks for Multimodal Integration in Pathology

[![White Paper](https://img.shields.io/badge/White%20Paper-GitHub%20Pages-blue)](https://nshramesh04.github.io/patho-genomic-fusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> **Read the full white paper:** [nshramesh04.github.io/patho-genomic-fusion](https://nshramesh04.github.io/patho-genomic-fusion)

PR-status classification in breast cancer by fusing RNA-Seq
transcriptomics with whole-slide imaging via an asymmetric
cross-attention architecture validated on 951 TCGA-BRCA patients.

---

## Results

| Metric | Value | Notes |
|:---|:---:|:---:|
| ROC-AUC | **0.8482** | +0.0801 vs late fusion baseline |
| PR-AUC | **0.9054** | at 2:1 class imbalance |
| Brier Score | **0.17** | well-calibrated |
| Cohort | N = 951 | TCGA-BRCA, 191-patient held-out cohort |
| Interpretability (p-value) | **0.018** | PR- vs PR+ phenotypic separation |

---

## Key Contributions

- **O(N) asymmetric cross-attention** resolves the structural
  incompatibility between fixed-length genomic vectors and
  variable-length WSI patch bags, without positional encodings
  or bag compression.
- **Top-1%-Mass Concentration metric** is a slide-size-invariant
  alternative to Shannon entropy that recovers p = 0.018
  phenotypic separation; entropy analysis suppresses this
  entirely (p = 0.401) due to the slide-size confound.
- **Cross-Modal Reliability Estimator** (Gated-Fusion v2)
  produces a per-patient confidence score for automated triage
  of discordant cases. The gate separates PR+ and PR- patients
  across the confidence spectrum: 83.1% of concordant cases
  are PR+ and 67.2% of flagged cases are PR-, with an
  escalation rate of 31.9% on the held-out cohort.
- **+0.0801 ROC-AUC** over the late-fusion baseline,
  attributable to genomic conditioning applied upstream of
  visual compression, at the search stage rather than the
  pooling stage.

---

## Repository Structure

```text
patho-genomic-fusion/
│
│  ── White Paper ──
├── paper.qmd              # Main Quarto document
├── index.qmd              # Site landing page
├── references.bib         # BibTeX bibliography
├── _quarto.yml            # Quarto site and render config
├── custom.css             # Site styling
│
│  ── Source Code ──
├── src/
│   ├── models/
│   │   ├── fusion_model.py                  # Asymmetric cross-attention
│   │   └── benchmark_fusion_topologies.py   # Baseline comparison harness
│   ├── data/
│   │   ├── dataset.py                       # MONAI CacheDataset pipeline
│   │   ├── download_hf_embeddings.py        # HuggingFace embedding download
│   │   ├── format_cbioportal.py             # cBioPortal data formatting
│   │   └── generate_mock_data.py            # Synthetic data for testing
│   ├── utils/
│   │   ├── attention_analysis.py            # Top-1%-Mass Concentration metric
│   │   ├── gate_diagnostics.py              # Gate alpha distribution analysis
│   │   ├── balanced_subsample_diagnostic.py # Class-balance robustness check
│   │   ├── calibrate_gate_platt.py          # Platt calibration for gate output
│   │   ├── train_gated_fusion_anchor.py     # Direction anchor loss training
│   │   ├── train_gated_fusion_e2e.py        # End-to-end gate training
│   │   ├── train_gated_fusion_var_reg.py    # Variance regularization training
│   │   ├── run_mil_baselines.py             # ABMIL and CLAM-SB baselines
│   │   ├── run_validation_analysis.py       # Held-out cohort inference
│   │   └── generate_report_figures.py       # Figure generation
│   └── trainer.py                           # Training loop and checkpointing
│
│  ── Configuration ──
├── configs/
│   └── model_config.yaml  # Hyperparameters
├── Makefile               # Pipeline automation
├── requirements.txt       # Pinned Python dependencies
│
│  ── Reports and Assets ──
├── reports/
│   └── figures/           # Source figures (canonical location)
├── checkpoints/           # Model checkpoints (gitignored)
└── scripts/               # Utility shell scripts
```

---

## Quick Start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data and run training
make mock
make train

# 3. Preview the white paper locally
quarto preview
```

## Deploying the White Paper

The white paper deploys automatically to GitHub Pages on every
push to main via GitHub Actions. No manual deployment step
is required.

```bash
# To preview locally before pushing:
quarto preview
```

---

## License

MIT License. See [LICENSE](./LICENSE).
