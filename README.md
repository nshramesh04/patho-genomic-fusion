# Asymmetric Cross-Attention Networks for Multimodal Integration in Pathology

[![White Paper](https://img.shields.io/badge/White%20Paper-GitHub%20Pages-blue)](https://nshramesh04.github.io/patho-genomic-fusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> **Read the full white paper:** [nshramesh04.github.io/patho-genomic-fusion](https://nshramesh04.github.io/patho-genomic-fusion)

PR-status classification in breast cancer by fusing RNA-Seq transcriptomics with whole-slide imaging (WSI) via a novel O(N) asymmetric cross-attention architecture validated on 951 TCGA-BRCA patients.

---

## Results

| Metric | Value | vs. Baseline |
|:---|:---:|:---:|
| ROC-AUC | **0.8426** | +0.0745 |
| PR-AUC | **0.9110** | +0.0142 |
| Brier Score | **0.17** | −0.04 |
| Cohort | N = 951 | TCGA-BRCA |
| Interpretability ($p$-value) | **0.018** | entropy: 0.401 |

---

## Key Contributions

- **O(N) asymmetric cross-attention** — resolves the structural incompatibility between fixed-length genomic vectors and variable-length WSI patch bags (960–39,052 patches per slide) without positional encodings or bag compression.
- **Top-1%-Mass Concentration metric** — a slide-size-invariant alternative to Shannon entropy; recovers *p* = 0.018 phenotypic separation that entropy analysis entirely suppresses (*p* = 0.401) due to the slide-size confound.
- **Cross-Modal Reliability Estimator** (Gated-Fusion v2) — adds a per-patient α ∈ (0,1) signal for automated triage of IHC-discordant borderline cases with 1,025 parameter overhead and no complexity cost.
- **+0.0745 ROC-AUC** over the late-fusion baseline, attributable to genomic conditioning applied *upstream* of visual compression — at the search stage, not the pooling stage.

---

## Repository Structure

```text
patho-genomic-fusion/
│
│  ── White Paper (single source of truth) ──
├── paper.qmd              # Master Quarto document
├── index.qmd              # Site landing page
├── references.bib         # BibTeX bibliography (Zotero-managed)
├── _quarto.yml            # Quarto site + render config
├── figures/               # Symlink → reports/figures/
│
│  ── Source Code ──
├── src/
│   ├── models/
│   │   ├── fusion_model.py                  # Asymmetric cross-attention nn.Module
│   │   └── benchmark_fusion_topologies.py   # Ablation harness
│   ├── data/
│   │   ├── dataset.py                       # MONAI CacheDataset pipeline
│   │   └── generate_mock_data.py            # Synthetic data for CI validation
│   ├── utils/
│   │   ├── attention_analysis.py            # Top-1%-Mass Concentration metric
│   │   ├── generate_report_figures.py       # Reproducible figure generation
│   │   └── run_validation_analysis.py       # Held-out cohort inference
│   └── trainer.py                           # Training loop + checkpointing
│
│  ── Configuration & Automation ──
├── configs/
│   └── model_config.yaml  # Hyperparameters
├── Makefile               # Pipeline automation (mock, train, paper, publish)
├── Dockerfile             # Reproducible environment
├── requirements.txt       # Pinned Python dependencies
│
│  ── Reports & Assets ──
├── reports/
│   └── figures/           # Source figures (PNG) — canonical location
├── notebooks/             # Exploratory analysis
└── scripts/               # Utility shell scripts
```

> **Architecture note:** `paper.qmd` is the single source of truth. `README.md` is a navigational landing page only — do not duplicate content here.

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

```bash
# Renders and pushes to the gh-pages branch automatically
quarto publish gh-pages
```

GitHub Pages is served from the `gh-pages` branch. No build artifacts are committed to `main`.

---

## License

MIT License. See [LICENSE](./LICENSE).
