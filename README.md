# Pathology-Genomic Fusion: Multimodal Foundation Models for Oncology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: MONAI](https://img.shields.io/badge/Framework-MONAI-green.svg)](https://monai.io/)

## 🧬 Project Overview

This project implements a multimodal deep learning framework designed to predict oncology outcomes by fusing **Whole Slide Images (WSI)** with **High-Dimensional Transcriptomic (RNA-Seq) data**. 

By leveraging **Pathology Foundation Models** (Virchow/UNI) and **Cross-Modal Attention**, the system identifies morphological features that are biologically grounded in molecular signatures, moving beyond "black-box" pixel classification toward interpretable precision diagnostics.

---

## 🏗️ Computational Architecture

The model is architected as a dual-stream encoder network followed by a transformer-based fusion bottleneck.

### 1. Imaging Stream: Morphological Encoding
*   **Input:** Gigapixel WSI patches ($224 \times 224$ pixels) at 20x magnification.
*   **Encoder:** Frozen SOTA Vision Transformer (ViT-L/14) Foundation Model (e.g., Virchow).
*   **Process:** WSIs are tiled into patches; the encoder generates a $D$-dimensional embedding for each patch, representing local tissue morphology.

### 2. Genomic Stream: Molecular Encoding
*   **Input:** Normalized bulk RNA-Seq counts (TPM/FPKM).
*   **Encoder:** Multi-Layer Perceptron (MLP) or Transcriptomic Transformer.
*   **Process:** High-dimensional gene expression vectors are projected into a latent space of the same dimensionality ($D$) as the image embeddings to facilitate cross-modal alignment.

### 3. Fusion Stream: Cross-Attention Transformer
*   **Mechanism:** Multi-Head Cross-Attention (MHCA).
*   **The Logic:**
    *   **Query (Q):** Derived from the Genomic Embedding.
    *   **Keys (K) & Values (V):** Derived from the Image Patch Embeddings.
*   **Outcome:** The model attends to specific morphological patches that are most predictive given the patient's unique genomic profile, outputting a fused "Patient Embedding."

---

## 🚀 Technical Highlights

*   **Foundation Model Backbone:** Leverages representations learned from 100M+ tissue patches.
*   **MONAI Ecosystem:** Advanced medical imaging transforms and dictionary-based data handling.
*   **Feature Caching:** Pre-extracting FM embeddings to reduce training time and enable rapid experimentation on local hardware.
*   **Interpretability:** Integrated Attention Rollout to visualize morphological-genomic correlations.

---

## 📁 Repository Structure

```text
patho-genomic-fusion/
├── configs/
│   └── model_config.yaml          # Cross-attention dims, training hyperparameters
├── data/                          # Generated data — gitignored, reproduced via make mock
│   ├── raw/                       # counts.csv (RNA-Seq), clinical metadata
│   ├── interim/patches/           # Tiled WSI patches from Stage 1
│   └── processed/                 # Image embeddings (.pt), scaled genomics (.parquet)
├── checkpoints/                   # Saved model weights — gitignored
├── src/
│   ├── data/
│   │   ├── dataset.py             # MONAI CacheDataset + Compose transform pipeline
│   │   └── generate_mock_data.py  # Synthetic patch embeddings & RNA-Seq counts
│   ├── models/
│   │   └── fusion_model.py        # PathoGenomicFusionModel (cross-attention nn.Module)
│   └── trainer.py                 # Training loop, MONAI ROCAUCMetric, checkpointing
├── notebooks/                     # Exploratory analysis
├── scripts/                       # Utility shell scripts
├── Makefile                       # Pipeline automation (mock, patch, fuse, evaluate…)
├── Dockerfile                     # Production-ready environment
└── requirements.txt               # Pinned dependencies (numpy, pandas, torch, monai)
```
## 🚀 Quick Start

1.  🛠️ **Setup:** Follow the [Installation Guide](./INSTALL.md) to prepare your environment.
2.  🏃 **Run:** See the [Usage Guide](./USAGE.md) to start the feature extraction and training pipeline.

## ⚖️ License
Distributed under the MIT License. See LICENSE for more information.

