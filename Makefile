# ==============================================================================
# Pathology-Genomic Fusion Pipeline Automation
# ==============================================================================

.PHONY: help mock patch genomics extract_all fuse evaluate clean paper publish train

# Default target when just running 'make'
help:
	@echo ""
	@echo "  White paper"
	@echo "  -----------"
	@echo "  make paper       - Render paper.qmd to _site/ (local preview)"
	@echo "  make publish     - Render and push to GitHub Pages (gh-pages branch)"
	@echo ""
	@echo "  ML pipeline"
	@echo "  -----------"
	@echo "  make mock        - Generate synthetic patch embeddings and genomic counts"
	@echo "  make train       - Run the training loop"
	@echo "  make patch       - Run WSI segmentation and tissue patching"
	@echo "  make genomics    - Normalize and scale raw RNA-Seq expression profiles"
	@echo "  make extract_all - Extract histopathology and genomic embeddings"
	@echo "  make fuse        - Execute Multi-Head Cross-Attention patient fusion"
	@echo "  make evaluate    - Train and evaluate downstream cross-validation models"
	@echo "  make pipeline    - Run the entire end-to-end pipeline sequentially"
	@echo ""
	@echo "  Maintenance"
	@echo "  -----------"
	@echo "  make clean       - Flush generated logs, cache, and interim arrays"
	@echo ""

# White paper targets
paper:
	quarto render paper.qmd

publish:
	quarto publish gh-pages

# Training shortcut (uses the active venv)
train:
	.venv/bin/python src/trainer.py

# Synthetic data generation for pipeline validation
mock:
	.venv/bin/python src/data/generate_mock_data.py

# Stage 1a: WSI Tiling & Segmentation
patch:
	python src/data/patch_wsi.py \
		--wsi_dir ./data/raw/wsis/ \
		--patch_dir ./data/interim/patches/ \
		--patch_size 224 \
		--magnification 20 \
		--seg_threshold 0.15 \
		--num_workers 8

# Stage 1b: Genomic Profile Normalization
genomics:
	python src/data/preprocess_genomics.py \
		--counts_path ./data/raw/counts.csv \
		--output_path ./data/processed/genomics_scaled.parquet \
		--variance_filter 2000

# Stage 2: Parallel Feature Extraction
extract_all:
	@echo "Starting parallel feature extraction..."
	python src/features/extract_tissue_features.py \
		--patch_dir ./data/interim/patches/ \
		--output_dir ./data/processed/image_embeddings/ \
		--model_name virchow \
		--batch_size 256 \
		--device cuda & \
	python src/features/extract_genomic_features.py \
		--genomics_path ./data/processed/genomics_scaled.parquet \
		--output_dir ./data/processed/genomic_embeddings/ \
		--latent_dim 512 & \
	wait
	@echo "All feature extractions complete."

# Stage 3: Multi-Head Cross-Attention Fusion
fuse:
	python src/models/run_fusion.py \
		--image_emb_dir ./data/processed/image_embeddings/ \
		--genomic_emb_dir ./data/processed/genomic_embeddings/ \
		--config configs/model_config.yaml \
		--output_dir ./data/fused_patient_representations/

# Stage 4: Downstream Task Evaluation (Defaults to Survival)
evaluate:
	python src/train.py \
		--task survival \
		--data_dir ./data/fused_patient_representations/ \
		--clinical_path ./data/raw/clinical_metadata.csv \
		--epochs 50 \
		--lr 0.0001 \
		--cv_folds 5

# Full End-to-End Execution Pipeline
pipeline: patch genomics extract_all fuse evaluate

# Environment and Cache Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf ./data/interim/patches/*
	rm -rf ./data/processed/image_embeddings/*
	rm -rf ./data/processed/genomic_embeddings/*
	rm -rf ./data/fused_patient_representations/*
	@echo "Workspace cleaned successfully."
