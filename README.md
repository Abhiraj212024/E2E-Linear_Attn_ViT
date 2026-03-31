# End-to-End Deep Learning: Linear Attention Vision Transformers for Jet Regression & Classification

**XCiT (Cross-Covariance Image Transformers) with O(N) attention for jet image classification and simultaneous mass/pT regression**

## Overview

Two-phase training approach:
- **Phase 1:** MAE pretraining on 60K unlabelled 8-channel jet images
- **Phase 2:** Multitask fine-tuning for classification + mass & pT regression

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Option A: Use public [Kaggle dataset](https://www.kaggle.com/datasets/abhirajraje/jet-data-vit)

Option B: Build from raw H5 files:
```bash
python dataset_construction_scripts/regression_dataset.py
```

### 3. Test a Model
```bash
jupyter notebook testing.ipynb
```

## Using the Testing Notebook

The `testing.ipynb` notebook evaluates pretrained models on test data. To test your models. It is requested to use this notebook for evaluation because PyTorch does not inherently save Custom Models which needs to be defined for using stored models:

1. **Update configuration** in the first code cell:
   - `H5_PATH`: Path to your labelled dataset H5 file
   - `MODEL_PATH`: Path to your model weights (e.g., `models/linear-vit/best_xcit_mt_ft.pth`)
   - `NORM_STATS_PATH` : Path to the pre determined(by me) normalization info used for training the models.

2. **Run all cells** to generate:
   - Classification metrics (accuracy, F1, ROC-AUC)
   - Regression metrics (RMSE, MAE for mass & pT)
   - Confusion matrix, ROC curve, and prediction scatter plots

**Example configurations:**
```python
# Test multitask model with pretraining
MODEL_PATH = "models/linear-vit/best_xcit_mt_ft.pth"

# Test multitask model from scratch
MODEL_PATH = "models/linear-vit/best_xcit_mt_sc.pth"

# Test baseline ViT
MODEL_PATH = "models/baseline-vit/best_vit.pth"
```

## Training Notebooks

- **`vit-jet-images-mae.ipynb`** — Phase 1: MAE pretraining
- **`vit-multitask.ipynb`** — Phase 2: Multitask fine-tuning with frozen → unfrozen backbone

## Pre-trained Models

Available in `models/`:

| Model | Path | Task |
|-------|------|------|
| XCiT + MAE | `linear-vit/best_xcit_mae.pth` | Encoder for pretraining |
| XCiT + Multitask (pretrained) | `linear-vit/best_xcit_mt_ft.pth` | Classification + regression |
| XCiT + Multitask (from scratch) | `linear-vit/best_xcit_mt_sc.pth` | Classification + regression (baseline) |
| Standard ViT Baseline | `baseline-vit/best_vit.pth` | Comparison baseline |

## Dataset Details

**Public Dataset:** [Jet Data ViT on Kaggle](https://www.kaggle.com/datasets/abhirajraje/jet-data-vit)

**Preprocessing:**
- Log1p transformation + per-sample max normalization
- 80/20 train-test split (stratified by class)
- Targets normalized using training split statistics

**Output from dataset construction** (`data-concat/`):
- Train/test images: `X_train.npy`, `X_test.npy`
- Labels: `y_train.npy`, `y_test.npy`
- Mass targets: `mass_train_norm.npy`, `mass_test_norm.npy`
- pT targets: `pt_train_norm.npy`, `pt_test_norm.npy`
- Normalization stats: `norm_stats.npy` (for denormalization)
