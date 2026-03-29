# E2E Linear Attention Vision Transformers

**GSoC 2024 Submission: End-to-End Classification and Regression using Linear Attention Vision Transformers**

## Overview

This project implements a two-phase approach to jet image classification and regression using XCiT (Cross-Covariance Image Transformers) with linear O(N) attention mechanisms:

- **Phase 1:** Masked Autoencoder (MAE) pretraining on 60,000 unlabelled 8-channel jet images
- **Phase 2:** Multitask fine-tuning for simultaneous classification and regression (mass & pT prediction)

The approach addresses challenges specific to sparse jet image data through patch-normalised targets, activity-weighted loss, and optimized mask ratios.

## Dataset

### Overview
The dataset consists of high-energy physics jet images with 8 channels, preprocessed for transformer-based learning.

### Public Kaggle Dataset
The processed dataset has been made publicly available on Kaggle:
- **Dataset Name:** [Jet Data ViT](https://www.kaggle.com/datasets/YOUR_KAGGLE_USERNAME/jet-data-vit)
- **Contents:** 
  - Unlabelled images for MAE pretraining (60,000 samples)
  - Labelled images for multitask learning (10,000 samples with class, mass, and pT labels)

### Dataset Construction

To recreate the dataset from raw data provided by evaluators, use the scripts in `dataset_construction_scripts/`:


#### Step 1: Regression Dataset (`regression_dataset.py`)
```bash
python regression_dataset.py
```

**Requirements:**
- Raw H5 file: `Dataset_Specific_labelled_full_only_for_2i.h5` (in the same directory or update the `file_path` variable)

**Output files** (saved to `data-concat/`):
- `X_train_mt.npy`, `X_test_mt.npy` — preprocessed images
- `y_train_mt.npy`, `y_test_mt.npy` — class labels
- `mass_train_norm.npy`, `mass_test_norm.npy` — normalized mass targets
- `mass_train_raw.npy`, `mass_test_raw.npy` — raw mass values
- `pt_train_norm.npy`, `pt_test_norm.npy` — normalized pT targets
- `pt_train_raw.npy`, `pt_test_raw.npy` — raw pT values
- `norm_stats.npy` — normalization statistics (mean, std for mass and pT)

**Processing:**
1. Loads images, class labels, mass, and pT from H5
2. Same image preprocessing as Phase 1 (log1p + per-sample max norm)
3. Stratified 80/20 split
4. Target normalization using training split statistics only
5. Saves normalization stats for inference

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0.0
- torchvision >= 0.15.0
- einops >= 0.7.0
- h5py >= 3.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- tqdm >= 4.65.0

3. **Prepare data:**
   - Download the public Kaggle dataset, OR
   - Run the dataset construction scripts with your raw H5 files

## Running the Notebooks

### Phase 1: MAE Pretraining
**Notebook:** `xcit-jet-images-mae.ipynb`

```bash
jupyter notebook xcit-jet-images-mae.ipynb
```

**Purpose:**
- Pretrain XCiT encoder on 60,000 unlabelled jet images (In this case, it was trained on 30,000 images due to computational constraints)
- Uses Masked Autoencoder approach with 60% masking ratio
- Sparse image-specific optimizations (patch-normalised targets, activity-weighted loss)

**Output:**
- Trained encoder weights: `models/linear-vit/best_xcit_mae.pth`
- Training logs and visualizations in `plots/linear-vit/`

**Key Hyperparameters:**
- Encoder: 192-dim, 8 blocks, 8 heads
- Decoder: 96-dim, 4 blocks (discarded after training)
- Training: 43 epochs, LR=5e-4, batch size=32
- Mask ratio: 60% (reduced from 75% to preserve signal in sparse images)

### Phase 2: Multitask Fine-tuning
**Notebook:** `xcit-multitask-mae.ipynb`

```bash
jupyter notebook xcit-multitask-mae.ipynb
```

**Purpose:**
- Fine-tune MAE-pretrained encoder on labelled data
- Simultaneous multitask learning: classification + mass regression + pT regression
- Two-stage training: frozen backbone → full network fine-tuning

**Output:**
- Fine-tuned models: 
  - `models/linear-vit/best_xcit_mt_ft.pth` — with pretraining
  - `models/linear-vit/best_xcit_mt_sc.pth` — from scratch (baseline)
- Training logs and performance metrics in `plots/linear-vit/`

**Training Stages:**
1. **Stage 1 (Frozen Backbone, 20 epochs):**
   - Freeze MAE encoder
   - Train 3 task heads: classifier (2-class), mass regressor, pT regressor
   - LR: 1e-3, batch size=64

2. **Stage 2 (Fine-tuning, 30 epochs):**
   - Unfreeze backbone
   - Backbone LR: 3e-5 (conservative learning rate)
   - Head LR: 1e-3

**Loss Weights:**
- Classification: 1.0 (cross-entropy)
- Mass regression: 1.0 (MSE)
- pT regression: 1.0 (MSE)
- Total: W_CLS·CE + W_MASS·MSE + W_PT·MSE

## Model Weights

Pre-trained model weights are available in `models/`:

### Baseline Models
- **`models/baseline-vit/best_vit.pth`** — Standard ViT baseline (for comparison)

### Linear Attention Models
- **`models/linear-vit/best_xcit_mae.pth`** — Phase 1: MAE-pretrained XCiT encoder
  - Dimensions: 192-dim embeddings, 8 transformer blocks
  - Ready for fine-tuning in Phase 2
  
- **`models/linear-vit/best_xcit.pth`** — Phase 2: Fine-tuned (pretrained + multitask)
  - Trained on 80% of labelled data
  - Best classification performance
  
- **`models/linear-vit/best_xcit_mt_ft.pth`** — Phase 2: Multitask with pretraining
  - Classification + regression (mass & pT)
  - Initialized from MAE pretraining
  
- **`models/linear-vit/best_xcit_mt_sc.pth`** — Phase 2: Multitask from scratch
  - Baseline: trained without pretraining
  - Demonstrates benefit of MAE pretraining
