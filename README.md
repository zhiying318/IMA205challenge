# IMA205 Challenge — White Blood Cell Classification

Author: Zhiying ZOU

13-class WBC classification Kaggle challenge for course IMA205, year 2025-26. The dataset is heavily imbalanced: SNE makes up ~45% of training data while PLY has only 11 samples. Two pipelines are developed in parallel: a hand-crafted feature + XGBoost baseline, and a CNN ensemble (ResNet-50, EfficientNet-B3, ConvNeXt-Large) trained with Focal Loss and k-fold CV.

**Best result**: 10-model ensemble (5× EfficientNet-B3 + 5× ConvNeXt-Large) with 8-view TTA and F1-weighted averaging.

---

## Environment Setup

Python 3.10 is required. The steps below create a virtual environment called `.ima205` inside the project folder, matching the exact package versions used during development.

```bash
cd IMA205-challenge

# Create and activate the virtual environment
python3.10 -m venv .ima205
source .ima205/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 build — adjust the index URL for other CUDA versions)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install the remaining dependencies
pip install -r requirements.txt
```

> If you are on a CPU-only machine or a different CUDA version, replace the `+cu118` suffix accordingly and use the matching `--index-url`. See https://pytorch.org/get-started/locally/ for the right command.

To reactivate the environment in a new terminal:

```bash
source IMA205-challenge/.ima205/bin/activate
```

---

## Repository Layout

```
IMA205-challenge/
│
├── .ima205/                      # Virtual environment (not committed)
├── requirements.txt              # Pinned dependencies
│
├── train/                        # Raw training images
├── test/                         # Raw test images
├── train_metadata.csv            # ID, label columns
├── test_metadata.csv             # ID column only
│
├── data_cropped/
│   ├── train/                    # Cropped training images, organized by class (ResNet)
│   ├── test/                     # Cropped test images (flat)
│   ├── train_eff/                # Same crop, separate folder for EfficientNet/ConvNeXt
│   └── test_eff/
│
├── CNN_models/                   # ResNet-50 pipeline
│   ├── preprocess.py             # Offline crop and save
│   ├── dataset.py                # DataLoader with class-conditional augmentation
│   ├── model.py                  # ResNet-18 / ResNet-50 definitions
│   ├── train.py                  # Single-split training
│   ├── predict.py                # TTA inference → submission CSV
│   └── confusion_matrix/
│
├── CNN_models_efficientnet/      # EfficientNet-B3 and ConvNeXt-Large pipeline
│   ├── preprocess.py
│   ├── dataset.py                # Same design + get_loaders_fold for k-fold
│   ├── model.py                  # EfficientNet-B3/B4 definitions
│   ├── train.py                  # K-fold training; toggle use_convnext flag
│   ├── predict.py                # Single-model TTA inference
│   ├── inference.py              # 10-model ensemble (simple average)
│   ├── inference_add_weights.py  # 10-model ensemble (F1-weighted) + post-processing
│   └── confusion_matrix_*/
│
├── submission_ensemble/          # Weighted ensemble submission files
├── visualisation.ipynb           # Original vs cropped images + class distribution
├── pipeline_v2_wbc_crop_aug_oversample.ipynb   # XGBoost pipeline notebook
└── submission_notebook.ipynb     # Clean submission notebook (both pipelines)
```

---

## Class Distribution (Training Set)

| Class | Count | Class | Count |
|-------|-------|-------|-------|
| SNE   | 13015 | MO    | 2746  |
| LY    | 8101  | BL    | 2012  |
| EO    | 861   | MY    | 441   |
| BA    | 415   | BNE   | 391   |
| VLY   | 366   | MMY   | 360   |
| PMY   | 114   | PC    | 68    |
| PLY   | 11    |       |       |

---

## Data Preprocessing

### WBC Crop

All images are cropped around the nucleus before training. The function `apply_wbc_crop_robust` in `preprocess.py`:

1. Converts to HSV and thresholds on saturation > 50 and value < 190 to isolate the stained nucleus
2. Applies morphological open + close to remove noise
3. Finds the connected component closest to the image center (ignoring components < 400 px²)
4. Crops a fixed 280×280 square centered on the nucleus centroid, shifting the window inward if it would fall outside image bounds

### Offline Preprocessing (CNN pipelines)

Run once before training. Images are saved to `data_cropped/` in `ImageFolder` format (class subdirectories for train, flat for test).

```bash
cd IMA205-challenge/CNN_models
python preprocess.py
```

This generates `data_cropped/train/` and `data_cropped/test/`. For EfficientNet/ConvNeXt, run the same script from `CNN_models_efficientnet/` to populate `data_cropped/train_eff/` and `data_cropped/test_eff/`.

---

## CNN Models

### ResNet-50

- Input: 256×256
- Head: Dropout(0.5) + Linear(2048 → 13)
- Loss: CrossEntropyLoss with √-smoothed class weights + label smoothing 0.1
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-2)
- Scheduler: ReduceLROnPlateau on macro-F1 (patience=2, factor=0.5)
- Training: single 80/20 split, 50 epochs

### EfficientNet-B3

- Input: 300×300 (native resolution)
- Head: Dropout(0.5) + Linear → 13
- Loss: Focal Loss (γ=2, label_smoothing=0.1)
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-2)
- Training: 5-fold stratified CV, 60 epochs per fold, AMP enabled
- Checkpoints: `best_efficientnet_fold{0..4}.pth`

### ConvNeXt-Large

- Input: 224×224, loaded via `timm` (`convnext_large`)
- Loss / Optimizer: same as EfficientNet-B3 but lr=5e-5
- Training: same 5-fold setup, AMP mandatory due to VRAM
- Checkpoints: `best_convnext_fold{0..4}.pth`

### Augmentation Strategy

Minority classes (BNE, MMY, PC, PLY, PMY, VLY) receive `RandAugment(num_ops=2, magnitude=9)`. All other classes get standard flips + RandomRotation(20°) + mild ColorJitter. Both groups are fed through a `WeightedRandomSampler` to balance per-batch class distribution.

---

## Training

### ResNet-50

```bash
source .ima205/bin/activate
cd CNN_models
python train.py
```

Saves `best_wbc_model.pth` when validation macro-F1 improves.

### EfficientNet-B3 (5-fold)

Edit the `__main__` block at the bottom of `CNN_models_efficientnet/train.py`:

```python
if __name__ == "__main__":
    train(use_convnext=False)   # EfficientNet-B3
```

Then:

```bash
source .ima205/bin/activate
cd CNN_models_efficientnet
python train.py
```

Saves `best_efficientnet_fold{k}.pth` for k in 0..4. Logs go to `training_efficientnet_kfold.log`.

### ConvNeXt-Large (5-fold)

```python
if __name__ == "__main__":
    train(use_convnext=True)    # ConvNeXt-Large
```

```bash
source .ima205/bin/activate
cd CNN_models_efficientnet
python train.py
```

Saves `best_convnext_fold{k}.pth` for k in 0..4. Logs go to `training_convnext_2_kfold.log`.

> Both architectures share the same `train.py` — only the flag differs. Training them simultaneously requires two separate terminal sessions.

To follow training progress live:

```bash
tail -f training_efficientnet_kfold.log
tail -f training_convnext_2_kfold.log
```

---

## Inference

### ResNet-50 (4-view TTA)

```bash
cd CNN_models
python predict.py
```

Averages softmax probabilities over: original, h-flip, v-flip, both flips.  
Output: `submission_resnet50_tta.csv`

### EfficientNet-B3 single model (8-view TTA)

```bash
cd CNN_models_efficientnet
python predict.py
```

8 views: 4 rotation angles (0°, 90°, 180°, 270°) × {original, h-flip}.  
Output: `submission_efficientnet_b3_focalloss_RandAug_Mixup_tta.csv`

### 10-model Ensemble (simple average)

```bash
cd CNN_models_efficientnet
python inference.py
```

Loads all 10 fold checkpoints, applies 8-view TTA per model, averages probabilities.  
Output: `submission_ensemble_{N}model_tta8.csv`

### 10-model Ensemble (F1-weighted + post-processing)

```bash
cd CNN_models_efficientnet
python inference_add_weights.py
```

Each model's probability vector is weighted by its validation macro-F1 (softmax-normalized, temperature=0.5). Post-processing rules handle the two biggest confusion pairs:
- **BNE↔SNE**: if BNE confidence < 0.45 and SNE > 0.15, boost SNE by ×1.5
- **VLY↔LY**: if VLY confidence < 0.40 and LY > 0.20, boost LY by ×1.3

Outputs both the post-processed and raw versions to `submission_ensemble/`.

---

## XGBoost Baseline

See `pipeline_v2_wbc_crop_aug_oversample.ipynb` or the clean version in `submission_notebook.ipynb` (Part 1).

Key design choices:
- ~718-dimensional hand-crafted features per image: two-level Otsu segmentation → nucleus shape, HSV/Lab color stats, LBP texture, N/C ratio, global HOG and Hu moments
- Minority classes oversampled to 600 samples before feature extraction
- 10 augmentation views per training image (rotation, flips, HSV jitter, blur, noise, gamma, translation)
- StandardScaler + XGBoost(n_estimators=500, max_depth=6) with balanced sample weights
- 5-fold stratified CV: mean macro-F1 ≈ 0.55
