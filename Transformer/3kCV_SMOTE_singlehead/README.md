# Transformer Classifier – Encoder–Decoder with SMOTE

## Overview

This repository provides a ready-to-use Transformer-based classifier for binary labeling of fixed-length, multichannel time-series windows (5s, 7s, 10s, or 15s).
Point `config.py` at a folder of pre-extracted `.xlsx` chunks, then run a single command to handle scaling, SMOTE oversampling (1:1), 3-fold stratified 
cross-validation, model training, and evaluation.The model uses sinusoidal positional encoding, multi-head self-attention, encoder–decoder blocks, and a softmax head to generate robust class predictions.

SMOTE synthetically oversamples the minority class to balance the training data. This helps the Transformer learn better decision boundaries when faced with 
imbalanced sensor recordings.

---

## Features

- Full encoder–decoder Transformer architecture with learnable start token  
- Sinusoidal positional encoding for timestep awareness  
- Multi-head self-attention and position-wise feed-forward layers  
- Global pooling and softmax classification head  
- Vanilla SMOTE oversampling in every training fold  
- Cosine-decay learning-rate schedule with restarts  
- Configurable scaling: RobustScaler (default) or StandardScaler via CLI  
- Modular settings for data paths, windowing, model hyperparameters, and training  

---

## Usage

```bash
# Default: RobustScaler
python run_transformer.py

# To use StandardScaler instead
python run_transformer.py standard
```

This single command executes:
1. Data loading, norm computation, and windowing  
2. Feature scaling (Robust or Standard)  
3. SMOTE oversampling on each training fold  
4. Model training across 3 folds with checkpointing  
5. Evaluation and artifact saving  

---

## Configuration Highlights (`config.py`)

Edit `config.py` to point to your data and tune key settings:

- **RAW_DATA_DIR**: Folder of pre-windowed `.xlsx` files  
- **TIMESTEPS**, **STRIDE**, **FEAT_COLS**: Window length, hop size, feature names  
- **D_MODEL**, **NUM_HEADS**, **DFF**, **NUM_ENC_LAYERS**, **NUM_DEC_LAYERS**, **DROPOUT**: Transformer dimensions  
- **EPOCHS**, **BATCH_SIZE**, **LR_SCHED**, **SEED**, **NUM_CLASSES**: Training parameters  

Switching between chunk durations requires only updating `RAW_DATA_DIR`. Use the `standard` argument to toggle scalers at runtime.

---

## Output Artifacts

Results are saved under:

```
./artifacts/artifacts_robust/    # for default RobustScaler
./artifacts/artifacts_standard/  # for StandardScaler
```

Each directory contains:

- **fold_<n>/**  
  - `best.keras`               (best model checkpoint)  
  - `history.json`             (training/validation metrics)  
  - `confusion.npy`            (validation confusion matrix)  
  - `classification_report.txt` (precision/recall/F1 details)  

- `cv_summary.txt`             (mean ± std of Accuracy, F1, AUROC, AUPRC)  
- `val_loss_cv.png`            (validation-loss curves)  
- `scaler.pkl`                 (saved scaler object)  

---

## Error Handling

- Raises a clear error if `RAW_DATA_DIR` is missing or contains no `.xlsx` files  
- Skips files shorter than `TIMESTEPS`, logging a warning  
- If SMOTE fails (e.g., single-class fold), training proceeds without oversampling  
- Checkpoints are written only when validation loss improves  

