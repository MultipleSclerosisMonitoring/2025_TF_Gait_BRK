# Transformer Classifier – Encoder–Decoder with SMOTEENN

## Overview  
This repository provides a turnkey Transformer-based classifier for binary labeling of fixed-length, multichannel time-series windows (5s, 7s, 10s, or 15s). 
Point `config.py` at a folder of pre-extracted `.xlsx` chunks and run a single script to perform scaling, SMOTEENN balancing, 3-fold cross-validation, 
model training, and evaluation. The model uses sinusoidal positional encoding, multi-head self-attention, feed-forward blocks, and global average pooling 
to output robust class predictions.

**Why SMOTEENN?**  
We combine SMOTE (to synthetically oversample the minority class) with Edited Nearest Neighbors (to remove noisy boundary samples). 
This two-step process yields a more balanced and cleaner training set, improving generalization on imbalanced sensor data.

---

## Features  
- Full encoder–decoder Transformer with three decoder blocks  
- Sinusoidal positional encoding for temporal order  
- SMOTEENN resampling in each CV fold for class balance  
- Configurable scaling: StandardScaler (default) or RobustScaler via CLI  
- Automatic 3-fold CV with checkpointing, metrics, and loss-curve plots

---

## Usage  
**One-Command Training**
```bash
# Default: StandardScaler
python run_transformer.py

# For RobustScaler:
python run_transformer.py robust
```

There’s no need to run other scripts manually—`run_transformer.py` handles:
- Data loading  
- Preprocessing  
- Model building  
- SMOTE application  
- Training across 3 CV folds  
- Evaluation and artifact saving  

---

## Configuration Highlights (`config.py`)  
- **RAW_DATA_DIR**: Path to folder of `.xlsx` chunks  
- **TIMESTEPS**, **STRIDE**, **FEAT_COLS**: Windowing parameters  
- **D_MODEL**, **NUM_HEADS**, **NUM_ENC_LAYERS**, **DROPOUT**: Transformer settings  
- **EPOCHS**, **BATCH_SIZE**, **SEED**: Training settings  
- **LR_SCHED**: CosineDecayRestarts learning-rate scheduler  

Only `RAW_DATA_DIR` needs updating to switch between chunk-size datasets; scaling method is chosen at runtime.

---

## Output Artifacts  
Results appear under `ARTIFACT_ROOT/artifacts_standard_cv3/` or `artifacts_robust_cv3/`:

- **Per-fold directories** (`fold_1`, `fold_2`, `fold_3`):  
  - `best.keras` (model checkpoint)  
  - `history.json` (training history)  
  - `confusion.npy` (confusion matrix)  
  - `classification_report.txt` (precision/recall/F1)  
- `cv_summary.txt` (aggregated Accuracy, F1, AUROC, AUPRC)  
- `val_loss_cv.png` (validation-loss curves)  

---

## Error Handling  
- Raises errors for missing or invalid `RAW_DATA_DIR`  
- Logs and skips files shorter than `TIMESTEPS`  
- Continues without SMOTEENN if resampling fails  
- Saves checkpoints only when validation loss improves  

---
