# Transformer Model – Dual-Head Encoder–Decoder

## Overview

This project implements a dual-purpose Transformer network that simultaneously performs **classification** and **feature reconstruction** on windowed time-series data. You supply pre-extracted fixed-length chunks (5s, 7s, 10s, or 15s) labeled with movement types; the model learns to predict the movement category and reconstruct the original signal features from each chunk.

A 3-fold stratified cross-validation ensures robust evaluation, and SMOTE oversampling combats class imbalance. You can choose between **StandardScaler** (default) and **RobustScaler** at runtime—no other steps required.

---

## Features

- **Dual-Head Output**  
  Predicts movement class (softmax) and reconstructs the original signal (linear), trained with a combined loss function.

- **Homoscedastic Uncertainty Loss**  
  Automatically learns optimal weighting between classification and reconstruction loss components.

- **Cross-Validation & SMOTE**  
  3-fold CV with per-fold SMOTE oversampling (1:1 class ratio) during training.

- **Cosine Decay Restart Scheduler**  
  Enables periodic restarts in the learning rate to improve convergence and generalization.

- **Plug-and-Play Configuration**  
  Easily customize hyperparameters and data paths via `config.py`.

- **Multiple Chunk Durations**  
  Trained on independent datasets created from 5s, 7s, 10s, and 15s time chunks each with Standard and Robust scalers.

---

## One-Command Training

```bash
# Default: StandardScaler
python run_transformer.py

# For RobustScaler:
python run_transformer.py robust
```

There’s no need to run other scripts manually `run_transformer.py` handles:
- Data loading  
- Preprocessing  
- Model building  
- SMOTE application  
- Training across 3 CV folds  
- Evaluation and artifact saving  

---

## Configuration (`config.py`)

Parameters are managed centrally in `config.py`. Be sure to update this file before training.

###  Must-Set Path

```python
RAW_DATA_DIR = Path("path/to/chunked_xlsx_folder")
```

Point this to the appropriate dataset folder e.g., `\data_extraction\chunks_5s`, `\data extraction_\chunks_7s`, etc.

###  Core Parameters

- **Chunk Feature Columns:**  
  `FEAT_COLS = ["A", "G", "M", "S0", "S1", "S2"]`

- **Windowing:**  
  `TIMESTEPS = 256`, `STRIDE = 128`

- **Transformer Hyperparameters:**  
  - `D_MODEL = 128`  
  - `NUM_HEADS = 4`  
  - `NUM_ENC_LAYERS = 3`, `NUM_DEC_LAYERS = 3`  
  - `DROPOUT = 0.1`  

- **Training Settings:**  
  - `EPOCHS = 120`  
  - `BATCH_SIZE = 64`  
  - `SEED = 42`  

- **Learning Rate Schedule:**  
  Cosine decay with restarts (`CosineDecayRestarts` from TensorFlow)

- **Loss Weights:**  
  Automatically learned via log-variance parameters in the `DualLossConcat` custom loss

---

## Output & Evaluation

After each run, results are saved under:

```plaintext
./artifacts/artifacts_standard_cv3/
./artifacts/artifacts_robust_cv3/
```

Each folder contains:
- Best model checkpoint (`.keras`) per fold
- Training history (`history.json`)
- Confusion matrix (`confusion.npy`)
- Fold summaries with Accuracy, F1, AUROC, AUPRC
- Cross-validation summary (`cv_summary.txt`)
- Validation loss curve plot (`val_loss_cv.png`)

---

## Error Handling

- Invalid or missing input paths trigger clear exceptions  
- SMOTE edge cases (e.g., small minority class) are logged without interruption  
- Checkpoints saved only when validation improves  

---
