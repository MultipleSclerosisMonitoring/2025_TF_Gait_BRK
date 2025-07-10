# Baseline CNN+LSTM Classifier with SMOTE

## Overview

This repository implements a baseline convolutional–LSTM network for binary classification of fixed-length, multichannel time-series sessions. 
Each session is trimmed to a common length, scaled with RobustScaler (default), and training data are balanced via SMOTE oversampling. The model 
combines convolutional feature extraction, LSTM temporal modeling, and dense layers to predict movement categories from sensor norms.

---

## Features

- Three stacked Conv1D blocks with pooling and batch normalization  
- Two-layer LSTM with dropout for sequence modeling  
- Dense classification head with softmax activation  
- Robust feature scaling to mitigate outliers  
- SMOTE oversampling (1:1 ratio) on the training split  
- ReduceLROnPlateau callback for dynamic learning-rate adjustment  

---

## Usage

```bash
python run_cnn_lstm.py --data_dir /path/to/xlsx_folder
```

- `--data_dir`: directory containing raw `.xlsx` session files

This single command executes:
1. Data loading and norm computation  
2. Session length unification and timestamp extraction  
3. Robust scaling of all features  
4. SMOTE on training data  
5. Model training with validation split and learning-rate scheduling  
6. Artifact saving and evaluation  

---

## Configuration Highlights

Located at the top of `run_cnn_lstm.py`:

- **EPOCHS**: 100  
- **BATCH_SIZE**: 128  
- **LR**: 3e-4  
- **PATIENCE_LR**: 8 (for ReduceLROnPlateau)  
- **TEST_SPLIT**: 0.2  
- **SEED**: 42  
- **FINAL_COLUMNS**: ["A", "G", "M", "S0", "S1", "S2"]  
- **EXPECTED_RAW**: raw sensor columns for norm calculation  

No additional files need editing—just point `--data_dir` at your chunk folder.

---

## Output Artifacts

Results are saved under `./artifacts_cnn_lstm/`:

- `model.keras`            – Trained model weights  
- `scaler.joblib`          – Fitted RobustScaler  
- `history.json`           – Training and validation loss/accuracy per epoch  
- `plt_cnn_lstm.png`       – Loss and accuracy plots  
- `cm.npy` & `cm.png`      – Confusion matrix array and heatmap  
- `classification.txt`     – Detailed precision/recall/F1 report  

---

## Error Handling

- Files missing a valid timestamp column are skipped with a log entry  
- Sessions with too many missing raw columns (>3) are skipped  
- Raises an error if no valid `.xlsx` files are found in `--data_dir`  

---

## Module Structure

All functionality is contained in `run_cnn_lstm.py`, which comprises:

1. **Data Loader**  
   Reads Excel files, computes Euclidean norms, unifies session lengths, scales features, and applies SMOTE.

2. **Model Builder**  
   Defines and compiles the Conv1D–LSTM architecture with specified hyperparameters.

3. **Training & Evaluation**  
   Splits data, trains the model with callbacks, evaluates on the validation set, and saves all artifacts.
