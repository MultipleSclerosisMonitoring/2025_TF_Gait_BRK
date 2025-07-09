### Pre-Trained Dual-Head Transformer Variants

This directory holds the finalized artifacts for the dual-head encoder–decoder Transformer (classification + reconstruction) trained with SMOTE oversampling. Models are organized by chunk duration (5 s, 7 s, 10 s, 15 s) and by scaling method. RobustScaler runs are provided as default; StandardScaler versions are available alongside them.

Inside each variant folder you’ll find:

- `fold_<n>/best.keras` — Best checkpoint per fold  
- `fold_<n>/history.json` — Training and validation metrics per epoch  
- `fold_<n>/confusion.npy` — Confusion matrix on validation data  
- `fold_<n>/classification_report.txt` — Precision/recall/F1 breakdown  
- `cv_summary.txt` — Aggregated Accuracy, F1, AUROC, AUPRC (mean ± std)  
- `val_loss_cv.png` — Validation-loss curves across folds  

Use these artifacts to review performance across chunk sizes and scaling choices, or to deploy the most suitable dual-head model variant.
