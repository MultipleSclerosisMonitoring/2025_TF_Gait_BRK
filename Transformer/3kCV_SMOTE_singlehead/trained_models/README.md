### Trained Model Artifacts

This directory holds the pre-trained variants of the classification-only Transformer, organized by chunk duration (5s, 7s, 10s, 15s) and Robust type.  

Each variant subfolder includes:  
- Fold-wise best model checkpoints (`.keras`)  
- Training histories (`history.json`)  
- Validation confusion matrices (`confusion.npy`)  
- Classification reports (`classification_report.txt`)  
- Cross-validation summary (`cv_summary.txt`) and loss-curve plot (`val_loss_cv.png`)  

Use these artifacts for rapid evaluation, compare results across configurations, or deploy the best-performing variant directly.
