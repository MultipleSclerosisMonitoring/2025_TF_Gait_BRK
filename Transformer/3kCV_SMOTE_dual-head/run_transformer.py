"""
run_transformer.py

3-fold cross-validation runner for the dual-head encoder-decoder Transformer
with baseline vanilla SMOTE (1:1 ratio) integrated.

Usage:
  python run_transformer.py            # StandardScaler, 3 folds
  python run_transformer.py robust     # RobustScaler, 3 folds
"""

from __future__ import annotations
import sys
import json
import gc
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

import config
from data_preprocessing import load_dataset
from transformer_model import build_transformer, DualLossConcat


class ClassAccuracy(tf.keras.metrics.Metric):
    """
    Custom Keras metric tracking only the classification accuracy slice.

    This metric ignores the reconstruction head and computes accuracy
    on the first `num_classes` outputs.

    Args:
        num_classes (int): Number of classes in the classification head.
        name (str): Name of the metric (defaults to 'class_acc').
        **kw: Additional keyword arguments passed to the base Metric.

    Attributes:
        num_classes (int): Stored number of classes.
        cat_acc (tf.keras.metrics.CategoricalAccuracy): Underlying accuracy metric.
    """

    def __init__(self, num_classes: int, name: str = "class_acc", **kw):
        super().__init__(name=name, **kw)
        self.num_classes = num_classes
        self.cat_acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None
    ) -> None:
        """
        Update internal state with a new batch of predictions.

        Args:
            y_true (tf.Tensor): Ground truth concatenated tensor of shape (batch, C+F).
            y_pred (tf.Tensor): Predicted concatenated tensor of same shape.
            sample_weight (tf.Tensor | None): Optional sample weights.
        """
        self.cat_acc.update_state(
            y_true[:, : self.num_classes],
            y_pred[:, : self.num_classes],
            sample_weight
        )

    def result(self) -> tf.Tensor:
        """
        Compute and return the classification accuracy.

        Returns:
            tf.Tensor: Scalar accuracy value.
        """
        return self.cat_acc.result()

    def reset_state(self) -> None:
        """
        Reset the metric's state for a new evaluation.
        """
        self.cat_acc.reset_state()


def main() -> None:
    """
    Execute 3-fold cross-validation training and evaluation.

    This script:
      1. Loads the dataset with chosen scaler.
      2. Builds a base Transformer model and snapshots initial weights.
      3. For each fold:
         - Applies SMOTE to balance classes.
         - Resets model weights.
         - Trains for EPOCHS/3.
         - Evaluates and logs metrics.
      4. Aggregates fold metrics and plots validation loss curves.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If data loading or windowing fails inside load_dataset.
    """
    # Determine scaler tag and create output directory.
    tag = "robust" if (len(sys.argv) > 1 and sys.argv[1].lower().startswith("rob")) else "standard"
    run_dir = config.ARTIFACT_ROOT / f"artifacts_{tag}_cv3"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1 — Data Loading
    print(f"[INFO] loading dataset ({tag} scaling)…")
    X, y_class, scaler = load_dataset(tag)
    joblib.dump(scaler, run_dir / "scaler.pkl")

    # Prepare reconstruction target (first timestep)
    y_recon = X[:, 0, :]
    y_concat = np.concatenate([y_class, y_recon], axis=-1)

    # 2 — Cross-validation setup
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    fold_histories: list[dict] = []
    fold_acc = []
    fold_f1 = []
    fold_auroc = []
    fold_auprc = []

    # Capture initial model weights for reinitialization each fold
    base_model = build_transformer(config.TIMESTEPS, X.shape[-1])
    init_weights = base_model.get_weights()

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(X, np.argmax(y_class, axis=1)),
        start=1
    ):
        print(f"\n——— Fold {fold}/3 ———")
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y_concat[train_idx], y_concat[val_idx]
        y_va_class = y_class[val_idx]

        # Apply vanilla SMOTE (1:1)
        y_labels = np.argmax(y_tr[:, : config.NUM_CLASSES], axis=1)
        n_tr, T, F = X_tr.shape
        X_tr_flat = X_tr.reshape(n_tr, T * F)
        smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=config.SEED)
        X_res_flat, y_res_labels = smote.fit_resample(X_tr_flat, y_labels)
        X_res = X_res_flat.reshape(-1, T, F)
        y_res_class = tf.keras.utils.to_categorical(y_res_labels, num_classes=config.NUM_CLASSES)
        y_res_recon = X_res[:, 0, :]
        y_tr_res = np.concatenate([y_res_class, y_res_recon], axis=-1)

        # Reset, compile, and train model for this fold
        model = build_transformer(config.TIMESTEPS, X.shape[-1])
        model.set_weights(init_weights)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LR_SCHED),
            loss=DualLossConcat(config.NUM_CLASSES),
            metrics=[ClassAccuracy(config.NUM_CLASSES)]
        )

        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=fold_dir / "best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        hist = model.fit(
            X_res, y_tr_res,
            validation_data=(X_va, y_va),
            epochs=int(config.EPOCHS / 3),
            batch_size=config.BATCH_SIZE,
            callbacks=[ckpt],
            verbose=2
        )

        # Evaluate on validation set
        preds = model.predict(X_va, batch_size=config.BATCH_SIZE)
        logits = preds[:, : config.NUM_CLASSES]

        if config.NUM_CLASSES > 1:
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            y_pred = np.argmax(probs, axis=1)
        else:
            probs = tf.nn.sigmoid(logits).numpy()
            y_pred = (probs[:, 0] >= 0.5).astype(int)

        y_true = np.argmax(y_va_class, axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        auroc = roc_auc_score(y_true, probs[:, 1])
        auprc = average_precision_score(y_true, probs[:, 1])

        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auroc.append(auroc)
        fold_auprc.append(auprc)

        print(classification_report(y_true, y_pred))
        print(f"Fold {fold}: Accuracy={acc:.4f}, F1={f1:.4f}, "
              f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

        # Save fold artifacts
        json.dump(hist.history, open(fold_dir / "history.json", "w"))
        np.save(fold_dir / "confusion.npy", confusion_matrix(y_true, y_pred))
        fold_histories.append(hist.history)

        tf.keras.backend.clear_session()
        gc.collect()

    # Aggregate and report cross-validation metrics
    mean_acc, std_acc = np.mean(fold_acc), np.std(fold_acc)
    mean_f1, std_f1 = np.mean(fold_f1), np.std(fold_f1)
    mean_auroc, std_auroc = np.mean(fold_auroc), np.std(fold_auroc)
    mean_auprc, std_auprc = np.mean(fold_auprc), np.std(fold_auprc)

    summary = (
        "\n——— Fold Summary ———\n"
        f"Accuracies: {fold_acc}\n"
        f"Mean±Std Accuracy: {mean_acc:.4f}±{std_acc:.4f}\n\n"
        f"F1 Scores: {fold_f1}\n"
        f"Mean±Std F1: {mean_f1:.4f}±{std_f1:.4f}\n\n"
        f"AUROC: {fold_auroc}\n"
        f"Mean±Std AUROC: {mean_auroc:.4f}±{std_auroc:.4f}\n\n"
        f"AUPRC: {fold_auprc}\n"
        f"Mean±Std AUPRC: {mean_auprc:.4f}±{std_auprc:.4f}\n"
    )
    print(summary)

    with open(run_dir / "cv_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary)

    # Plot validation loss curves
    plt.figure(figsize=(7, 5))
    for f, h in enumerate(fold_histories, 1):
        plt.plot(h["val_loss"], label=f"fold{f}")
    plt.title("Validation Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "val_loss_cv.png")
    plt.close()

    print(f"[INFO] Summary and curves saved in {run_dir.resolve()}")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(config.SEED)
    main()
