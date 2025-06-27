"""
run_transformer.py

3-fold CV runner for full encoder–decoder Transformer classifier
with SMOTEENN balancing (no reconstruction head).

Usage:
  python run_transformer.py            # StandardScaler
  python run_transformer.py robust     # RobustScaler
"""

from __future__ import annotations
import sys
import json
import gc
import joblib
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score
)
from imblearn.combine import SMOTEENN

import config
from data_preprocessing import load_dataset
from transformer_model import build_encoder_decoder_classifier, ClassAccuracy


def main() -> None:
    """
    Execute 3-fold CV with SMOTEENN balancing.

    Steps:
      1. Load and scale data.
      2. Snapshot initial model weights.
      3. For each fold:
         - Balance with SMOTEENN.
         - Reset model weights.
         - Train for EPOCHS/3 epochs.
         - Evaluate and save metrics and artifacts.
      4. Aggregate metrics and plot validation-loss curves.
    """
    # Choose scaler
    tag = "robust" if (len(sys.argv) > 1 and sys.argv[1].lower().startswith("rob")) else "standard"
    run_dir = config.ARTIFACT_ROOT / f"artifacts_{tag}_cv3"
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Loading dataset ({tag} scaler)…")
    X, y, scaler = load_dataset(tag)
    joblib.dump(scaler, run_dir / f"scaler_{tag}.pkl")

    # 3-fold stratified CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    base_model = build_encoder_decoder_classifier(config.TIMESTEPS, X.shape[-1])
    init_w = base_model.get_weights()

    histories, accs, f1s, aurocs, auprcs = [], [], [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y.argmax(1)), start=1):
        print(f"\n── Fold {fold}/3 ──")
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        # Flatten for SMOTEENN
        n, T, F = X_tr.shape
        X_flat = X_tr.reshape(n, T * F)
        labels = np.argmax(y_tr, axis=1)

        if len(np.unique(labels)) > 1:
            sampler = SMOTEENN(sampling_strategy=1.0, random_state=config.SEED)
            Xr_flat, yr = sampler.fit_resample(X_flat, labels)
        else:
            print(f"Fold {fold}: only one class present, skipping SMOTEENN")
            Xr_flat, yr = X_flat, labels

        Xr = Xr_flat.reshape(-1, T, F)
        y_r = tf.keras.utils.to_categorical(yr, num_classes=config.NUM_CLASSES)

        # Build & compile
        model = build_encoder_decoder_classifier(config.TIMESTEPS, F)
        model.set_weights(init_w)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LR_SCHED),
            loss="categorical_crossentropy",
            metrics=[ClassAccuracy(config.NUM_CLASSES)]
        )

        # Directory & checkpoint
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            fold_dir / "best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        # Train
        hist = model.fit(
            Xr,
            y_r,
            validation_data=(X_va, y_va),
            epochs=int(config.EPOCHS / 3),
            batch_size=config.BATCH_SIZE,
            callbacks=[ckpt],
            verbose=2
        )
        histories.append(hist.history)

        # Evaluate
        probs = model.predict(X_va, batch_size=config.BATCH_SIZE)
        preds = probs.argmax(1)
        true = y_va.argmax(1)

        a = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="macro")
        au = roc_auc_score(true, probs[:, 1])
        ap = average_precision_score(true, probs[:, 1])

        accs.append(a)
        f1s.append(f1)
        aurocs.append(au)
        auprcs.append(ap)

        report = classification_report(true, preds, digits=4)
        (fold_dir / "classification_report.txt").write_text(report)
        (fold_dir / "history.json").write_text(json.dumps(hist.history))
        np.save(fold_dir / "confusion.npy", confusion_matrix(true, preds))

        print(report)
        print(f"Fold {fold}: Acc={a:.4f}, F1={f1:.4f}, AUROC={au:.4f}, AUPRC={ap:.4f}")

        tf.keras.backend.clear_session()
        gc.collect()

    # Aggregate and log summary
    import numpy as _np
    summary = (
        "\n── CV Summary ──\n"
        f"Accuracies: {accs}\n"
        f"Mean±Std: {_np.mean(accs):.4f}±{_np.std(accs):.4f}\n\n"
        f"F1 Scores: {f1s}\n"
        f"Mean±Std: {_np.mean(f1s):.4f}±{_np.std(f1s):.4f}\n\n"
        f"AUROC: {aurocs}\n"
        f"Mean±Std: {_np.mean(aurocs):.4f}±{_np.std(aurocs):.4f}\n\n"
        f"AUPRC: {auprcs}\n"
        f"Mean±Std: {_np.mean(auprcs):.4f}±{_np.std(auprcs):.4f}\n"
    )
    (run_dir / "cv_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # Plot validation-loss curves
    plt.figure(figsize=(7, 5))
    for i, h in enumerate(histories, start=1):
        plt.plot(h["val_loss"], label=f"fold{i}")
    plt.title("Validation Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "val_loss_cv.png")
    plt.close()

    print(f"[INFO] All artifacts saved in {run_dir.resolve()}")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(config.SEED)
    main()
