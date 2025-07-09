"""
run_transformer.py

3-fold CV runner for the encoder-decoder Transformer (classification only)
with vanilla SMOTE (1:1 ratio) integrated.

Usage:
  python run_transformer.py            # RobustScaler (default), 3 folds
  python run_transformer.py standard   # StandardScaler, 3 folds
"""

import sys, gc, json, joblib
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE

import config
from data_preprocessing import load_dataset
from transformer_model import build_transformer_classifier  # classification‐only

def main() -> None:
    # choose scaler
    tag = "standard" if (len(sys.argv) > 1 and sys.argv[1].lower().startswith("std")) else "robust"
    run_dir = config.ARTIFACT_ROOT / f"artifacts_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print(f"[INFO] loading dataset ({tag} scaling)…")
    X, y_onehot, scaler, feat_names = load_dataset(tag)
    joblib.dump(scaler, run_dir / "scaler.pkl")

    # Extract raw labels for SMOTE
    y_raw = np.argmax(y_onehot, axis=1)

    # 2) CV setup
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    fold_histories, fold_acc, fold_f1, fold_auroc, fold_auprc = [], [], [], [], []

    # snapshot initial weights
    base = build_transformer_classifier(config.TIMESTEPS, X.shape[-1])
    init_w = base.get_weights()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_raw), start=1):
        print(f"\n——— Fold {fold}/3 ———")

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_raw, y_va_raw = y_raw[tr_idx], y_raw[va_idx]
        y_va = tf.keras.utils.to_categorical(y_va_raw, num_classes=config.NUM_CLASSES)

        # SMOTE on training set
        n_tr, T, F = X_tr.shape
        X_flat = X_tr.reshape(n_tr, T*F)
        sm = SMOTE(sampling_strategy=1.0, random_state=config.SEED)
        X_res_flat, y_res_raw = sm.fit_resample(X_flat, y_tr_raw)
        X_res = X_res_flat.reshape(-1, T, F)
        y_res = tf.keras.utils.to_categorical(y_res_raw, num_classes=config.NUM_CLASSES)

        # per-fold directory
        fd = run_dir / f"fold_{fold}"
        fd.mkdir(exist_ok=True)

        # build & compile
        model = build_transformer_classifier(config.TIMESTEPS, X.shape[-1])
        model.set_weights(init_w)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LR_SCHED),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        ckpt = tf.keras.callbacks.ModelCheckpoint(
            fd / "best.keras", monitor="val_loss", save_best_only=True, verbose=1
        )

        # train
        hist = model.fit(
            X_res, y_res,
            validation_data=(X_va, y_va),
            epochs=int(config.EPOCHS / 3),
            batch_size=config.BATCH_SIZE,
            callbacks=[ckpt],
            verbose=2
        )
        fold_histories.append(hist.history)

        # evaluate
        preds = model.predict(X_va, batch_size=config.BATCH_SIZE)
        probs = preds if config.NUM_CLASSES > 1 else tf.nn.sigmoid(preds).numpy()
        y_pred = np.argmax(probs, axis=1)
        y_true = y_va_raw

        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred, average="macro")
        auroc = roc_auc_score(y_true, probs[:,1] if config.NUM_CLASSES>1 else probs[:,0])
        auprc = average_precision_score(y_true, probs[:,1] if config.NUM_CLASSES>1 else probs[:,0])

        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auroc.append(auroc)
        fold_auprc.append(auprc)

        # save metrics & plots
        rpt = classification_report(y_true, y_pred, digits=4)
        print(rpt, f"Fold{fold}: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}", sep="\n")
        (fd / "classification_report.txt").write_text(rpt)

        with open(fd / "history.json", "w") as f:
            json.dump(hist.history, f, indent=2)

        cm = confusion_matrix(y_true, y_pred)
        np.save(fd / "confusion.npy", cm)

        tf.keras.backend.clear_session()
        gc.collect()

    # 4) summarize CV
    summary = (
        "\n——— CV Summary ———\n"
        f"Accuracies: {fold_acc}\n"
        f"Mean±Std Acc: {np.mean(fold_acc):.4f}±{np.std(fold_acc):.4f}\n\n"
        f"F1 Scores: {fold_f1}\n"
        f"Mean±Std F1: {np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f}\n\n"
        f"AUROC: {fold_auroc}\n"
        f"Mean±Std AUROC: {np.mean(fold_auroc):.4f}±{np.std(fold_auroc):.4f}\n\n"
        f"AUPRC: {fold_auprc}\n"
        f"Mean±Std AUPRC: {np.mean(fold_auprc):.4f}±{np.std(fold_auprc):.4f}\n"
    )
    print(summary)
    (run_dir / "cv_summary.txt").write_text(summary)

    # 5) plot CV loss curves
    plt.figure(figsize=(7,5))
    for i, h in enumerate(fold_histories, start=1):
        plt.plot(h["val_loss"], label=f"fold{i}")
    plt.title("Validation Loss per Fold")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "val_loss_cv.png"); plt.close()

    print(f"[INFO] Artifacts saved in {run_dir.resolve()}")

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(config.SEED)
    main()
