"""
Baseline CNN + LSTM with RobustScaler & SMOTE balancing

Usage:
    python run_cnn_lstm.py --data_dir /path/to/xlsx_folder

Outputs under ./artifacts_cnn_lstm/:
  • model.keras
  • scaler.joblib
  • history.json
  • plt_cnn_lstm.png
  • cm.npy, cm.png
  • classification.txt
"""
import os, json, joblib, pickle
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ── Hyperparams ─────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 128
LR = 3e-4
PATIENCE_LR = 8
TEST_SPLIT = 0.2
SEED = 42

FINAL_COLUMNS = ["A", "G", "M", "S0", "S1", "S2"]
EXPECTED_RAW = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"]


# ── Data loader (RobustScaler) ─────────────────────────────
def load_supervised_data(data_dir: Path):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith((".xlsx", ".xls"))])
    data_list, labels_list, file_list, err_list, timestamps = [], [], [], [], []

    for fname in files:
        fp = data_dir / fname
        try:
            label = 1 if "Right" in fname else 0
            df = pd.read_excel(fp)
            if "_time" in df:
                df = df.rename(columns={"_time": "timestamp"})
                df["timestamp"] = df["timestamp"].astype("int64")
            else:
                err_list.append(f"{fname}: no timestamp")
                continue

            missing = [c for c in EXPECTED_RAW if c not in df.columns]
            if len(missing) > 3:
                err_list.append(f"{fname}: missing {missing}")
                continue

            # compute norms
            df["A"] = np.linalg.norm(df[["Ax", "Ay", "Az"]], axis=1)
            df["G"] = np.linalg.norm(df[["Gx", "Gy", "Gz"]], axis=1)
            df["M"] = np.linalg.norm(df[["Mx", "My", "Mz"]], axis=1)

            data_list.append(df)
            labels_list.append(label)
            file_list.append(fname)
        except Exception as e:
            err_list.append(f"{fname}: {e}")
            continue

    if not data_list:
        raise RuntimeError("No valid files.")

    # unify session lengths & collect timestamps
    min_len = min(df.shape[0] for df in data_list)
    for idx, df in enumerate(data_list):
        timestamps.append((df["timestamp"].iloc[0],
                           df["timestamp"].iloc[min_len - 1],
                           file_list[idx]))
        data_list[idx] = df.loc[:min_len - 1, FINAL_COLUMNS].to_numpy()

    # stack sessions
    X = np.stack(data_list, axis=0)  # (N_sessions, T, 6)
    y_raw = np.array(labels_list)

    # scale (RobustScaler)
    N, T, F = X.shape
    scaler = RobustScaler()
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, T, F)

    return X_scaled, y_raw, timestamps, scaler, err_list


# ── Model builder ───────────────────────────────────────────
def build_model(input_shape, num_classes=2):
    model = Sequential([
        Conv1D(32, 3, activation="relu", padding="same", input_shape=input_shape),
        MaxPooling1D(2), BatchNormalization(),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2), BatchNormalization(),
        Conv1D(64, 3, activation="relu", padding="same"),
        MaxPooling1D(2), BatchNormalization(),
        LSTM(64, return_sequences=True), Dropout(0.4),
        LSTM(32, return_sequences=False), Dropout(0.4),
        Dense(32, activation="relu"), Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ── Main ────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    args = p.parse_args()

    ART_DIR = Path("artifacts_cnn_lstm")
    ART_DIR.mkdir(exist_ok=True)

    # load + preprocess
    X, y_raw, timestamps, scaler, errs = load_supervised_data(args.data_dir)
    joblib.dump(scaler, ART_DIR / "scaler.joblib")

    # one-hot labels
    num_classes = 2
    y = to_categorical(y_raw, num_classes)

    # train/test split
    X_tr, X_val, y_tr, y_val, y_raw_tr, y_raw_val = train_test_split(
        X, y, y_raw, test_size=TEST_SPLIT, stratify=y, random_state=SEED)

    # SMOTE on training data
    n_tr, T, F = X_tr.shape
    X_tr_flat = X_tr.reshape(n_tr, T * F)
    sm = SMOTE(random_state=SEED)
    X_res_flat, y_res_raw = sm.fit_resample(X_tr_flat, y_raw_tr)
    X_res = X_res_flat.reshape(-1, T, F)
    y_res = to_categorical(y_res_raw, num_classes)

    # build & train
    model = build_model((T, F), num_classes)
    lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=PATIENCE_LR, min_lr=1e-5, verbose=1)

    history = model.fit(
        X_res, y_res,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_cb],
        verbose=2
    )

    # save artifacts
    model.save(ART_DIR / "model.keras")
    with open(ART_DIR / "history.json", "w") as f: json.dump(history.history, f)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], marker="o", label="train loss")
    plt.plot(history.history["val_loss"], marker="o", label="val loss")
    plt.legend();
    plt.title("Loss");
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], marker="o", label="train acc")
    plt.plot(history.history["val_accuracy"], marker="o", label="val acc")
    plt.legend();
    plt.title("Accuracy");
    plt.grid()
    plt.tight_layout()
    plt.savefig(ART_DIR / "plt_cnn_lstm.png")

    # Eval
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    np.save(ART_DIR / "cm.npy", cm)
    report = classification_report(y_true, y_pred)
    with open(ART_DIR / "classification.txt", "w") as f: f.write(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"CNN+LSTM val acc={acc:.3f}")
    plt.savefig(ART_DIR / "cm.png")

    print(f"Validation Accuracy: {acc:.4f}")
    print(report)
    print(f"Artifacts in {ART_DIR.resolve()}")


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    main()
