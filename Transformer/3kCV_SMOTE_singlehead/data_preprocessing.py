"""
Loads raw Excel files, computes Euclidean norms for accelerometer, gyroscope and magnetometer data,
applies sliding-window segmentation, scales features, and returns arrays plus the scaler and feature names.
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.utils import to_categorical

import config

__all__ = ["load_dataset"]


def _read_one_file(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a single Excel file, compute feature norms, and assign its label.

    Returns:
        features (n_rows × n_feats), label (0 or 1)
    """
    label = 1 if "Right" in path.stem else 0
    expected = [
        "Ax", "Ay", "Az",
        "Gx", "Gy", "Gz",
        "Mx", "My", "Mz",
        "S0", "S1", "S2"
    ]
    df = pd.read_excel(path)
    missing = [c for c in expected if c not in df.columns]
    if len(missing) > 3:
        raise ValueError(f"{path.name}: too many missing cols {missing}")

    # Compute Euclidean norms
    df["A"] = np.linalg.norm(df[["Ax", "Ay", "Az"]], axis=1)
    df["G"] = np.linalg.norm(df[["Gx", "Gy", "Gz"]], axis=1)
    df["M"] = np.linalg.norm(df[["Mx", "My", "Mz"]], axis=1)

    # Extract only the final columns
    features = df[config.FINAL_COLUMNS].to_numpy()
    return features, label


def _window_signal(
    signal: np.ndarray,
    timesteps: int,
    stride: int
) -> np.ndarray:
    """
    Segment a multivariate time series into overlapping windows.

    Returns:
        n_windows × timesteps × n_features
    """
    wins = []
    for start in range(0, signal.shape[0] - timesteps + 1, stride):
        wins.append(signal[start : start + timesteps])
    return np.stack(wins) if wins else np.empty((0, timesteps, signal.shape[1]))


def load_dataset(
    scaler_type: str = "robust"
) -> tuple[np.ndarray, np.ndarray, object, list[str]]:
    """
    Load, window, scale sensor data, and return:

      X           → ndarray (n_samples, timesteps, n_features)
      y           → ndarray one-hot (n_samples, NUM_CLASSES)
      scaler      → fitted scaler object
      feat_names  → list of column names (config.FINAL_COLUMNS)

    Args:
        scaler_type: "standard" or "robust"

    Raises:
        RuntimeError if no files or no windows generated.
    """
    # 1) Discover files
    files = sorted(Path(config.RAW_DATA_DIR).glob("*.xls*"))
    files = [f for f in files if not f.name.startswith("~$")]
    if not files:
        raise RuntimeError(f"No Excel files in {config.RAW_DATA_DIR}")

    # 2) Read & collect sessions
    X_list, y_list = [], []
    for fp in files:
        try:
            arr, lbl = _read_one_file(fp)
            if arr.shape[0] < config.TIMESTEPS:
                print(f"Skipping {fp.name}: only {arr.shape[0]} rows (< {config.TIMESTEPS})")
                continue
            X_list.append(arr)
            y_list.append(lbl)
        except Exception as e:
            print(f"Skip {fp.name}: {e}")

    if not X_list:
        raise RuntimeError("No valid data processed. Check TIMESTEPS/STRIDE or data files.")

    # 3) Equalize session lengths
    min_len = min(arr.shape[0] for arr in X_list)
    X_list = [arr[:min_len] for arr in X_list]

    # 4) Windowing
    windows, labels = [], []
    for arr, lbl in zip(X_list, y_list):
        w = _window_signal(arr, config.TIMESTEPS, config.STRIDE)
        windows.append(w)
        labels += [lbl] * len(w)
    X = np.concatenate(windows, axis=0)
    if X.shape[0] == 0:
        raise RuntimeError("No windows generated. Adjust TIMESTEPS/STRIDE.")

    # 5) One-hot labels
    y = to_categorical(labels, num_classes=config.NUM_CLASSES)

    # 6) Scaling
    n, t, f = X.shape
    flat = X.reshape(-1, f)
    scaler_cls = StandardScaler if scaler_type == "standard" else RobustScaler
    scaler = scaler_cls()
    flat_scaled = scaler.fit_transform(flat)
    X = flat_scaled.reshape(n, t, f)

    # 7) Persist scaler
    tag = "std" if scaler_type == "standard" else "rob"
    out_path = config.ARTIFACT_DIR / f"scaler_{tag}.joblib"
    joblib.dump(scaler, out_path)
    print(f"Saved scaler to {out_path}")

    # 8) Return with feature‐names
    feat_names = config.FINAL_COLUMNS.copy()
    return X, y, scaler, feat_names


if __name__ == "__main__":
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting data preprocessing…")
    X, y, scaler, feat_names = load_dataset()
    print(f"Done. X.shape={X.shape}, y.shape={y.shape}, features={feat_names}")
    print("Scaler saved at:", config.ARTIFACT_DIR.resolve())
