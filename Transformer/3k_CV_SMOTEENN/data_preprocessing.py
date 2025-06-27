"""
Load raw .xlsx files, compute norms, window the signals,
scale features, and return arrays for classification.
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
    Read one Excel chunk, compute A/G/M norms, and infer its label.

    Args:
        path (Path): Path to the .xlsx file.

    Returns:
        Tuple[np.ndarray, int]:
            - data: Array of shape (timesteps, 6) with features [A, G, M, S0, S1, S2].
            - label: Integer class label (0 = Left, 1 = Right).
    """
    df = pd.read_excel(path)
    df["A"] = np.linalg.norm(df[["Ax", "Ay", "Az"]], axis=1)
    df["G"] = np.linalg.norm(df[["Gx", "Gy", "Gz"]], axis=1)
    df["M"] = np.linalg.norm(df[["Mx", "My", "Mz"]], axis=1)
    data = df[config.FINAL_COLUMNS].to_numpy()
    label = 1 if "Right" in path.stem else 0
    return data, label


def _window_signal(
    sig: np.ndarray,
    timesteps: int,
    stride: int
) -> np.ndarray:
    """
    Split a multivariate time series into overlapping windows.

    Args:
        sig (np.ndarray): Input array of shape (n_samples, n_features).
        timesteps (int): Window length.
        stride (int): Hop length.

    Returns:
        np.ndarray: Array of shape (n_windows, timesteps, n_features).
    """
    windows = []
    for start in range(0, sig.shape[0] - timesteps + 1, stride):
        windows.append(sig[start : start + timesteps])
    return np.stack(windows) if windows else np.empty((0, timesteps, sig.shape[1]))


def load_dataset(
    scaler_type: str = "standard"
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Discover Excel files, preprocess signals, window them, and fit a scaler.

    Args:
        scaler_type (str): One of "standard" or "robust".

    Returns:
        Tuple[np.ndarray, np.ndarray, object]:
            - X: Array of shape (N, timesteps, features).
            - y: One-hot labels of shape (N, NUM_CLASSES).
            - scaler: Fitted StandardScaler or RobustScaler.

    Raises:
        RuntimeError:
            - If no valid .xlsx files are found.
            - If no windows are generated after windowing.
    """
    files = sorted(Path(config.RAW_DATA_DIR).glob("*.xls*"))
    files = [f for f in files if not f.name.startswith("~$")]
    if not files:
        raise RuntimeError(f"No valid .xlsx files in {config.RAW_DATA_DIR}")

    raw_sessions, labels = [], []
    for f in files:
        data, lbl = _read_one_file(f)
        if data.shape[0] < config.TIMESTEPS:
            print(f"Skipping {f.name}: only {data.shape[0]} rows")
            continue
        raw_sessions.append(data)
        labels.append(lbl)

    if not raw_sessions:
        raise RuntimeError("No valid sessions loaded.")

    # Trim all sessions to shortest length
    min_len = min(s.shape[0] for s in raw_sessions)
    raw_sessions = [s[:min_len] for s in raw_sessions]

    # Windowing + label expansion
    X_list, y_list = [], []
    for sess, lbl in zip(raw_sessions, labels):
        w = _window_signal(sess, config.TIMESTEPS, config.STRIDE)
        X_list.append(w)
        y_list += [lbl] * len(w)

    X = np.concatenate(X_list, axis=0)
    if X.shape[0] == 0:
        raise RuntimeError("No windows generated.")
    y = to_categorical(y_list, num_classes=config.NUM_CLASSES)

    # Flatten → scale → reshape
    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    flat = scaler.fit_transform(flat)
    X = flat.reshape(X.shape)

    # Persist the scaler
    tag = "std" if scaler_type == "standard" else "rob"
    joblib.dump(scaler, config.ARTIFACT_DIR / f"scaler_{tag}.joblib")

    return X, y, scaler


if __name__ == "__main__":
    X, y, scaler = load_dataset()
    print(f"Preprocessed. X.shape={X.shape}, y.shape={y.shape}")
