"""
Loads raw Excel files, computes Euclidean norms for accelerometer, gyroscope and magnetometer data,
applies sliding-window segmentation, scales features, and returns trainable arrays.
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

    Args:
        path (Path): Path to the input .xlsx file.

    Returns:
        Tuple[np.ndarray, int]:
            - 2D array of shape (n_rows, len(config.FINAL_COLUMNS)) containing features.
            - Integer label (0 for Left, 1 for Right).

    Raises:
        ValueError: If more than 3 expected columns are missing.
    """
    label = 1 if "Right" in path.stem else 0
    expected = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz", "S0", "S1", "S2"]
    df = pd.read_excel(path)
    missing = [c for c in expected if c not in df.columns]
    if len(missing) > 3:
        raise ValueError(f"{path.name}: too many missing cols {missing}")

    # Compute Euclidean norms
    df["A"] = np.linalg.norm(df[["Ax", "Ay", "Az"]], axis=1)
    df["G"] = np.linalg.norm(df[["Gx", "Gy", "Gz"]], axis=1)
    df["M"] = np.linalg.norm(df[["Mx", "My", "Mz"]], axis=1)

    features = df[config.FINAL_COLUMNS].to_numpy()
    return features, label


def _window_signal(
    signal: np.ndarray,
    timesteps: int,
    stride: int
) -> np.ndarray:
    """
    Segment a multivariate time series into overlapping windows.

    Args:
        signal (np.ndarray): Input array of shape (n_samples, n_features).
        timesteps (int): Number of timesteps per window.
        stride (int): Number of steps to slide the window.

    Returns:
        np.ndarray: Windows stacked into shape (n_windows, timesteps, n_features).
    """
    wins = []
    for start in range(0, signal.shape[0] - timesteps + 1, stride):
        wins.append(signal[start : start + timesteps])
    return np.stack(wins) if wins else np.empty((0, timesteps, signal.shape[1]))


def load_dataset(
    scaler_type: str = "standard"
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Discover raw files, read and window signals, scale features, and return arrays.

    Args:
        scaler_type (str, optional): Either "standard" or "robust" scaler.
            Defaults to "standard".

    Returns:
        Tuple[np.ndarray, np.ndarray, object]:
            - X: Array of shape (n_samples, timesteps, n_features).
            - y: One-hot encoded labels of shape (n_samples, config.NUM_CLASSES).
            - scaler: Fitted scaler object.

    Raises:
        RuntimeError:
            - If no Excel files found.
            - If no valid data processed.
            - If no windows generated.
    """
    # Discover Excel files and filter out temporary/locked files
    files = sorted(Path(config.RAW_DATA_DIR).glob("*.xls*"))
    files = [f for f in files if not f.name.startswith("~$")]
    if not files:
        raise RuntimeError(f"No valid .xlsx files found in {config.RAW_DATA_DIR}")

    X_list, y_list = [], []
    for fp in files:
        try:
            raw, lbl = _read_one_file(fp)
            if raw.shape[0] < config.TIMESTEPS:
                print(
                    f"Skipping {fp.name}: not enough rows "
                    f"(found {raw.shape[0]}, need ≥ {config.TIMESTEPS})"
                )
                continue
            X_list.append(raw)
            y_list.append(lbl)
        except Exception as e:
            print(f"Skip {fp.name}: {e}")

    if not X_list:
        raise RuntimeError("No valid Excel data processed. Check your data or windowing parameters.")

    # Equalize session lengths
    min_len = min(arr.shape[0] for arr in X_list)
    X_list = [arr[:min_len] for arr in X_list]

    # Windowing
    windows, labels = [], []
    for arr, lbl in zip(X_list, y_list):
        w = _window_signal(arr, config.TIMESTEPS, config.STRIDE)
        windows.append(w)
        labels += [lbl] * len(w)
    X = np.concatenate(windows, axis=0)
    if X.shape[0] == 0:
        raise RuntimeError("No windows generated. Check your TIMESTEPS and STRIDE parameters.")
    y = to_categorical(labels, num_classes=config.NUM_CLASSES)

    # Scaling
    flat = X.reshape(-1, X.shape[-1])
    scaler_cls = StandardScaler if scaler_type == "standard" else RobustScaler
    scaler = scaler_cls()
    flat = scaler.fit_transform(flat)
    X = flat.reshape(X.shape)

    # Persist scaler
    tag = "std" if scaler_type == "standard" else "rob"
    out_path = config.ARTIFACT_DIR / f"scaler_{tag}.joblib"
    joblib.dump(scaler, out_path)
    print(f"Saved scaler to {out_path}")

    return X, y, scaler


if __name__ == "__main__":
    # Ensure artifact dir exists and run preprocessing
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting data preprocessing…")
    X, y, scaler = load_dataset()
    print(f"Done. X.shape={X.shape}, y.shape={y.shape}")
    print("Scaler saved at:", config.ARTIFACT_DIR.resolve())
