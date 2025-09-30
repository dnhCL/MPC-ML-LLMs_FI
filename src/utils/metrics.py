from __future__ import annotations
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        Valores verdaderos (shape: (n,) o (n, d)).
    y_pred : np.ndarray
        Predicciones (misma forma que y_true).

    Returns
    -------
    float
        Error absoluto medio.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty arrays not allowed")
    return float(np.mean(np.abs(y_true - y_pred)))
