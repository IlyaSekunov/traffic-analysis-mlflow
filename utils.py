"""
Utility functions for data loading, preprocessing, and model persistence.

All functions are stateless and can be imported from any module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_PATH,
    RANDOM_STATE,
    RESOURCES_DIR,
    SCALER_PATH,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_xy(
    x_path: Path,
    y_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature matrix X and target vector y from .npy files.

    Parameters
    ----------
    x_path:
        Path to the feature matrix.
    y_path:
        Path to the target vector.  Defaults to ``<x_path.parent>/y_data.npy``.

    Returns
    -------
    (X, y) after removing rows that contain NaN or infinite values.

    Raises
    ------
    FileNotFoundError
        When either file cannot be found on disk.
    ValueError
        When the number of rows in X and y differ after loading.
    """
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")

    X = np.load(x_path)
    logger.debug("Loaded X from %s — shape: %s", x_path, X.shape)

    if y_path is None:
        y_path = x_path.parent / "y_data.npy"

    if not y_path.exists():
        raise FileNotFoundError(f"Target file not found: {y_path}")

    y = np.load(y_path)
    logger.debug("Loaded y from %s — shape: %s", y_path, y.shape)

    if len(X) != len(y):
        raise ValueError(
            f"Row count mismatch: X has {len(X)} rows, y has {len(y)} rows."
        )

    X, y = _remove_nan_rows(X, y)
    X, y = _clip_infinite(X, y)

    logger.info("Data ready — samples: %d  features: %d", X.shape[0], X.shape[1])
    return X, y


def _remove_nan_rows(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows where X or y contains NaN."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    removed = (~mask).sum()
    if removed:
        logger.warning("Removed %d rows containing NaN.", removed)
    return X[mask], y[mask]


def _clip_infinite(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace infinite values with the column max/min."""
    if np.isinf(X).any():
        logger.warning("Infinite values detected in X — replacing with column extremes.")
        X = np.nan_to_num(X, posinf=np.nanmax(X), neginf=np.nanmin(X))
    if np.isinf(y).any():
        logger.warning("Infinite values detected in y — replacing with extremes.")
        y = np.nan_to_num(y, posinf=np.nanmax(y), neginf=np.nanmin(y))
    return X, y


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_scaler(scaler: StandardScaler, path: Path = SCALER_PATH) -> None:
    """Serialize *scaler* to *path* using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler saved to %s", path)


def load_scaler(path: Path = SCALER_PATH) -> StandardScaler:
    """Deserialize a scaler from *path*.

    Raises
    ------
    FileNotFoundError
        When the scaler file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    scaler = joblib.load(path)
    logger.info("Scaler loaded from %s", path)
    return scaler


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def resources_summary() -> dict:
    """Return a dict describing the contents of the resources directory."""
    summary = {
        "resources_dir": str(RESOURCES_DIR),
        "model_exists": MODEL_PATH.exists(),
        "scaler_exists": SCALER_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }
    return summary