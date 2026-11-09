"""Data normalization utilities for EEG signals.

Implements fold-independent normalization according to project specifications
to strictly prevent data leakage across subjects or folds.

Normalization contract:
    stats = fit_normalizer(train_x)       # train_x is (N, C, T) or (C, T)
    out   = apply_normalizer(x, stats)

Statistics (mean and std) are computed exclusively on the training set and applied
consistently to validation and test sets.
"""

from typing import Any, Dict
import numpy as np


def fit_normalizer(
    train_x: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Compute per-channel normalization statistics from training EEG data only.

    Parameters
    ----------
    train_x : np.ndarray
        Training EEG array of shape (N, C, T) or (C, T).
        N: number of epochs
        C: number of channels
        T: number of time points
    eps : float, default=1e-8
        Numerical tolerance to protect against division by zero in flat channels.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean': per-channel mean with shape (1, C, 1) or (C, 1), float32
        - 'std': per-channel standard deviation with zero-variance protected, float32
        - 'eps': epsilon value used for numerical stability
        - 'n_channels': channel count C
    """
    if not isinstance(train_x, np.ndarray):
        train_x = np.asarray(train_x)

    if train_x.ndim == 3:
        # train_x is (N, C, T): compute statistics across epochs and time points per channel
        # Result shape: (1, C, 1) to enable seamless broadcasting over (N, C, T)
        mean = np.mean(train_x, axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.std(train_x, axis=(0, 2), keepdims=True).astype(np.float32)
        n_channels = train_x.shape[1]
    elif train_x.ndim == 2:
        # train_x is (C, T): compute statistics across time points per channel
        mean = np.mean(train_x, axis=1, keepdims=True).astype(np.float32)
        std = np.std(train_x, axis=1, keepdims=True).astype(np.float32)
        n_channels = train_x.shape[0]
    else:
        raise ValueError(
            f"Expected train_x to have 2 or 3 dimensions (C, T) or (N, C, T), "
            f"got shape {train_x.shape} with {train_x.ndim} dimensions."
        )

    # Protect against flat / zero-variance channels (e.g. disconnected, ground or reference channels)
    # If std is near zero, set denominator scale to 1.0 so (x - mean) remains 0.0 without NaNs
    safe_std = np.where(std < eps, 1.0, std).astype(np.float32)

    return {
        "mean": mean,
        "std": safe_std,
        "raw_std": std,
        "eps": float(eps),
        "n_channels": int(n_channels),
    }


def apply_normalizer(
    x: np.ndarray,
    stats: Dict[str, Any],
) -> np.ndarray:
    """Apply previously fitted normalization statistics to EEG data.

    Parameters
    ----------
    x : np.ndarray
        EEG array of shape (N, C, T) or (C, T).
    stats : dict
        Statistics returned by `fit_normalizer`.

    Returns
    -------
    np.ndarray
        Normalized EEG array with the same shape as `x`, dtype float32.
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    mean = stats["mean"]
    std = stats["std"]
    eps = stats.get("eps", 1e-8)

    orig_shape = x.shape

    if x.ndim == 3:
        # x is (N, C, T)
        if mean.ndim == 2:
            # Expand (C, 1) to (1, C, 1)
            mean = mean[np.newaxis, :, :]
            std = std[np.newaxis, :, :]

        if x.shape[1] != stats.get("n_channels", x.shape[1]):
            raise ValueError(
                f"Channel count mismatch: normalizer was fit with {stats.get('n_channels')} "
                f"channels but input has {x.shape[1]} channels."
            )

        # Broadcast (1, C, 1) across (N, C, T)
        denom = np.maximum(std, eps)
        out = (x.astype(np.float32, copy=False) - mean) / denom

    elif x.ndim == 2:
        # x is (C, T)
        if mean.ndim == 3:
            # Squeeze (1, C, 1) to (C, 1)
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        if x.shape[0] != stats.get("n_channels", x.shape[0]):
            raise ValueError(
                f"Channel count mismatch: normalizer was fit with {stats.get('n_channels')} "
                f"channels but input has {x.shape[0]} channels."
            )

        denom = np.maximum(std, eps)
        out = (x.astype(np.float32, copy=False) - mean) / denom

    else:
        raise ValueError(
            f"Expected x to have 2 or 3 dimensions (C, T) or (N, C, T), "
            f"got shape {x.shape} with {x.ndim} dimensions."
        )

    # Ensure shape preservation and float32 dtype
    assert out.shape == orig_shape, f"Shape mismatch: {out.shape} vs original {orig_shape}"
    return out.astype(np.float32, copy=False)

