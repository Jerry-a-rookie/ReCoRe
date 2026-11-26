from __future__ import annotations

import hashlib

import numpy as np


def nested_window_indices(
    total_windows: int,
    ratio: float,
    seed: int,
) -> np.ndarray:
    """Return a deterministic nested subset of chronological window starts."""
    total_windows = int(total_windows)
    ratio = float(ratio)
    if total_windows <= 0:
        raise ValueError("total_windows must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("recore_train_window_ratio must be in (0, 1]")
    if ratio == 1.0:
        return np.arange(total_windows, dtype=np.int64)

    count = max(1, min(total_windows, int(np.ceil(total_windows * ratio))))
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(total_windows)
    return np.sort(permutation[:count].astype(np.int64, copy=False))


def window_indices_sha256(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()

