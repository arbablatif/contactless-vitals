"""Sliding-window helpers."""

import numpy as np


def window_starts(n: int, length: int, step: int) -> np.ndarray:
    """Start indices of every full window of `length` samples in n samples."""
    if length > n:
        return np.empty(0, dtype=int)
    return np.arange(0, n - length + 1, step)


def overlap_add(
    pieces: list[np.ndarray],
    starts: np.ndarray,
    n: int,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Join overlapping pieces into one signal, normalising by total weight.

    Dividing by the accumulated weight keeps the amplitude correct everywhere,
    including the signal's ends, where fewer windows overlap.
    """
    total = np.zeros(n)
    norm = np.zeros(n)
    w = np.ones(len(pieces[0])) if weight is None else weight
    for piece, start in zip(pieces, starts):
        total[start : start + w.size] += w * piece
        norm[start : start + w.size] += w
    return np.divide(total, norm, out=np.zeros(n), where=norm > 1e-9)


def sliding_windows(
    x: np.ndarray, fs: float, window_s: float, step_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a waveform into overlapping windows for the later heart-rate stage.

    Windows touching a NaN sample are left out, so every returned window is
    usable. Each is scaled to zero mean and unit variance.

    Returns:
        (start_times_s, windows) with shapes (W,) and (W, window_length).
    """
    x = np.asarray(x, dtype=float)
    length = int(round(window_s * fs))
    step = max(1, int(round(step_s * fs)))
    starts = [
        s
        for s in window_starts(x.size, length, step)
        if np.all(np.isfinite(x[s : s + length]))
    ]
    if not starts:
        return np.empty(0), np.empty((0, length))
    stack = np.stack([x[s : s + length] for s in starts])
    stack = stack - stack.mean(axis=1, keepdims=True)
    stack = stack / np.maximum(stack.std(axis=1, keepdims=True), 1e-12)
    return np.asarray(starts) / fs, stack
