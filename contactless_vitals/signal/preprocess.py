"""Preprocessing: uniform resampling, gap handling, detrending and band-pass.

Every function here is pure: arrays in, arrays out.
"""

import math

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve


def estimate_fs(t: np.ndarray) -> float:
    """Sampling rate from timestamps, ignoring dropped-frame gaps.

    This uses the mean step, not the median. Real capture timestamps can bunch
    into short and long steps, and the median then lands on one of those groups:
    on a real recording it gave 26 Hz for a true rate of 29 Hz. Steps over 1.5
    times the median are gaps, not the normal rhythm, so they are left out.
    """
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Need at least two increasing timestamps to estimate the rate.")
    # Rounded because truncated lengths such as int(60 * fs) would otherwise turn on
    # float noise: 29.9999999999999 Hz gives 1799 and 30 Hz gives 1800.
    return round(float(1.0 / dt[dt <= 1.5 * np.median(dt)].mean()), 6)


def first_of_each_time(t: np.ndarray) -> np.ndarray:
    """Mask keeping one sample per timestamp, as interpolation needs rising times."""
    t = np.asarray(t, dtype=float)
    return np.concatenate(([True], np.diff(t) > 0))


def to_uniform_grid(
    t: np.ndarray, x: np.ndarray, fs: float, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample samples onto an evenly spaced time grid.

    Webcams do not deliver frames at perfectly even intervals, and filters
    assume they do. Non-finite rows are treated as missing. Gaps up to
    `max_gap_s` are linearly interpolated. Longer gaps, and time outside the
    first and last good sample, stay NaN so nothing is invented.

    Args:
        t: timestamps in seconds, shape (T,).
        x: samples, shape (T,) or (T, C).
        fs: output rate in Hz.
        max_gap_s: longest gap to interpolate across.

    Returns:
        (grid_t, grid_x) with grid_x shaped like x along its trailing axes.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x

    n_out = int(math.floor((t[-1] - t[0]) * fs)) + 1
    grid_t = t[0] + np.arange(n_out) / fs
    out = np.full((n_out, x2.shape[1]), np.nan)

    good = np.all(np.isfinite(x2), axis=1)
    if good.sum() >= 2:
        tg, xg = t[good], x2[good]
        unique = first_of_each_time(tg)
        tg, xg = tg[unique], xg[unique]
        for c in range(x2.shape[1]):
            out[:, c] = np.interp(grid_t, tg, xg[:, c], left=np.nan, right=np.nan)
        # Blank out grid points that fall inside a gap too long to trust.
        for i in np.flatnonzero(np.diff(tg) > max_gap_s):
            out[(grid_t > tg[i]) & (grid_t < tg[i + 1])] = np.nan

    return grid_t, out[:, 0] if squeeze else out


def finite_runs(mask: np.ndarray) -> list[slice]:
    """Return slices covering each consecutive run of True values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [slice(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def detrend(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Remove slow drift with the smoothness-priors method (Tarvainen 2002).

    It finds the smoothest curve that follows the signal and subtracts it.
    Unlike a straight-line detrend it follows curved lighting drift, and unlike
    a high-pass filter it has no start-up transient.

    Tarvainen's lambda sets the cutoff indirectly. The trend filter's gain is
    1 / (1 + lambda^2 w^4), so lambda = 1 / w_c^2 puts the half-gain point at
    `cutoff_hz`, whatever the frame rate.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return x - x.mean()
    lam = 1.0 / (2.0 * math.pi * cutoff_hz / fs) ** 2
    second_diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    system = sparse.eye(n) + lam**2 * (second_diff.T @ second_diff)
    return x - spsolve(system.tocsc(), x)


def design_bandpass(fs: float, band_hz: tuple[float, float], order: int) -> np.ndarray:
    """Design a Butterworth band-pass whose edges are -3 dB after filtfilt.

    Zero-phase filtering runs the filter twice, which squares its gain, so a
    plain design would be -6 dB at the requested edges. We widen the design
    band to compensate, keeping the same geometric centre.
    """
    low, high = band_hz
    if not 0 < low < high < fs / 2:
        raise ValueError(f"Band {band_hz} Hz must lie inside (0, {fs / 2}) Hz.")
    widen = (math.sqrt(2.0) - 1.0) ** (1.0 / (2 * order))
    bandwidth = (high - low) / widen
    centre_sq = low * high
    design_high = (bandwidth + math.sqrt(bandwidth**2 + 4 * centre_sq)) / 2
    design_low = centre_sq / design_high
    return signal.butter(
        order, [design_low, design_high], btype="bandpass", fs=fs, output="sos"
    )


def bandpass(
    x: np.ndarray, fs: float, band_hz: tuple[float, float], order: int
) -> np.ndarray:
    """Zero-phase band-pass. Zero phase keeps pulse timing, but needs the whole signal."""
    x = np.asarray(x, dtype=float)
    sos = design_bandpass(fs, band_hz, order)
    # Pad by about three periods of the lowest frequency to tame edge transients.
    padlen = min(x.size - 1, int(3 * fs / band_hz[0]))
    return signal.sosfiltfilt(sos, x, padlen=padlen)
