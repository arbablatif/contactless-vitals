"""The three classic rPPG algorithms: GREEN, CHROM and POS.

Each takes an RGB trace of shape (T, 3), evenly sampled with no NaNs, and returns
a raw pulse signal of shape (T,). Detrending and the final band-pass are shared
and happen in `extract_pulse`.

Polarity: all outputs follow the same convention, up = more blood. More blood
absorbs more light, so raw camera brightness moves the opposite way. GREEN and
POS are negated to match; CHROM already comes out this way round.
"""

from collections.abc import Callable

import numpy as np
from scipy.signal import get_window

from contactless_vitals.config import SignalConfig
from contactless_vitals.signal.preprocess import bandpass, detrend
from contactless_vitals.signal.windows import overlap_add, window_starts

_EPS = 1e-12


def green(rgb: np.ndarray, fs: float, cfg: SignalConfig) -> np.ndarray:
    """GREEN (Verkruysse 2008): the green channel relative to its mean.

    Haemoglobin absorbs green light strongly, so this channel has the largest
    pulse. It has no defence against motion or lighting changes.
    """
    g = rgb[:, 1]
    return -(g / g.mean() - 1.0)


def _moving_average(x: np.ndarray, length: int) -> np.ndarray:
    """Centred moving average along axis 0, with edge padding."""
    kernel = np.ones(length) / length
    left, right = length // 2, length - 1 - length // 2
    padded = np.pad(x, ((left, right), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(x.shape[1])],
        axis=1,
    )


def chrom(rgb: np.ndarray, fs: float, cfg: SignalConfig) -> np.ndarray:
    """CHROM (de Haan and Jeanne 2013): two chrominance signals, tuned per window.

    X and Y are colour differences chosen so that, for standardised skin, a
    change in light intensity appears equally in both while the pulse does not.
    Subtracting them with the ratio alpha = std(X) / std(Y) cancels the shared
    distortion and keeps the pulse.

    Departure from a strictly per-window version: a 1.6 s window holds about one
    cycle at 0.7 Hz, too short to band-pass reliably. So normalisation uses a
    moving average and X, Y are band-passed over the whole trace. Alpha and the
    Hann overlap-add are still per window, as in the paper.
    """
    length = _algo_window(rgb.shape[0], fs, cfg)
    norm = rgb / np.maximum(_moving_average(rgb, length), _EPS)
    r, g, b = norm[:, 0], norm[:, 1], norm[:, 2]
    x = bandpass(3.0 * r - 2.0 * g, fs, cfg.band_hz, cfg.filter_order)
    y = bandpass(1.5 * r + g - 1.5 * b, fs, cfg.band_hz, cfg.filter_order)

    length += length % 2  # an even periodic Hann sums to a constant at 50% overlap
    length = min(length, x.size)
    starts = window_starts(x.size, length, max(1, length // 2))
    pieces = []
    for s in starts:
        xw, yw = x[s : s + length], y[s : s + length]
        alpha = xw.std() / max(yw.std(), _EPS)
        piece = xw - alpha * yw
        pieces.append(piece - piece.mean())
    return overlap_add(pieces, starts, x.size, get_window("hann", length))


def pos(rgb: np.ndarray, fs: float, cfg: SignalConfig) -> np.ndarray:
    """POS (Wang et al. 2017): project onto the plane orthogonal to skin tone.

    After dividing each channel by its window mean, skin colour becomes the
    vector [1, 1, 1], and brightness changes move the signal along it. The
    projection P has rows orthogonal to [1, 1, 1], so those changes vanish
    exactly. Alpha then balances the two projected signals to suppress
    specular reflections. Follows Algorithm 1 of the paper, stepping one sample.
    """
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    length = _algo_window(rgb.shape[0], fs, cfg)
    starts = window_starts(rgb.shape[0], length, 1)
    pieces = []
    for s in starts:
        window = rgb[s : s + length]
        s1, s2 = projection @ (window / np.maximum(window.mean(axis=0), _EPS)).T
        piece = s1 + (s1.std() / max(s2.std(), _EPS)) * s2
        pieces.append(piece - piece.mean())
    return -overlap_add(pieces, starts, rgb.shape[0])


def _algo_window(n: int, fs: float, cfg: SignalConfig) -> int:
    """Length of the CHROM and POS internal window, in samples."""
    return min(n, max(2, int(round(cfg.algo_window_s * fs))))


ALGORITHMS: dict[str, Callable[[np.ndarray, float, SignalConfig], np.ndarray]] = {
    "green": green,
    "chrom": chrom,
    "pos": pos,
}


def extract_pulse(
    rgb: np.ndarray, fs: float, method: str, cfg: SignalConfig | None = None
) -> np.ndarray:
    """Run one algorithm, then detrend, band-pass and scale to unit variance."""
    cfg = cfg or SignalConfig()
    if method not in ALGORITHMS:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(ALGORITHMS)}.")
    raw = ALGORITHMS[method](np.asarray(rgb, dtype=float), fs, cfg)
    pulse = bandpass(
        detrend(raw, fs, cfg.detrend_cutoff_hz), fs, cfg.band_hz, cfg.filter_order
    )
    return (pulse - pulse.mean()) / max(pulse.std(), _EPS)
