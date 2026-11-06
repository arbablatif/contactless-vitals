"""Waveform quality checks against a frequency or signal that is already known.

Nothing here searches for a heart rate. Every function is given the answer
(a known pulse frequency or a reference waveform) and measures how well a
waveform agrees with it.
"""

import numpy as np
from scipy import signal


def snr_at_known_frequency(
    x: np.ndarray,
    fs: float,
    f0_hz: float,
    band_hz: tuple[float, float] = (0.7, 4.0),
    half_width_hz: float = 0.1,
) -> float:
    """Pulse SNR in dB, after de Haan and Jeanne (2013).

    Signal is the power near the known frequency and its second harmonic. Noise
    is all other power in the band. Zero dB means the two are equal.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # The Hann main lobe spreads a pure tone over +/- 2 / duration.
    half_width = max(half_width_hz, 2.0 * fs / x.size)
    freqs, power = signal.periodogram(
        x - x.mean(), fs, window="hann", nfft=max(x.size, round(60 * fs))
    )
    # Bins can sit exactly on a band edge. The tolerance includes them every time,
    # instead of letting float rounding decide.
    tolerance = 1e-9
    in_band = (freqs >= band_hz[0] - tolerance) & (freqs <= band_hz[1] + tolerance)
    near_pulse = (np.abs(freqs - f0_hz) <= half_width + tolerance) | (
        np.abs(freqs - 2 * f0_hz) <= 2 * half_width + tolerance
    )
    wanted = power[in_band & near_pulse].sum()
    rest = power[in_band & ~near_pulse].sum()
    return float(10.0 * np.log10(max(wanted, 1e-20) / max(rest, 1e-20)))


def windowed_snr(
    x: np.ndarray,
    t: np.ndarray,
    fs: float,
    reference_t: np.ndarray,
    reference_hz: np.ndarray,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> np.ndarray:
    """SNR per window, for real recordings where the known rate changes over time.

    The rate comes from the contact sensor, so this measures the waveform against
    the sensor's answer. Windows with gaps or no valid reference are skipped.
    """
    length, step = int(window_s * fs), int(step_s * fs)
    scores = []
    for start in range(0, x.size - length + 1, step):
        piece = x[start : start + length]
        span = (reference_t >= t[start]) & (reference_t < t[start + length - 1])
        rates = reference_hz[span]
        rates = rates[rates > 0]
        if np.all(np.isfinite(piece)) and rates.size:
            scores.append(snr_at_known_frequency(piece, fs, float(rates.mean())))
    return np.asarray(scores)


def shifted_correlations(
    x: np.ndarray,
    reference: np.ndarray,
    fs: float,
    min_shift_s: float = 3.0,
    step_s: float = 2.0,
) -> np.ndarray:
    """Chance baseline for correlation_with_lag.

    Band-passed signals correlate by accident, and the lag search adds to that,
    because it can line up almost any two rhythms. Sliding the reference far out
    of step, wrapping it round, keeps its rhythm but breaks the true link. The
    scores that remain are what the same measure gives by chance alone.
    """
    shifts = np.arange(min_shift_s, x.size / fs - min_shift_s, step_s)
    return np.array(
        [correlation_with_lag(x, np.roll(reference, int(s * fs)), fs)[0] for s in shifts]
    )


def correlation_with_lag(
    x: np.ndarray, reference: np.ndarray, fs: float, max_lag_s: float = 0.5
) -> tuple[float, float]:
    """Best Pearson correlation between two waveforms over small time shifts.

    The lag allows for delay between face and fingertip, where the pulse wave
    arrives at a different moment. Returns (correlation, lag_seconds). A positive
    lag means x runs later than the reference.
    """
    x = np.asarray(x, dtype=float)
    reference = np.asarray(reference, dtype=float)
    best_r, best_lag = -np.inf, 0
    for lag in range(-int(max_lag_s * fs), int(max_lag_s * fs) + 1):
        a = x[max(lag, 0) : x.size + min(lag, 0)]
        b = reference[max(-lag, 0) : reference.size + min(-lag, 0)]
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
            continue
        r = float(np.corrcoef(a[keep], b[keep])[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag / fs
