"""Filters, detrending, resampling and windowing, tested on exact signals."""

import numpy as np
import pytest
from conftest import sine
from scipy import signal

from contactless_vitals.signal.preprocess import (
    bandpass,
    design_bandpass,
    detrend,
    estimate_fs,
    finite_runs,
    to_uniform_grid,
)
from contactless_vitals.signal.windows import overlap_add, sliding_windows

FS = 30.0
BAND = (0.7, 4.0)


def zero_phase_gain_db(freq_hz: float, order: int = 3) -> float:
    """Gain of the filter applied twice, as filtfilt does."""
    _, h = signal.freqz_sos(design_bandpass(FS, BAND, order), worN=[freq_hz], fs=FS)
    return float(20 * np.log10(np.abs(h[0]) ** 2))


@pytest.mark.parametrize("edge_hz", BAND)
def test_band_edges_are_minus_3_db_after_zero_phase_filtering(edge_hz):
    assert zero_phase_gain_db(edge_hz) == pytest.approx(-3.0, abs=0.3)


def test_passband_is_flat_and_stopband_is_rejected():
    assert abs(zero_phase_gain_db(1.5)) < 0.1
    assert zero_phase_gain_db(0.2) < -40
    assert zero_phase_gain_db(10.0) < -40


def test_bandpass_does_not_shift_the_signal_in_time():
    t, x = sine(1.5)
    y = bandpass(x, FS, BAND, 3)
    middle = slice(150, -150)
    assert np.corrcoef(x[middle], y[middle])[0, 1] > 0.9999


def test_bandpass_rejects_an_invalid_band():
    with pytest.raises(ValueError):
        design_bandpass(6.0, BAND, 3)


def test_detrend_removes_curved_drift_but_keeps_the_pulse():
    t, pulse = sine(0.7)
    drift = 5.0 * (t / t[-1]) ** 2 + 2.0 * np.sin(2 * np.pi * 0.03 * t)
    cleaned = detrend(pulse + drift, FS, 0.25)
    middle = slice(150, -150)
    assert np.std(cleaned[middle] - pulse[middle]) < 0.05


def test_detrend_cutoff_does_not_depend_on_frame_rate():
    gains = []
    for fs in (15.0, 30.0, 60.0):
        _, x = sine(0.7, fs=fs)
        middle = slice(int(5 * fs), int(-5 * fs))
        gains.append(detrend(x, fs, 0.25)[middle].std() / x[middle].std())
    assert max(gains) - min(gains) < 0.01


def test_estimate_fs_ignores_a_dropped_frame():
    t = np.delete(np.arange(300) / 30.0, 100)
    assert estimate_fs(t) == pytest.approx(30.0)


def test_uniform_grid_corrects_uneven_timestamps():
    rng = np.random.default_rng(0)
    t = np.sort(np.arange(600) / 30.0 + rng.uniform(-0.008, 0.008, 600))
    grid_t, x = to_uniform_grid(t, 2.0 * t + 1.0, 30.0, 0.5)
    assert np.allclose(np.diff(grid_t), 1 / 30.0)
    assert np.allclose(x, 2.0 * grid_t + 1.0)


def test_short_gaps_are_filled_and_long_gaps_stay_empty():
    t = np.arange(300) / 30.0
    x = t.copy()
    x[50:55] = np.nan  # 0.17 s: fill
    x[150:200] = np.nan  # 1.67 s: leave
    grid_t, y = to_uniform_grid(t, x, 30.0, 0.5)
    assert np.all(np.isfinite(y[50:55]))
    assert np.all(np.isnan(y[150:200]))
    assert np.isfinite(y[149]) and np.isfinite(y[200])


def test_finite_runs():
    mask = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    assert finite_runs(mask) == [slice(1, 3), slice(5, 6)]


def test_overlap_add_rebuilds_a_constant_at_every_sample():
    starts = np.arange(0, 91, 5)
    y = overlap_add([np.full(10, 3.0)] * starts.size, starts, 100)
    assert np.allclose(y, 3.0)


def test_sliding_windows_shape_scale_and_nan_handling():
    t, x = sine(1.0, duration_s=30.0)
    x[450] = np.nan  # the 15 s mark
    starts, windows = sliding_windows(x, FS, 10.0, 1.0)
    assert windows.shape[1] == 300
    assert np.all(np.isfinite(windows))
    assert np.allclose(windows.std(axis=1), 1.0)
    assert not np.any((starts <= 15.0) & (starts + 10.0 > 15.0))
