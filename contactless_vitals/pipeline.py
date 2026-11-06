"""Step two of the pipeline: an RGB trace in, rPPG waveforms out."""

from dataclasses import dataclass

import numpy as np

from contactless_vitals.config import SignalConfig
from contactless_vitals.extract import RgbTrace
from contactless_vitals.signal.preprocess import estimate_fs, finite_runs, to_uniform_grid
from contactless_vitals.signal.rppg import ALGORITHMS, extract_pulse
from contactless_vitals.signal.windows import sliding_windows

COMBINED = "combined"


@dataclass
class RppgResult:
    """rPPG waveforms on one evenly spaced time grid.

    Attributes:
        t: grid times in seconds, shape (N,).
        fs: grid rate in Hz.
        waveforms: waveforms[source][method] has shape (N,) and is NaN where
            there was no usable signal. Sources are "combined" and each region.
        rgb: the resampled RGB trace per source, shape (N, 3).
    """

    t: np.ndarray
    fs: float
    waveforms: dict[str, dict[str, np.ndarray]]
    rgb: dict[str, np.ndarray]

    def windows(self, source: str, method: str, cfg: SignalConfig):
        """Sliding windows of one waveform: (start_times_s, windows)."""
        return sliding_windows(
            self.waveforms[source][method], self.fs, cfg.analysis_window_s, cfg.analysis_step_s
        )


def run_rppg(
    trace: RgbTrace,
    cfg: SignalConfig | None = None,
    methods: tuple[str, ...] = tuple(ALGORITHMS),
) -> RppgResult:
    """Compute every waveform for the combined trace and for each region.

    Each stretch of good signal is processed separately, so a long gap cannot
    smear filter artefacts into the data around it.
    """
    cfg = cfg or SignalConfig()
    fs = cfg.fs_hz or estimate_fs(trace.t)
    sources = {COMBINED: trace.combined()}
    sources.update({name: trace.region(name) for name in trace.regions})

    grid_t = None
    waveforms: dict[str, dict[str, np.ndarray]] = {}
    resampled: dict[str, np.ndarray] = {}
    min_samples = int(cfg.min_segment_s * fs)

    for source, rgb in sources.items():
        grid_t, grid_rgb = to_uniform_grid(trace.t, rgb, fs, cfg.max_gap_s)
        resampled[source] = grid_rgb
        waveforms[source] = {m: np.full(grid_t.size, np.nan) for m in methods}
        good = np.all(np.isfinite(grid_rgb), axis=1)
        for run in finite_runs(good):
            if run.stop - run.start < min_samples:
                continue
            for method in methods:
                waveforms[source][method][run] = extract_pulse(
                    grid_rgb[run], fs, method, cfg
                )
    return RppgResult(grid_t, fs, waveforms, resampled)
