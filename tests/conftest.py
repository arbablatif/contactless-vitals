"""Shared test helpers."""

from functools import lru_cache

import numpy as np

from contactless_vitals.data.synthetic import (
    ScriptedLandmarks,
    SyntheticSpec,
    synthetic_frames,
)
from contactless_vitals.extract import RgbTrace, extract_rgb_trace
from contactless_vitals.pipeline import RppgResult, run_rppg


@lru_cache(maxsize=None)
def process_video(spec: SyntheticSpec) -> tuple[RgbTrace, RppgResult]:
    """Render a synthetic video and run the full pipeline. Cached per spec."""
    trace = extract_rgb_trace(synthetic_frames(spec), ScriptedLandmarks(spec))
    return trace, run_rppg(trace)


def sine(freq_hz: float, fs: float = 30.0, duration_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(fs * duration_s)) / fs
    return t, np.sin(2.0 * np.pi * freq_hz * t)
