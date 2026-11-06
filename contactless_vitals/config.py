"""All tunable settings in one place, as frozen dataclasses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkinConfig:
    """Settings for choosing skin pixels inside each face region.

    Thresholds are relative to each region's own median colour, not fixed
    YCrCb limits, so they do not assume any particular skin tone.
    """

    chroma_k: float = 3.0  # allowed chroma distance, in robust standard deviations
    chroma_floor: float = 6.0  # minimum chroma tolerance, in 8-bit levels
    dark_ratio: float = 0.5  # reject pixels darker than this x median luma (hair, shadow)
    bright_ratio: float = 1.35  # reject pixels brighter than this x median luma (highlights)
    saturation_level: int = 250  # reject clipped pixels: they cannot carry a pulse
    min_pixels: int = 30  # below this a region's mean is too noisy to use


@dataclass(frozen=True)
class SignalConfig:
    """Settings for turning an RGB trace into an rPPG waveform."""

    band_hz: tuple[float, float] = (0.7, 4.0)  # 42 to 240 beats per minute
    filter_order: int = 3  # Butterworth prototype order
    detrend_cutoff_hz: float = 0.25  # well below the band, so the pulse is untouched
    algo_window_s: float = 1.6  # CHROM and POS internal window
    max_gap_s: float = 0.5  # interpolate gaps up to this long; longer ones stay invalid
    min_segment_s: float = 5.0  # valid stretches shorter than this are discarded
    analysis_window_s: float = 10.0  # output windows for the later heart-rate stage
    analysis_step_s: float = 1.0
    fs_hz: float | None = None  # uniform grid rate; None uses the video's median rate
