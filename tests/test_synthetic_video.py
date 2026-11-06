"""The whole pipeline on synthetic videos with a known pulse frequency.

These check the signal against the known answer. The code under test never
searches for a frequency.
"""

import numpy as np
import pytest
from conftest import process_video

from contactless_vitals.data.synthetic import (
    SCENARIOS,
    SKIN_TONES,
    SyntheticSpec,
    pulse_waveform,
    synthetic_frames,
)
from contactless_vitals.pipeline import COMBINED
from contactless_vitals.quality import correlation_with_lag, snr_at_known_frequency

METHODS = ("green", "chrom", "pos")
GOOD_SNR_DB = 10.0  # pulse power at least ten times everything else in the band


def snr(spec: SyntheticSpec, method: str, source: str = COMBINED) -> float:
    _, result = process_video(spec)
    return snr_at_known_frequency(result.waveforms[source][method], result.fs, spec.pulse_hz)


def test_frames_are_8_bit_rgb_with_the_requested_count():
    spec = SyntheticSpec(duration_s=1.0)
    frames = list(synthetic_frames(spec))
    assert len(frames) == 30
    assert frames[0][1].shape == (spec.height, spec.width, 3)
    assert frames[0][1].dtype == np.uint8


def test_same_seed_gives_identical_video():
    spec = SyntheticSpec(duration_s=0.2)
    first = [f for _, f in synthetic_frames(spec)]
    second = [f for _, f in synthetic_frames(spec)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("pulse_hz", [0.8, 1.2, 2.0, 3.0, 3.5])
@pytest.mark.parametrize("method", METHODS)
def test_known_pulse_is_recovered_across_the_band(method, pulse_hz):
    assert snr(SyntheticSpec(pulse_hz=pulse_hz), method) > GOOD_SNR_DB


@pytest.mark.parametrize("method", METHODS)
def test_waveform_shape_matches_the_true_pulse(method):
    spec = SyntheticSpec()
    _, result = process_video(spec)
    truth = pulse_waveform(result.t, spec.pulse_hz)
    r, lag = correlation_with_lag(result.waveforms[COMBINED][method], truth, result.fs, 0.1)
    assert r > 0.95
    assert abs(lag) <= 2 / result.fs


@pytest.mark.parametrize("method", METHODS)
def test_no_pulse_means_no_pulse_is_reported(method):
    """Negative control: with no pulse the pipeline must not invent one."""
    assert snr(SyntheticSpec(pulse_amplitude=0.0), method) < 0.0


@pytest.mark.parametrize("region", ["forehead", "left_cheek", "right_cheek"])
@pytest.mark.parametrize("method", METHODS)
def test_each_region_carries_the_pulse_alone(method, region):
    assert snr(SyntheticSpec(), method, region) > GOOD_SNR_DB


@pytest.mark.parametrize("method", METHODS)
def test_slow_lighting_drift_is_removed(method):
    assert snr(SCENARIOS["slow_lighting_drift"], method) > GOOD_SNR_DB


@pytest.mark.parametrize("scenario", ["in_band_brightness", "in_band_specular"])
def test_chrom_and_pos_beat_green_when_distortion_is_inside_the_band(scenario):
    """The reason CHROM and POS exist: band-pass filtering cannot remove this."""
    spec = SCENARIOS[scenario]
    assert snr(spec, "green") < 0.0
    assert snr(spec, "chrom") > GOOD_SNR_DB
    assert snr(spec, "pos") > GOOD_SNR_DB


@pytest.mark.parametrize("method", METHODS)
def test_head_sway_with_tracking_noise(method):
    assert snr(SCENARIOS["head_sway"], method) > GOOD_SNR_DB


@pytest.mark.parametrize("tone", SKIN_TONES)
@pytest.mark.parametrize("method", METHODS)
def test_pulse_is_recovered_on_each_skin_tone(method, tone):
    spec = SyntheticSpec(skin_rgb=SKIN_TONES[tone])
    trace, _ = process_video(spec)
    assert trace.pixels.min() > 100, "the skin mask must not reject this skin tone"
    assert snr(spec, method) > GOOD_SNR_DB


@pytest.mark.parametrize("method", METHODS)
def test_weak_pulse_on_dark_skin_is_detectable_but_measurably_worse(method):
    """A known weakness, pinned on purpose rather than hidden.

    Melanin absorbs light before it reaches the blood, so the pulse is weaker;
    half strength is a rough stand-in. Sensor noise does not shrink with it, so
    SNR falls. CHROM and POS suffer most because they use the dim blue channel.
    The pulse must still beat everything else in the band, but this case does
    not reach GOOD_SNR_DB for CHROM or POS. See "Limitations" in the README.
    """
    weak = SyntheticSpec(skin_rgb=SKIN_TONES["dark"], pulse_amplitude=0.004)
    assert 3.0 < snr(weak, method) < snr(SyntheticSpec(skin_rgb=SKIN_TONES["light"]), method)


@pytest.mark.parametrize("method", METHODS)
def test_uneven_frame_timing_is_handled(method):
    assert snr(SyntheticSpec(timing_jitter_s=0.008), method) > GOOD_SNR_DB


def test_short_face_loss_is_bridged():
    spec = SyntheticSpec(dropped_frames=((200, 206),))
    _, result = process_video(spec)
    assert np.all(np.isfinite(result.waveforms[COMBINED]["pos"]))
    assert snr(spec, "pos") > GOOD_SNR_DB


def test_long_face_loss_leaves_a_gap_and_nothing_is_invented():
    spec = SyntheticSpec(dropped_frames=((270, 330),))
    _, result = process_video(spec)
    x = result.waveforms[COMBINED]["pos"]
    gap = (result.t > 9.1) & (result.t < 10.9)
    assert np.all(np.isnan(x[gap]))
    assert np.isfinite(x[result.t < 8.9]).all()
    assert np.isfinite(x[result.t > 11.1]).all()


def test_too_short_a_stretch_is_discarded():
    spec = SyntheticSpec(duration_s=4.0)
    _, result = process_video(spec)
    assert np.all(np.isnan(result.waveforms[COMBINED]["pos"]))


def test_sliding_windows_from_a_result():
    from contactless_vitals.config import SignalConfig

    cfg = SignalConfig()
    _, result = process_video(SyntheticSpec())
    starts, windows = result.windows(COMBINED, "pos", cfg)
    assert windows.shape == (11, 300)
    assert starts[1] - starts[0] == pytest.approx(1.0)
