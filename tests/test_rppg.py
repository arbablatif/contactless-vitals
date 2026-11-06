"""Algorithm behaviour on exact colour traces, separate from any video."""

import numpy as np
import pytest

from contactless_vitals.config import SignalConfig
from contactless_vitals.data.synthetic import PULSE_SIGNATURE
from contactless_vitals.signal.rppg import ALGORITHMS, chrom, extract_pulse, green, pos

FS = 30.0
CFG = SignalConfig()
SKIN = np.array([170.0, 115.0, 85.0])
METHODS = sorted(ALGORITHMS)


def trace(pulse_hz=1.2, amplitude=0.008, brightness=None, seconds=20.0):
    """Exact mean-RGB trace: skin colour x pulse x optional brightness change."""
    t = np.arange(int(seconds * FS)) / FS
    pulse = np.sin(2 * np.pi * pulse_hz * t)
    rgb = SKIN * (1.0 - amplitude * PULSE_SIGNATURE * pulse[:, None])
    if brightness is not None:
        rgb = rgb * (1.0 + brightness(t))[:, None]
    return t, pulse, rgb


@pytest.mark.parametrize("method", METHODS)
def test_up_means_more_blood(method):
    _, pulse, rgb = trace()
    output = extract_pulse(rgb, FS, method, CFG)
    assert np.corrcoef(output[60:-60], pulse[60:-60])[0, 1] > 0.99


@pytest.mark.parametrize("method", METHODS)
def test_output_has_unit_variance(method):
    _, _, rgb = trace()
    assert extract_pulse(rgb, FS, method, CFG).std() == pytest.approx(1.0)


@pytest.mark.parametrize("gain", [0.5, 2.0])
@pytest.mark.parametrize("method", METHODS)
def test_camera_exposure_does_not_change_the_result(method, gain):
    _, _, rgb = trace()
    assert np.allclose(
        extract_pulse(rgb, FS, method, CFG), extract_pulse(rgb * gain, FS, method, CFG), atol=1e-6
    )


def test_pos_projection_is_orthogonal_to_skin_tone():
    """A pure brightness change must vanish exactly: this is POS's central claim."""
    t = np.arange(600) / FS
    rgb = SKIN * (1.0 + 0.05 * np.sin(2 * np.pi * 1.9 * t))[:, None]
    assert np.abs(pos(rgb, FS, CFG)).max() < 1e-9


def test_chrom_cancels_a_pure_brightness_change():
    t = np.arange(600) / FS
    rgb = SKIN * (1.0 + 0.05 * np.sin(2 * np.pi * 1.9 * t))[:, None]
    assert np.abs(chrom(rgb, FS, CFG)).max() < 1e-6


def test_green_has_no_defence_against_brightness():
    t = np.arange(600) / FS
    rgb = SKIN * (1.0 + 0.05 * np.sin(2 * np.pi * 1.9 * t))[:, None]
    assert green(rgb, FS, CFG).std() > 0.03


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        extract_pulse(np.ones((100, 3)), FS, "ica", CFG)
