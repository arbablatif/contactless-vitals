"""Real MediaPipe on a public-domain portrait with a known pulse added.

MediaPipe needs a real face, so the procedural faces cannot test it. This uses
the NASA astronaut portrait bundled with scikit-image, which is public domain.
The model downloads on first use; these tests skip if that is not possible.
"""

from urllib.error import URLError

import numpy as np
import pytest

from contactless_vitals.config import SkinConfig
from contactless_vitals.data.synthetic import SyntheticSpec, animate_portrait
from contactless_vitals.extract import extract_rgb_trace
from contactless_vitals.face.landmarks import MediaPipeLandmarks, ensure_model
from contactless_vitals.face.regions import N_LANDMARKS, REGION_NAMES
from contactless_vitals.face.skin import region_masks
from contactless_vitals.pipeline import COMBINED, run_rppg
from contactless_vitals.quality import snr_at_known_frequency

pytestmark = pytest.mark.mediapipe
SPEC = SyntheticSpec(pulse_hz=1.3, duration_s=12.0, sway_px=3.0, sway_hz=0.4)


@pytest.fixture(scope="module")
def portrait():
    skimage_data = pytest.importorskip("skimage.data")
    return skimage_data.astronaut()


@pytest.fixture(scope="module")
def provider():
    try:
        ensure_model()
    except URLError as error:
        pytest.skip(f"face model unavailable: {error}")
    with MediaPipeLandmarks() as landmarks:
        yield landmarks


@pytest.fixture(scope="module")
def trace(portrait, provider):
    return extract_rgb_trace(animate_portrait(portrait, SPEC), provider)


def test_landmarks_have_the_expected_shape(portrait):
    with MediaPipeLandmarks() as landmarks:
        points = landmarks.detect(portrait, 0.0)
    assert points.shape == (N_LANDMARKS, 2)
    h, w = portrait.shape[:2]
    assert 0 < points[:, 0].min() and points[:, 0].max() < w
    assert 0 < points[:, 1].min() and points[:, 1].max() < h


def test_no_face_returns_none():
    with MediaPipeLandmarks() as landmarks:
        assert landmarks.detect(np.full((240, 320, 3), 128, np.uint8), 0.0) is None


def test_regions_are_on_skin_and_clear_of_the_eyes(portrait):
    with MediaPipeLandmarks() as landmarks:
        points = landmarks.detect(portrait, 0.0)
    masks = region_masks(portrait, points, SkinConfig())
    for name in REGION_NAMES:
        assert masks[name].sum() > 200, name
    every_region = np.any(np.stack(list(masks.values())), axis=0)
    for iris_centre in (468, 473):
        x, y = np.rint(points[iris_centre]).astype(int)
        assert not every_region[y, x]
    # The subject's left cheek appears on the right of the image.
    left_x = np.nonzero(masks["left_cheek"])[1].mean()
    right_x = np.nonzero(masks["right_cheek"])[1].mean()
    assert left_x > right_x


def test_face_is_tracked_in_every_frame(trace):
    assert np.all(trace.pixels > 0)


def test_tracking_follows_the_moving_head(trace):
    """Stable pixel counts mean the regions moved with the face."""
    counts = trace.pixels.astype(float)
    assert np.all(counts.std(axis=0) / counts.mean(axis=0) < 0.1)


@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_known_pulse_is_recovered_through_real_tracking(trace, method):
    result = run_rppg(trace)
    x = result.waveforms[COMBINED][method]
    assert snr_at_known_frequency(x, result.fs, SPEC.pulse_hz) > 6.0
