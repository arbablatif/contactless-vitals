"""Skin mask and region averaging, on hand-built patches."""

import numpy as np
import pytest
from conftest import process_video

from contactless_vitals.config import SkinConfig
from contactless_vitals.data.synthetic import SKIN_TONES, ScriptedLandmarks, SyntheticSpec, synthetic_frames
from contactless_vitals.extract import RgbTrace
from contactless_vitals.face.regions import N_LANDMARKS, REGION_LANDMARKS, REGION_NAMES
from contactless_vitals.face.skin import region_masks, region_means, skin_mask

CFG = SkinConfig()


def patch(tone: str) -> tuple[np.ndarray, np.ndarray]:
    """Skin patch with 25% hair and a small clipped highlight."""
    rng = np.random.default_rng(0)
    image = np.array(SKIN_TONES[tone], float) + rng.normal(0, 2, (40, 40, 3))
    image[:10] = (15, 10, 8)  # hair, darker than every skin tone
    image[30:33, 30:33] = 255  # highlight
    return np.clip(image, 0, 255).astype(np.uint8), np.ones((40, 40), bool)


@pytest.mark.parametrize("tone", SKIN_TONES)
def test_mask_keeps_skin_and_drops_hair_and_highlights(tone):
    image, roi = patch(tone)
    mask = skin_mask(image, roi, CFG)
    assert not mask[:10].any()
    assert not mask[30:33, 30:33].any()
    assert mask[10:30].mean() > 0.95


def test_mask_stays_inside_the_region():
    image, roi = patch("medium")
    roi[:, 20:] = False
    assert not skin_mask(image, roi, CFG)[:, 20:].any()


def test_landmark_sets_have_valid_unique_indices():
    every = [i for indices in REGION_LANDMARKS.values() for i in indices]
    assert len(every) == len(set(every))
    assert all(0 <= i < N_LANDMARKS for i in every)


def test_region_mean_matches_the_true_skin_colour_despite_hair_in_the_region():
    spec = SyntheticSpec(duration_s=0.1, sensor_noise=1.0)
    t, frame = next(synthetic_frames(spec))
    means, counts = region_means(frame, ScriptedLandmarks(spec).detect(frame, t), CFG)
    assert np.all(counts > 100)
    # Skin texture varies by about 4%. Hair would pull the forehead far lower.
    assert np.allclose(means, np.array(spec.skin_rgb), rtol=0.05)


def test_landmarks_off_screen_give_nan_not_a_crash():
    spec = SyntheticSpec(duration_s=0.1)
    _, frame = next(synthetic_frames(spec))
    means, counts = region_means(frame, np.full((N_LANDMARKS, 2), -500.0), CFG)
    assert np.all(np.isnan(means))
    assert np.all(counts == 0)


def test_region_masks_match_region_means():
    spec = SyntheticSpec(duration_s=0.1)
    t, frame = next(synthetic_frames(spec))
    landmarks = ScriptedLandmarks(spec).detect(frame, t)
    _, counts = region_means(frame, landmarks, CFG)
    masks = region_masks(frame, landmarks, CFG)
    assert [int(masks[n].sum()) for n in REGION_NAMES] == counts.tolist()


def test_combined_trace_has_no_step_when_a_region_drops_out():
    """Regions differ in colour, so a naive average would jump here."""
    n = 100
    rgb = np.empty((n, 3, 3))
    rgb[:, 0] = (200.0, 150.0, 120.0)
    rgb[:, 1] = (150.0, 100.0, 80.0)
    rgb[:, 2] = (120.0, 80.0, 60.0)
    pixels = np.full((n, 3), 500)
    rgb[40:60, 0] = np.nan
    pixels[40:60, 0] = 0
    combined = RgbTrace(np.arange(n) / 30.0, rgb, pixels).combined()
    assert np.allclose(combined, combined[0])


def test_trace_round_trips_through_disk(tmp_path):
    trace, _ = process_video(SyntheticSpec(duration_s=6.0))
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = RgbTrace.load(path)
    assert np.array_equal(loaded.rgb, trace.rgb, equal_nan=True)
    assert loaded.regions == trace.regions
