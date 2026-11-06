"""UBFC-rPPG loader, on a small fake dataset in the expected layout."""

import numpy as np
import pytest

from contactless_vitals import cli
from contactless_vitals.data import ubfc
from contactless_vitals.data.synthetic import (
    ScriptedLandmarks,
    SyntheticSpec,
    frame_times,
    pulse_waveform,
    synthetic_frames,
    write_lossless_video,
)
from contactless_vitals.extract import cached_rgb_trace, extract_rgb_trace
from contactless_vitals.pipeline import COMBINED, run_rppg
from contactless_vitals.quality import (
    correlation_with_lag,
    shifted_correlations,
    snr_at_known_frequency,
    windowed_snr,
)

SPEC = SyntheticSpec(duration_s=8.0)


def write_truth(path, n=240, hr=72.0, fs=30.0):
    t = np.arange(n) / fs
    rows = [pulse_waveform(t, SPEC.pulse_hz), np.full(n, hr), t]
    path.write_text("\n".join(" ".join(f"{v:.6e}" for v in row) for row in rows) + "\n")


@pytest.fixture
def dataset(tmp_path):
    for name in ("subject10", "subject2"):
        folder = tmp_path / name
        folder.mkdir()
        write_lossless_video(synthetic_frames(SPEC), folder / "vid.avi", SPEC.fps)
        write_truth(folder / "ground_truth.txt")
    (tmp_path / "subject3").mkdir()  # incomplete: must be skipped
    return tmp_path


def test_subjects_are_found_in_natural_order_and_incomplete_ones_skipped(dataset):
    assert [s.name for s in ubfc.discover_subjects(dataset)] == ["subject2", "subject10"]


def test_ground_truth_rows_are_read_in_order(dataset):
    truth = ubfc.read_ground_truth(dataset / "subject2" / "ground_truth.txt")
    assert truth.ppg.size == truth.hr_bpm.size == truth.t.size == 240
    assert np.all(truth.hr_bpm == 72.0)
    assert truth.t[-1] == pytest.approx(239 / 30)


@pytest.mark.parametrize(
    "text, message",
    [
        ("1 2 3\n4 5 6\n", "expected 3 rows"),
        ("1 2 3\n4 5\n0 1 2\n", "differ in length"),
        ("1 2 x\n4 5 6\n0 1 2\n", "non-numeric"),
        ("1 2 nan\n4 5 6\n0 1 2\n", "NaN"),
        ("1 2 3\n4 5 6\n0 2 1\n", "goes backwards"),
    ],
)
def test_bad_ground_truth_stops_with_a_clear_message(tmp_path, text, message):
    path = tmp_path / "ground_truth.txt"
    path.write_text(text)
    with pytest.raises(ubfc.GroundTruthFormatError, match=message):
        ubfc.read_ground_truth(path)


def test_repeated_time_is_accepted_as_in_real_files(tmp_path):
    path = tmp_path / "ground_truth.txt"
    path.write_text("1 2 3\n70 70 70\n0 1 1\n")
    assert ubfc.read_ground_truth(path).t.tolist() == [0.0, 1.0, 1.0]


def test_video_round_trip_is_pixel_exact(tmp_path):
    frames = list(synthetic_frames(SyntheticSpec(duration_s=0.3)))
    path = tmp_path / "vid.avi"
    write_lossless_video(iter(frames), path, 30.0)
    read = list(ubfc.iter_video_frames(path))
    assert len(read) == len(frames)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(frames, read))


def test_frames_use_recorded_timestamps_when_they_fit(dataset):
    subject = ubfc.discover_subjects(dataset)[0]
    truth = ubfc.read_ground_truth(subject.ground_truth_path)
    times = [t for t, _ in ubfc.subject_frames(subject)]
    assert np.allclose(times, truth.t)


def test_mismatched_timestamps_fall_back_with_a_warning(dataset):
    info = ubfc.VideoInfo(240, 30.0, 160, 120)
    wrong_units = ubfc.GroundTruth(np.zeros(240), np.zeros(240), np.arange(240) * 33.3)
    with pytest.warns(UserWarning, match="nominal frame rate"):
        assert ubfc.frame_timestamps(info, wrong_units) is None


def test_full_pipeline_from_files_on_disk(dataset):
    subject = ubfc.discover_subjects(dataset)[0]
    trace = extract_rgb_trace(ubfc.subject_frames(subject), ScriptedLandmarks(SPEC))
    result = run_rppg(trace)
    x = result.waveforms[COMBINED]["pos"]
    assert snr_at_known_frequency(x, result.fs, SPEC.pulse_hz) > 10.0
    reference = cli.reference_on_grid(
        result, ubfc.read_ground_truth(subject.ground_truth_path), cli.SignalConfig()
    )
    assert correlation_with_lag(x, reference, result.fs)[0] > 0.9


def test_windowed_snr_uses_the_sensor_rate_and_skips_bad_windows():
    fs = 30.0
    t = np.arange(900) / fs
    x = np.sin(2 * np.pi * 1.2 * t)
    x[850] = np.nan  # spoils the last window
    rate = np.full(t.size, 1.2)
    scores = windowed_snr(x, t, fs, t, rate)
    assert scores.size == 4
    assert np.all(scores > 20)
    assert windowed_snr(x, t, fs, t, np.zeros(t.size)).size == 0


def test_trace_cache_computes_once(dataset):
    subject = ubfc.discover_subjects(dataset)[0]
    calls = []

    def compute():
        calls.append(1)
        return extract_rgb_trace(ubfc.subject_frames(subject), ScriptedLandmarks(SPEC))

    cache = dataset / "cache"
    first = cached_rgb_trace(subject.video_path, cache, compute)
    second = cached_rgb_trace(subject.video_path, cache, compute)
    assert len(calls) == 1
    assert np.array_equal(first.rgb, second.rgb, equal_nan=True)


def test_check_dataset_reports_subjects_and_problems(dataset):
    lines, ok = ubfc.check_dataset(dataset)
    assert ok and "Found 2 subjects" in lines[0]
    (dataset / "subject2" / "ground_truth.txt").write_text("1 2\n")
    lines, ok = ubfc.check_dataset(dataset)
    assert not ok and any("PROBLEM" in line for line in lines)


def test_missing_folder_is_reported(tmp_path):
    lines, ok = ubfc.check_dataset(tmp_path / "nowhere")
    assert not ok


def test_instructions_name_the_source_and_citation():
    text = ubfc.access_instructions()
    assert ubfc.DATASET_URL in text and "Bobbia" in text and ubfc.ROOT_ENV_VAR in text


def test_frame_times_match_spec():
    assert frame_times(SPEC).size == 240


def test_shifted_correlations_sit_below_a_true_match():
    rng = np.random.default_rng(0)
    fs = 30.0
    t = np.arange(1500) / fs
    # A drifting rhythm, so a shifted copy really is out of step.
    phase = 2 * np.pi * np.cumsum(1.5 + 0.3 * np.sin(2 * np.pi * 0.07 * t)) / fs
    reference = np.sin(phase)
    x = reference + 0.3 * rng.standard_normal(t.size)
    chance = shifted_correlations(x, reference, fs)
    assert chance.size == 22
    assert np.median(chance) < correlation_with_lag(x, reference, fs)[0] - 0.2


def test_snr_does_not_depend_on_float_noise_in_the_rate():
    # At 1.2 Hz, bins sit exactly on the edge of the signal band.
    rng = np.random.default_rng(0)
    t = np.arange(900) / 30.0
    x = np.sin(2 * np.pi * 1.2 * t) + 0.5 * rng.standard_normal(t.size)
    scores = [snr_at_known_frequency(x, 30.0 + d, 1.2) for d in (-1e-13, 0.0, 1e-13)]
    assert max(scores) - min(scores) < 1e-6
