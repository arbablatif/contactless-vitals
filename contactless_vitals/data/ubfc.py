"""Loader for UBFC-rPPG DATASET_2.

The videos are distributed through links on the authors' page, not a stable
download API, so this module does not fetch them. It tells you where to get them,
then checks and reads what you downloaded.

Expected layout, one folder per subject:

    <root>/subject1/vid.avi
    <root>/subject1/ground_truth.txt

`ground_truth.txt` holds three rows of space-separated numbers: the PPG waveform,
heart rate from the oximeter, and time in seconds. Row one is confirmed by
rPPG-Toolbox. Rows two and three come from a secondary source, so the parser
checks them and stops with a clear message if they do not look right.
"""

import os
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

DATASET_URL = "https://sites.google.com/view/ybenezeth/ubfcrppg"
CITATION = (
    "S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "
    '"Unsupervised skin tissue segmentation for remote photoplethysmography", '
    "Pattern Recognition Letters, 2017."
)
ROOT_ENV_VAR = "UBFC_RPPG_ROOT"
VIDEO_NAMES = ("vid.avi", "vid.mp4")
GROUND_TRUTH_NAME = "ground_truth.txt"


class GroundTruthFormatError(ValueError):
    """The ground-truth file is not in the expected three-row format."""


@dataclass(frozen=True)
class Subject:
    name: str
    video_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class GroundTruth:
    """Contact sensor recording: waveform, heart rate (bpm) and time (seconds)."""

    ppg: np.ndarray
    hr_bpm: np.ndarray
    t: np.ndarray


@dataclass(frozen=True)
class VideoInfo:
    n_frames: int
    fps: float
    width: int
    height: int


def access_instructions() -> str:
    """How to obtain the dataset, shown when it is missing."""
    return (
        "UBFC-rPPG is not bundled and is never committed to this repository.\n"
        f"1. Download DATASET_2 from the authors' page: {DATASET_URL}\n"
        "2. Check the conditions of use on that page. The authors ask you to cite:\n"
        f"   {CITATION}\n"
        f"3. Set {ROOT_ENV_VAR} in .env to the folder holding subject1/, subject2/, ...\n"
        "4. Run: python -m contactless_vitals check-data"
    )


def resolve_root(root: str | Path | None = None) -> Path | None:
    """Dataset folder from the argument, then the environment or .env."""
    if root is not None:
        return Path(root)
    load_dotenv()
    value = os.environ.get(ROOT_ENV_VAR)
    return Path(value) if value else None


def _natural_key(name: str) -> list:
    """Sort subject2 before subject10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def discover_subjects(root: Path) -> list[Subject]:
    """Find every subject folder that has both a video and a ground-truth file."""
    subjects = []
    for folder in sorted(root.glob("subject*"), key=lambda p: _natural_key(p.name)):
        video = next((folder / n for n in VIDEO_NAMES if (folder / n).exists()), None)
        truth = folder / GROUND_TRUTH_NAME
        if folder.is_dir() and video is not None and truth.exists():
            subjects.append(Subject(folder.name, video, truth))
    return subjects


def read_ground_truth(path: Path) -> GroundTruth:
    """Parse a ground-truth file, refusing anything that does not fit the format."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise GroundTruthFormatError(
            f"{path}: expected 3 rows (PPG, heart rate, time), found {len(lines)}."
        )
    try:
        rows = [np.array(line.split(), dtype=float) for line in lines[:3]]
    except ValueError as error:
        raise GroundTruthFormatError(f"{path}: non-numeric value ({error}).") from error

    lengths = [row.size for row in rows]
    if len(set(lengths)) != 1:
        raise GroundTruthFormatError(f"{path}: rows differ in length: {lengths}.")
    if not all(np.all(np.isfinite(row)) for row in rows):
        raise GroundTruthFormatError(f"{path}: contains NaN or infinite values.")
    if np.any(np.diff(rows[2]) <= 0):
        raise GroundTruthFormatError(f"{path}: row 3 (time) does not always increase.")
    return GroundTruth(ppg=rows[0], hr_bpm=rows[1], t=rows[2])


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"OpenCV could not open {path}.")
    info = VideoInfo(
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return info


def iter_video_frames(
    path: Path, times: np.ndarray | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, RGB frame) one at a time, so long videos never fill memory.

    Args:
        path: video file.
        times: timestamp per frame. If None, uses frame index divided by frame rate.
    """
    fps = video_info(path).fps
    capture = cv2.VideoCapture(str(path))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (times is not None and index >= times.size):
                break
            t = float(times[index]) if times is not None else index / fps
            yield t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def frame_timestamps(info: VideoInfo, truth: GroundTruth) -> np.ndarray | None:
    """Use recorded timestamps if they fit the video; otherwise None.

    Recorded times capture real frame jitter, so they are preferred. They only
    apply if there is one per frame and they span the video's length. Any other
    case falls back to the nominal frame rate, with a warning.
    """
    expected = info.n_frames / info.fps if info.fps > 0 else 0.0
    span = float(truth.t[-1] - truth.t[0])
    if truth.t.size == info.n_frames and 0.8 * expected <= span <= 1.2 * expected:
        return truth.t
    warnings.warn(
        f"Ground-truth time ({truth.t.size} samples over {span:.1f}) does not match "
        f"the video ({info.n_frames} frames, {expected:.1f} s). "
        "Using the nominal frame rate.",
        stacklevel=2,
    )
    return None


def subject_frames(subject: Subject) -> Iterator[tuple[float, np.ndarray]]:
    """Frames of one subject, on the best timestamps available."""
    info = video_info(subject.video_path)
    truth = read_ground_truth(subject.ground_truth_path)
    return iter_video_frames(subject.video_path, frame_timestamps(info, truth))


def check_dataset(root: Path) -> tuple[list[str], bool]:
    """Check a downloaded dataset. Returns (report lines, all_ok)."""
    if not root.is_dir():
        return [f"Folder not found: {root}"], False
    subjects = discover_subjects(root)
    if not subjects:
        return [f"No subject folders with a video and {GROUND_TRUTH_NAME} in {root}"], False

    lines = [f"Found {len(subjects)} subjects in {root}"]
    ok = True
    for subject in subjects:
        try:
            info = video_info(subject.video_path)
            truth = read_ground_truth(subject.ground_truth_path)
            span = truth.t[-1] - truth.t[0]
            lines.append(
                f"  {subject.name}: {info.n_frames} frames at {info.fps:.1f} fps, "
                f"{info.width}x{info.height}; {truth.ppg.size} PPG samples over {span:.1f} s"
            )
        except (OSError, GroundTruthFormatError) as error:
            lines.append(f"  {subject.name}: PROBLEM: {error}")
            ok = False
    return lines, ok
