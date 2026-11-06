"""Face landmark tracking with the MediaPipe Tasks FaceLandmarker.

MediaPipe is imported lazily, so `--mock` runs and the fast tests never load it
or need its model file.
"""

import hashlib
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

# A versioned URL, not "latest", so the checksum below stays valid.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


class LandmarkProvider(Protocol):
    """Anything that can locate face landmarks in a frame."""

    def detect(self, frame_rgb: np.ndarray, t_s: float) -> np.ndarray | None:
        """Return landmarks in pixels, shape (478, 2), or None if no face is found."""

    def close(self) -> None:
        """Release resources."""


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the 3.7 MB model on first use and verify its checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        path.unlink()
        raise RuntimeError("Face model checksum mismatch; removed the file. Try again.")
    return path


class MediaPipeLandmarks:
    """Tracks one face through a video. Use as a context manager or call close()."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ensure_model(model_path))
            ),
            # VIDEO mode tracks between frames, which is steadier and faster
            # than detecting the face from scratch every time.
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_rgb: np.ndarray, t_s: float) -> np.ndarray | None:
        # MediaPipe requires strictly increasing whole-millisecond timestamps.
        t_ms = max(int(round(t_s * 1000)), self._last_ms + 1)
        self._last_ms = t_ms
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect_for_video(image, t_ms)
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
