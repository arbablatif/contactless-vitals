"""Synthetic face videos with a known pulse.

The pixel model follows the skin-reflection model of Wang et al. (2017):

    pixel_k(t) = I(t) * [ d_k * (1 - A * pbv_k * p(t)) + s(t) * spec_k ] + noise

where k is the colour channel, I the light intensity, d the skin colour, p the
pulse, pbv the pulse's colour signature, s the specular (shine) strength.

Because this is the model the algorithms assume, these videos prove the code is
correct. They do not show how well it works on real people.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from contactless_vitals.face.regions import N_LANDMARKS, REGION_LANDMARKS

# Relative pulse strength in R, G, B (de Haan and Jeanne 2013).
PULSE_SIGNATURE = np.array([0.33, 0.77, 0.53])

SKIN_TONES = {
    "light": (224, 172, 140),
    "medium": (170, 115, 85),
    "dark": (90, 58, 42),
}


@dataclass(frozen=True)
class SyntheticSpec:
    """Everything that defines a synthetic video. All randomness comes from `seed`."""

    pulse_hz: float = 1.2
    pulse_amplitude: float = 0.008  # fractional change at full pulse signature
    duration_s: float = 20.0
    fps: float = 30.0
    width: int = 160
    height: int = 120
    skin_rgb: tuple[int, int, int] = SKIN_TONES["medium"]
    sensor_noise: float = 2.0  # per-pixel noise, in 8-bit levels
    light_drift_depth: float = 0.0  # slow lighting change, fractional
    light_drift_hz: float = 0.1
    brightness_depth: float = 0.0  # brightness flicker, fractional
    brightness_hz: float = 1.9  # placed inside the pulse band on purpose
    specular_depth: float = 0.0  # white shine as a fraction of full scale
    specular_hz: float = 1.9
    sway_px: float = 0.0  # side-to-side head motion
    sway_hz: float = 0.9
    landmark_jitter_px: float = 0.0  # tracking noise on landmark positions
    timing_jitter_s: float = 0.0  # uneven frame timestamps
    dropped_frames: tuple[tuple[int, int], ...] = ()  # (start, stop) with no face
    seed: int = 0


# Named conditions shared by the unit tests and evaluate.py.
SCENARIOS: dict[str, SyntheticSpec] = {
    "clean": SyntheticSpec(),
    "slow_lighting_drift": SyntheticSpec(light_drift_depth=0.15),
    "in_band_brightness": SyntheticSpec(brightness_depth=0.03),
    "in_band_specular": SyntheticSpec(specular_depth=0.015),
    "head_sway": SyntheticSpec(sway_px=5.0, landmark_jitter_px=0.3),
}


def pulse_waveform(t: np.ndarray, pulse_hz: float) -> np.ndarray:
    """Pulse shape: a fundamental plus a second harmonic, so it is not a pure sine."""
    phase = 2.0 * np.pi * pulse_hz * np.asarray(t, dtype=float)
    return (np.sin(phase) + 0.35 * np.sin(2.0 * phase + 0.8)) / 1.35


def frame_times(spec: SyntheticSpec) -> np.ndarray:
    """Frame timestamps, with optional jitter that keeps them increasing."""
    n = int(round(spec.duration_s * spec.fps))
    t = np.arange(n) / spec.fps
    if spec.timing_jitter_s > 0:
        rng = np.random.default_rng(spec.seed + 1)
        t = np.sort(t + rng.uniform(-1, 1, n) * spec.timing_jitter_s)
    return t


def _sway(spec: SyntheticSpec, t: float) -> float:
    return spec.sway_px * float(np.sin(2.0 * np.pi * spec.sway_hz * t))


def _layout(spec: SyntheticSpec) -> dict[str, tuple[float, float, float, float]]:
    """Face geometry as (centre x, centre y, radius x, radius y) in pixels."""
    w, h = spec.width, spec.height
    cx, cy, ax, ay = 0.5 * w, 0.52 * h, 0.3 * w, 0.42 * h
    return {
        "face": (cx, cy, ax, ay),
        "forehead": (cx, cy - 0.55 * ay, 0.6 * ax, 0.24 * ay),
        "right_cheek": (cx - 0.5 * ax, cy + 0.2 * ay, 0.3 * ax, 0.25 * ay),
        "left_cheek": (cx + 0.5 * ax, cy + 0.2 * ay, 0.3 * ax, 0.25 * ay),
    }


def _ellipse(xx, yy, geometry) -> np.ndarray:
    cx, cy, ax, ay = geometry
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0


def _base_image(spec: SyntheticSpec) -> tuple[np.ndarray, np.ndarray]:
    """Still face: returns (float RGB image, float skin weight map)."""
    w, h = spec.width, spec.height
    layout = _layout(spec)
    cx, cy, ax, ay = layout["face"]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rng = np.random.default_rng(spec.seed)

    image = np.empty((h, w, 3))
    image[:] = (70.0, 85.0, 105.0)  # background
    skin = _ellipse(xx, yy, layout["face"])

    # Gentle blotchy texture, so that motion actually changes pixel values.
    blotches = cv2.resize(
        rng.normal(0.0, 1.0, (h // 10, w // 10)), (w, h), interpolation=cv2.INTER_CUBIC
    )
    shading = 1.0 + 0.04 * blotches
    image[skin] = np.asarray(spec.skin_rgb, float) * shading[skin][:, None]

    # Non-skin features. Hair reaches into the forehead region on purpose,
    # so the skin mask has something to reject.
    hair = skin & (yy < cy - 0.70 * ay)
    eyes = _ellipse(xx, yy, (cx - 0.45 * ax, cy - 0.15 * ay, 0.2 * ax, 0.08 * ay)) | (
        _ellipse(xx, yy, (cx + 0.45 * ax, cy - 0.15 * ay, 0.2 * ax, 0.08 * ay))
    )
    mouth = _ellipse(xx, yy, (cx, cy + 0.62 * ay, 0.35 * ax, 0.09 * ay))
    image[hair] = (35.0, 25.0, 20.0)
    image[eyes] = (240.0, 240.0, 240.0)
    image[mouth] = (150.0, 45.0, 55.0)
    weight = (skin & ~hair & ~eyes & ~mouth).astype(float)
    return image, weight


def synthetic_frames(spec: SyntheticSpec) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, uint8 RGB frame) pairs for the whole video."""
    image, weight = _base_image(spec)
    rng = np.random.default_rng(spec.seed + 2)
    size = (spec.width, spec.height)

    for t in frame_times(spec):
        shift = np.array([[1.0, 0.0, _sway(spec, t)], [0.0, 1.0, 0.0]])
        moved = cv2.warpAffine(
            image, shift, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        skin = cv2.warpAffine(weight, shift, size, flags=cv2.INTER_LINEAR)[..., None]

        light = (
            1.0
            + spec.light_drift_depth * np.sin(2 * np.pi * spec.light_drift_hz * t)
            + spec.brightness_depth * np.sin(2 * np.pi * spec.brightness_hz * t)
        )
        pulse = spec.pulse_amplitude * PULSE_SIGNATURE * pulse_waveform(t, spec.pulse_hz)
        shine = spec.specular_depth * (1 + np.sin(2 * np.pi * spec.specular_hz * t)) * 255.0

        frame = light * (moved * (1.0 - skin * pulse) + skin * shine)
        frame = frame + rng.normal(0.0, spec.sensor_noise, frame.shape)
        # Sensor noise before rounding matters: the pulse is only 1 to 2 grey
        # levels, and without noise 8-bit rounding would erase it.
        yield float(t), np.clip(np.rint(frame), 0, 255).astype(np.uint8)


class ScriptedLandmarks:
    """Landmark provider for procedural faces, in place of MediaPipe.

    MediaPipe will not detect a drawn face, so this places each region's
    landmark indices around the region's known position, following the head.
    Only those indices are filled in; the rest stay at zero.
    """

    def __init__(self, spec: SyntheticSpec):
        self.spec = spec
        self._rng = np.random.default_rng(spec.seed + 3)
        self._template = np.zeros((N_LANDMARKS, 2))
        layout = _layout(spec)
        for name, indices in REGION_LANDMARKS.items():
            cx, cy, ax, ay = layout[name]
            theta = np.linspace(0.0, 2.0 * np.pi, len(indices), endpoint=False)
            self._template[list(indices), 0] = cx + ax * np.cos(theta)
            self._template[list(indices), 1] = cy + ay * np.sin(theta)
        self._frame_index = -1

    def detect(self, frame_rgb: np.ndarray, t_s: float) -> np.ndarray | None:
        self._frame_index += 1
        if any(a <= self._frame_index < b for a, b in self.spec.dropped_frames):
            return None
        points = self._template + np.array([_sway(self.spec, t_s), 0.0])
        if self.spec.landmark_jitter_px > 0:
            points = points + self._rng.normal(
                0, self.spec.landmark_jitter_px, points.shape
            )
        return points

    def close(self) -> None:
        """Nothing to release; matches the provider interface."""


def animate_portrait(
    image_rgb: np.ndarray, spec: SyntheticSpec
) -> Iterator[tuple[float, np.ndarray]]:
    """Add a known pulse to a still photo, for testing with real MediaPipe.

    The pulse is applied to the whole picture, not just skin. That does not
    affect the test, because only skin regions are measured.
    """
    rng = np.random.default_rng(spec.seed + 4)
    h, w = image_rgb.shape[:2]
    base = image_rgb.astype(float)
    for t in frame_times(spec):
        shift = np.array([[1.0, 0.0, _sway(spec, t)], [0.0, 1.0, 0.0]])
        moved = cv2.warpAffine(
            base, shift, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        pulse = spec.pulse_amplitude * PULSE_SIGNATURE * pulse_waveform(t, spec.pulse_hz)
        frame = moved * (1.0 - pulse) + rng.normal(0.0, spec.sensor_noise, moved.shape)
        yield float(t), np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def write_lossless_video(
    frames: Iterator[tuple[float, np.ndarray]], path: Path, fps: float
) -> int:
    """Write frames with the lossless FFV1 codec and return the frame count.

    Lossy codecs are not an option: MJPG moved pixels by up to 3 levels in
    testing, more than the pulse itself.
    """
    writer = None
    count = 0
    for _, frame in frames:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open an FFV1 video writer.")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        count += 1
    if writer is not None:
        writer.release()
    return count


def with_changes(spec: SyntheticSpec, **changes) -> SyntheticSpec:
    """Copy a spec with some fields changed."""
    return dataclasses.replace(spec, **changes)
