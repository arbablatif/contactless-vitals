"""Step one of the pipeline: frames in, per-frame mean RGB per region out."""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from contactless_vitals.config import SkinConfig
from contactless_vitals.face.landmarks import LandmarkProvider
from contactless_vitals.face.regions import REGION_NAMES
from contactless_vitals.face.skin import region_means


@dataclass
class RgbTrace:
    """Colour over time for each face region.

    Attributes:
        t: frame timestamps in seconds, shape (T,).
        rgb: mean colour, shape (T, R, 3). NaN where a region was unusable.
        pixels: skin pixels averaged, shape (T, R).
        regions: region names, length R.
    """

    t: np.ndarray
    rgb: np.ndarray
    pixels: np.ndarray
    regions: tuple[str, ...] = REGION_NAMES

    def region(self, name: str) -> np.ndarray:
        """RGB trace of one region, shape (T, 3)."""
        return self.rgb[:, self.regions.index(name), :]

    def combined(self) -> np.ndarray:
        """All regions merged into one trace, weighted by pixel count, shape (T, 3).

        Each region is first divided by its own average colour. Otherwise a
        region dropping out for a moment would make the mix jump in colour, and
        that step would look like a signal.
        """
        region_mean = np.nanmean(self.rgb, axis=0, keepdims=True)
        relative = self.rgb / region_mean
        weights = np.where(np.isfinite(relative[..., 0]), self.pixels, 0).astype(float)
        total = weights.sum(axis=1, keepdims=True)
        blended = np.nansum(relative * weights[..., None], axis=1)
        blended = np.divide(
            blended, total, out=np.full_like(blended, np.nan), where=total > 0
        )
        overall = np.average(
            region_mean[0], axis=0, weights=np.maximum(self.pixels.sum(axis=0), 1)
        )
        return blended * overall

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=self.t, rgb=self.rgb, pixels=self.pixels, regions=np.array(self.regions)
        )

    @classmethod
    def load(cls, path: Path) -> "RgbTrace":
        with np.load(path) as f:
            return cls(f["t"], f["rgb"], f["pixels"], tuple(str(r) for r in f["regions"]))


def extract_rgb_trace(
    frames: Iterable[tuple[float, np.ndarray]],
    provider: LandmarkProvider,
    cfg: SkinConfig | None = None,
) -> RgbTrace:
    """Track the face through `frames` and record each region's mean colour.

    Frames with no detected face are kept as NaN rows, so timing stays honest
    and later stages can decide what to do with the gap.
    """
    cfg = cfg or SkinConfig()
    times, colours, counts = [], [], []
    for t, frame in frames:
        landmarks = provider.detect(frame, t)
        if landmarks is None:
            means = np.full((len(REGION_NAMES), 3), np.nan)
            n = np.zeros(len(REGION_NAMES), dtype=np.int64)
        else:
            means, n = region_means(frame, landmarks, cfg)
        times.append(t)
        colours.append(means)
        counts.append(n)
    if not times:
        raise ValueError("No frames to process.")
    return RgbTrace(np.array(times), np.stack(colours), np.stack(counts))


def cached_rgb_trace(
    video_path: Path,
    cache_dir: Path,
    compute: Callable[[], RgbTrace],
    cfg: SkinConfig | None = None,
) -> RgbTrace:
    """Reuse a saved trace, since face tracking is the slowest step.

    The key covers the file's identity and the skin settings, so changing
    either recomputes rather than returning stale results.
    """
    cfg = cfg or SkinConfig()
    stat = video_path.stat()
    key = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{asdict(cfg)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache_dir / f"{video_path.parent.name}_{digest}.npz"
    if path.exists():
        return RgbTrace.load(path)
    trace = compute()
    trace.save(path)
    return trace
