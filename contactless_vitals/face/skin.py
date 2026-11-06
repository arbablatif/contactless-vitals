"""Skin masking and per-region colour averaging."""

import cv2
import numpy as np

from contactless_vitals.config import SkinConfig
from contactless_vitals.face.regions import REGION_LANDMARKS

_POLY_SHIFT = 4  # OpenCV fixed-point bits: polygons keep 1/16-pixel precision


def skin_mask(crop_rgb: np.ndarray, roi: np.ndarray, cfg: SkinConfig) -> np.ndarray:
    """Keep the pixels in `roi` that look like the region's own skin.

    Most of a well-placed region is skin, so its median colour is a skin
    colour for this person under this light. Pixels far from it (hair, glasses,
    highlights, shadow) are dropped. Everything is relative to that median, so
    no skin tone is assumed.

    Args:
        crop_rgb: uint8 image, shape (H, W, 3).
        roi: boolean region mask, shape (H, W).
        cfg: thresholds.

    Returns:
        Boolean mask of shape (H, W), a subset of `roi`.
    """
    if not roi.any():
        return roi
    ycc = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

    keep = roi.copy()
    median_luma = float(np.median(luma[roi]))
    keep &= luma >= cfg.dark_ratio * median_luma
    keep &= luma <= cfg.bright_ratio * median_luma
    keep &= crop_rgb.max(axis=2) < cfg.saturation_level

    for chroma in (cr, cb):
        values = chroma[roi]
        median = float(np.median(values))
        robust_std = 1.4826 * float(np.median(np.abs(values - median)))
        keep &= np.abs(chroma - median) <= max(cfg.chroma_floor, cfg.chroma_k * robust_std)
    return keep


def region_means(
    frame_rgb: np.ndarray, landmarks_px: np.ndarray, cfg: SkinConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RGB of the skin pixels in each region for one frame.

    Averaging thousands of pixels is what makes the pulse measurable: it is far
    smaller than per-pixel noise, and averaging shrinks that noise.

    Returns:
        (means, counts) with shapes (R, 3) and (R,). A region without enough
        skin pixels has NaN means and a count of 0.
    """
    h, w = frame_rgb.shape[:2]
    means = np.full((len(REGION_LANDMARKS), 3), np.nan)
    counts = np.zeros(len(REGION_LANDMARKS), dtype=np.int64)

    for i, indices in enumerate(REGION_LANDMARKS.values()):
        crop, mask = _region_crop_and_mask(frame_rgb, landmarks_px[list(indices)], h, w, cfg)
        if crop is None:
            continue
        n = int(mask.sum())
        if n >= cfg.min_pixels:
            means[i] = crop[mask].mean(axis=0)
            counts[i] = n
    return means, counts


def _region_crop_and_mask(frame_rgb, polygon, h, w, cfg):
    """Crop to the polygon's bounding box and build its skin mask there."""
    if not np.all(np.isfinite(polygon)):
        return None, None
    x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
    x1 = min(int(np.ceil(polygon[:, 0].max())) + 1, w)
    y1 = min(int(np.ceil(polygon[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None, None

    crop = frame_rgb[y0:y1, x0:x1]
    roi = np.zeros(crop.shape[:2], dtype=np.uint8)
    points = np.rint((polygon - [x0, y0]) * (1 << _POLY_SHIFT)).astype(np.int32)
    cv2.fillPoly(roi, [points], 1, lineType=cv2.LINE_8, shift=_POLY_SHIFT)
    return crop, skin_mask(crop, roi.astype(bool), cfg)


def region_masks(
    frame_rgb: np.ndarray, landmarks_px: np.ndarray, cfg: SkinConfig
) -> dict[str, np.ndarray]:
    """Full-frame skin mask per region, for drawing overlays."""
    h, w = frame_rgb.shape[:2]
    masks = {}
    for name, indices in REGION_LANDMARKS.items():
        polygon = landmarks_px[list(indices)]
        full = np.zeros((h, w), dtype=bool)
        crop, mask = _region_crop_and_mask(frame_rgb, polygon, h, w, cfg)
        if crop is not None:
            x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
            full[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
        masks[name] = full
    return masks
