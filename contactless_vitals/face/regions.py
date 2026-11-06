"""Skin regions defined by MediaPipe face-mesh landmark indices.

The forehead and cheeks are used because they are well supplied with blood, are
fairly flat, and stay still when someone talks or blinks. Each polygon stays
clear of the eyes, brows, nose and lips.

"Left" and "right" are the subject's own, so `left_cheek` appears on the right of
an unmirrored image.
"""

N_LANDMARKS = 478  # 468 face-mesh points plus 10 iris points

REGION_LANDMARKS: dict[str, tuple[int, ...]] = {
    "forehead": (69, 108, 151, 337, 299, 296, 336, 9, 107, 66),
    "left_cheek": (346, 347, 348, 329, 371, 423, 426, 427, 411, 352),
    "right_cheek": (117, 118, 119, 100, 142, 203, 206, 207, 187, 123),
}

REGION_NAMES: tuple[str, ...] = tuple(REGION_LANDMARKS)
