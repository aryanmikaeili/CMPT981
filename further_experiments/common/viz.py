"""Visualization helper shared across the further-experiment scripts.

The original `train1.py:Trainer.visualize` hardcodes the title position
at x=100 with font scale 0.6, which overflows the 256-pixel canvas for
any title longer than "PSNR: XX.XX" (so the per-branch PSNR values were
silently clipped from the saved PNGs).  This helper auto-shrinks the
font scale until the title fits, and writes the value cleanly.
"""

import os

import cv2
import numpy as np


def save_pair_with_text(image_rgb: np.ndarray,
                        text: str,
                        out_path: str,
                        header_h: int = 50) -> None:
    """Save `image_rgb` (uint8 HxWx3 in RGB order) under a centered title.

    Parameters
    ----------
    image_rgb : np.ndarray
        Stacked GT|prediction (or any image) in RGB order, dtype uint8.
    text : str
        Title to render inside the white header bar above `image_rgb`.
    out_path : str
        Destination PNG path. Parent directories are created as needed.
    header_h : int
        Height in pixels of the white header that holds the title.
    """
    h, w, _ = image_rgb.shape
    canvas = np.full((h + header_h, w, 3), 255, dtype=np.uint8)
    canvas[header_h:, :, :] = image_rgb
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_bgr = (255, 0, 0)
    thickness = 1
    margin = 6

    scale = 0.6
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    while text_w > (w - 2 * margin) and scale > 0.25:
        scale -= 0.05
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(margin, (w - text_w) // 2)
    y = (header_h + text_h) // 2
    cv2.putText(canvas, text, (x, y), font, scale, color_bgr, thickness,
                lineType=cv2.LINE_AA)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(out_path, canvas)
