import os
import re
import sys
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def list_images(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files.sort(key=natural_key)
    return [os.path.join(input_dir, f) for f in files]


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def write_image(path: str, image: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def ensure_gray2d(image: np.ndarray) -> np.ndarray:
    if sys.version_info >= (3, 12):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
    return image


def adjust_hsv_lightness_by_percentile(image: np.ndarray, low: int = 3, high: int = 97) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vmin = float(np.percentile(v, low))
    vmax = float(np.percentile(v, high))
    if vmax <= vmin:
        v_new = v.copy()
    else:
        v_new = ((v.astype(np.float32) - vmin) / (vmax - vmin) * 255.0)
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)
    adjusted_hsv = cv2.merge((h, s, v_new))
    adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image


def basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def extract_numeric_token(name: str) -> str:
    m = re.search(r"(\d+)", name)
    return m.group(1).zfill(3) if m else name


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image.squeeze(axis=2), cv2.COLOR_GRAY2BGR)
    return image


def resize_keep_aspect(image: np.ndarray, target_short_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if min(h, w) == target_short_side:
        return image
    if h < w:
        new_h = target_short_side
        new_w = int(w * (target_short_side / h))
    else:
        new_w = target_short_side
        new_h = int(h * (target_short_side / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def sorted_detections_xyxy(detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    return sorted(detections, key=lambda xyxy: xyxy[0])


