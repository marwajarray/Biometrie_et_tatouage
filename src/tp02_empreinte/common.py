from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Decision:
    score: float
    accepted: bool


def read_image_grayscale(image_path: str | Path) -> np.ndarray:
    path = str(image_path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {path}")
    return img


def resize_square(img: np.ndarray, size: int = 300) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def equalize_hist(img_gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img_gray)


def binarize(img_gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    _, out = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    return out


def normalize_01(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_01(a.astype(np.float64))
    b = normalize_01(b.astype(np.float64))
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
