from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import Decision, read_image_grayscale, resize_square, equalize_hist


@dataclass(frozen=True)
class ORBConfig:
    size: int = 300
    nfeatures: int = 800
    ratio_test: float = 0.75
    decision_threshold: float = 0.25  # à ajuster expérimentalement


def _preprocess_orb(image_path: str | Path, cfg: ORBConfig) -> np.ndarray:
    img = read_image_grayscale(image_path)
    img = resize_square(img, cfg.size)
    img = equalize_hist(img)
    return img


def similarity(img_path_1: str | Path, img_path_2: str | Path, cfg: ORBConfig = ORBConfig()) -> float:
    img1 = _preprocess_orb(img_path_1, cfg)
    img2 = _preprocess_orb(img_path_2, cfg)

    orb = cv2.ORB_create(nfeatures=cfg.nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = 0
    for m, n in matches:
        if m.distance < cfg.ratio_test * n.distance:
            good += 1

    denom = max(len(kp1), len(kp2), 1)
    score = good / denom
    return float(np.clip(score, 0.0, 1.0))


def verify(img_path_1: str | Path, img_path_2: str | Path, cfg: ORBConfig = ORBConfig()) -> Decision:
    s = similarity(img_path_1, img_path_2, cfg)
    return Decision(score=s, accepted=(s >= cfg.decision_threshold))
