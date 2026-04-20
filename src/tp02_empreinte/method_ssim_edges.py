from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as compare_ssim

from .common import Decision, equalize_hist, read_image_grayscale, resize_square, binarize


@dataclass(frozen=True)
class SSIMConfig:
    size: int = 300
    bin_threshold: int = 128
    decision_threshold: float = 0.75


def preprocess(image_path: str | Path, cfg: SSIMConfig = SSIMConfig()) -> np.ndarray:
    """
    Prétraitement imposé (TP):
    - Conversion en niveaux de gris
    - Redimensionnement (300×300)
    - Égalisation histogramme
    - Binarisation (seuil = 128)
    - Extraction des contours (FIND_EDGES)

    Référence énoncé: https://www.genspark.ai/api/files/s/jHsYkqz5
    """
    img = read_image_grayscale(image_path)
    img = resize_square(img, cfg.size)
    img = equalize_hist(img)
    img = binarize(img, cfg.bin_threshold)

    pil = Image.fromarray(img)
    edges = pil.filter(ImageFilter.FIND_EDGES)
    out = np.array(edges, dtype=np.uint8)
    return out


def similarity(img_path_1: str | Path, img_path_2: str | Path, cfg: SSIMConfig = SSIMConfig()) -> float:
    img1 = preprocess(img_path_1, cfg)
    img2 = preprocess(img_path_2, cfg)
    score = compare_ssim(img1, img2, data_range=255)
    return float(score)


def verify(img_path_1: str | Path, img_path_2: str | Path, cfg: SSIMConfig = SSIMConfig()) -> Decision:
    s = similarity(img_path_1, img_path_2, cfg)
    return Decision(score=s, accepted=(s >= cfg.decision_threshold))
