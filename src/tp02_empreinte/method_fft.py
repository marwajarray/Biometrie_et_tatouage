from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import Decision, read_image_grayscale, resize_square, equalize_hist, cosine_similarity


@dataclass(frozen=True)
class FFTConfig:
    size: int = 300
    keep_low_freq: int = 80  # patch central (80x80)
    decision_threshold: float = 0.85  # similarité cosinus, à ajuster


def _fft_signature(img: np.ndarray, cfg: FFTConfig) -> np.ndarray:
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    c = mag.shape[0] // 2
    k = cfg.keep_low_freq // 2
    patch = mag[c - k : c + k, c - k : c + k]
    return patch.flatten()


def similarity(img_path_1: str | Path, img_path_2: str | Path, cfg: FFTConfig = FFTConfig()) -> float:
    img1 = equalize_hist(resize_square(read_image_grayscale(img_path_1), cfg.size))
    img2 = equalize_hist(resize_square(read_image_grayscale(img_path_2), cfg.size))

    s1 = _fft_signature(img1, cfg)
    s2 = _fft_signature(img2, cfg)
    return cosine_similarity(s1, s2)


def verify(img_path_1: str | Path, img_path_2: str | Path, cfg: FFTConfig = FFTConfig()) -> Decision:
    s = similarity(img_path_1, img_path_2, cfg)
    return Decision(score=s, accepted=(s >= cfg.decision_threshold))
