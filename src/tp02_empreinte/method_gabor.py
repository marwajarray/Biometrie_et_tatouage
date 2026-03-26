from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import Decision, read_image_grayscale, resize_square, equalize_hist, cosine_similarity


@dataclass(frozen=True)
class GaborConfig:
    size: int = 300
    ksize: int = 21
    sigmas: tuple[float, ...] = (3.0, 5.0)
    lambdas: tuple[float, ...] = (6.0, 10.0)
    gammas: tuple[float, ...] = (0.5,)
    psis: tuple[float, ...] = (0.0,)
    thetas: int = 8
    decision_threshold: float = 0.90  # similarité cosinus, à ajuster


def _gabor_features(img: np.ndarray, cfg: GaborConfig) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0

    feats: list[float] = []
    for sigma in cfg.sigmas:
        for lambd in cfg.lambdas:
            for gamma in cfg.gammas:
                for psi in cfg.psis:
                    for t in range(cfg.thetas):
                        theta = t * np.pi / cfg.thetas
                        kernel = cv2.getGaborKernel(
                            (cfg.ksize, cfg.ksize),
                            sigma=sigma,
                            theta=theta,
                            lambd=lambd,
                            gamma=gamma,
                            psi=psi,
                            ktype=cv2.CV_32F,
                        )
                        resp = cv2.filter2D(img_f, cv2.CV_32F, kernel)
                        feats.append(float(resp.mean()))
                        feats.append(float(resp.var()))

    return np.array(feats, dtype=np.float32)


def similarity(img_path_1: str | Path, img_path_2: str | Path, cfg: GaborConfig = GaborConfig()) -> float:
    img1 = equalize_hist(resize_square(read_image_grayscale(img_path_1), cfg.size))
    img2 = equalize_hist(resize_square(read_image_grayscale(img_path_2), cfg.size))

    f1 = _gabor_features(img1, cfg)
    f2 = _gabor_features(img2, cfg)
    return cosine_similarity(f1, f2)


def verify(img_path_1: str | Path, img_path_2: str | Path, cfg: GaborConfig = GaborConfig()) -> Decision:
    s = similarity(img_path_1, img_path_2, cfg)
    return Decision(score=s, accepted=(s >= cfg.decision_threshold))
