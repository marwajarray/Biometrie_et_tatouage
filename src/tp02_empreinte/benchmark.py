from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .method_ssim_edges import similarity as ssim_similarity, SSIMConfig
from .method_orb import similarity as orb_similarity, ORBConfig
from .method_fft import similarity as fft_similarity, FFTConfig
from .method_gabor import similarity as gabor_similarity, GaborConfig


@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return float(accuracy), float(precision), float(recall), float(f1)


def load_pairs_csv(pairs_csv: str | Path) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    with open(pairs_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["ref"], row["probe"], int(row["label"])))
    if not pairs:
        raise ValueError("Aucune paire trouvée dans le CSV. Colonnes attendues: ref, probe, label")
    return pairs


def get_similarity_fn(method: str) -> tuple[Callable[[str, str], float], float]:
    method = method.lower().strip()
    if method == "ssim":
        cfg = SSIMConfig()
        return (lambda a, b: ssim_similarity(a, b, cfg)), cfg.decision_threshold
    if method == "orb":
        cfg = ORBConfig()
        return (lambda a, b: orb_similarity(a, b, cfg)), cfg.decision_threshold
    if method == "fft":
        cfg = FFTConfig()
        return (lambda a, b: fft_similarity(a, b, cfg)), cfg.decision_threshold
    if method == "gabor":
        cfg = GaborConfig()
        return (lambda a, b: gabor_similarity(a, b, cfg)), cfg.decision_threshold
    raise ValueError("Méthode inconnue. Choix: ssim, orb, fft, gabor")


def run_benchmark(pairs: list[tuple[str, str, int]], method: str, threshold: float | None = None) -> BenchmarkResult:
    sim_fn, default_thr = get_similarity_fn(method)
    thr = default_thr if threshold is None else float(threshold)

    y_true = np.array([lab for _, _, lab in pairs], dtype=np.int32)
    scores = np.array([sim_fn(a, b) for a, b, _ in pairs], dtype=np.float64)
    y_pred = (scores >= thr).astype(np.int32)

    acc, prec, rec, f1 = _metrics(y_true, y_pred)
    return BenchmarkResult(method=method, threshold=thr, accuracy=acc, precision=prec, recall=rec, f1=f1)
