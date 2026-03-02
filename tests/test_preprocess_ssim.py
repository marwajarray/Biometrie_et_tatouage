from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tp02_empreinte.method_ssim_edges import preprocess, similarity


def _save_img(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_preprocess_shape_and_type(tmp_path: Path):
    img = np.zeros((500, 500), dtype=np.uint8)
    img[200:300, 200:300] = 255

    p = tmp_path / "a.png"
    _save_img(p, img)

    out = preprocess(str(p))
    assert out.shape == (300, 300)
    assert out.dtype == np.uint8


def test_similarity_identical_is_high(tmp_path: Path):
    img = np.zeros((400, 400), dtype=np.uint8)
    img[50:350, 190:210] = 255

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _save_img(p1, img)
    _save_img(p2, img)

    s = similarity(str(p1), str(p2))
    assert s > 0.9
