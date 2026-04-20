from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tp02_empreinte.method_orb import similarity as orb_sim
from tp02_empreinte.method_fft import similarity as fft_sim
from tp02_empreinte.method_gabor import similarity as gabor_sim


def _save(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_methods_run_without_crash(tmp_path: Path):
    a = np.zeros((320, 320), dtype=np.uint8)
    b = np.zeros((320, 320), dtype=np.uint8)

    # motifs différents pour éviter score identique trivial
    a[60:260, 150:170] = 255
    b[120:200, 60:260] = 255

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _save(p1, a)
    _save(p2, b)

    s1 = orb_sim(str(p1), str(p2))
    s2 = fft_sim(str(p1), str(p2))
    s3 = gabor_sim(str(p1), str(p2))

    for s in (s1, s2, s3):
        assert isinstance(s, float)
        assert -1.0 <= s <= 1.0
