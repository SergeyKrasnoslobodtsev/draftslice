from pathlib import Path

import cv2
import numpy as np
import pytest

W, H = 200, 150


def _write_sheet(root: Path, stem: str, polys: list[list[tuple[float, float]]]) -> None:
    """Белый лист W x H и разметка: полигоны в пикселях листа."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    img = np.full((H, W, 3), 255, np.uint8)
    lines = []
    for poly in polys:
        cv2.fillPoly(img, [np.array(poly, np.int32)], (0, 0, 0))
        xy = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly)
        lines.append(f"0 {xy}")
    cv2.imwrite(str(root / "images" / f"{stem}.png"), img)
    (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def triangle(x: float, y: float, s: float = 10) -> list[tuple[float, float]]:
    return [(x, y), (x + s, y), (x + s / 2, y + s)]


@pytest.fixture
def export(tmp_path) -> Path:
    """Экспорт в стиле Label Studio: четыре листа с 4, 3, 2 и 1 объектом."""
    root = tmp_path / "export"
    (root).mkdir()
    (root / "classes.txt").write_text("arrow\n", encoding="utf-8")
    counts = {"aa-1": 4, "bb-2": 3, "cc-3": 2, "dd-4": 1}
    for stem, n in counts.items():
        _write_sheet(root, stem, [triangle(10 + 40 * i, 20 + 5 * i) for i in range(n)])
    return root
