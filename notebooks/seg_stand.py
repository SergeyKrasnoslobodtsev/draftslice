"""Стенд сегментации на виды: прогон по папке изображений с сохранением промежуточных шагов.

    python notebooks/seg_stand.py data/images --out research/results/seg_stand

Для каждого изображения - своя папка с PNG по шагам `seg.segment` и кропами `run()`, в корне -
`summary.json`: счётчики по шагам, параметры, ID требований и версия канона (канон, раздел 12).
Параметры - умолчания `run()`, чтобы стенд всегда прогонял то, что лежит в проде.
"""

import argparse
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
from draw import colorize

from draftslice.core.io import load_image
from draftslice.core.preprocess import preprocess
from draftslice.core.region_seg import seg
from draftslice.core.region_seg.pipeline import run

REQS = ["DEL-07", "KEEP-01", "VIEW-01", "VIEW-02", "OUT-04"]
CANON = "1.0"
PARAMS = {n: p.default for n, p in inspect.signature(run).parameters.items() if p.default is not p.empty}
EXTS = {".png", ".jpg", ".jpeg"}

GREY = (160, 160, 160)
RED = (230, 40, 40)


def _save(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _paint(binary: np.ndarray, marked: np.ndarray) -> np.ndarray:
    """Краска серым на белом, отмеченные пикселы красным."""
    out = np.full((*binary.shape, 3), 255, np.uint8)
    out[binary > 0] = GREY
    out[marked] = RED
    return out


def stand_one(img: np.ndarray, out: Path, params: dict) -> dict:
    """Шаги сегментации одного листа в PNG, возвращает счётчики."""
    out.mkdir(parents=True, exist_ok=True)

    binary = preprocess(img, target=None).binary
    _save(out / "01_binary.png", cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2RGB))

    edge = seg._edge_lines(binary, params["sheet_cov"])
    _save(out / "02_edge_lines.png", _paint(binary, edge))
    binary[edge] = 0

    labels, seeds = seg._find_seeds(binary, params["sheet_cov"], params["thick_factor"], params["area_factor"])
    _save(out / "03_seeds.png", _paint(binary, np.isin(labels, seeds)))

    filled = seg._fill_seeds(labels, seeds, params["min_extent"])
    _save(out / "04_filled.png", _paint(binary, filled > 0))

    crops = run(img, **params)
    seg_map = np.zeros(binary.shape, np.uint16)
    for c in crops:
        seg_map[c.y : c.y + c.h, c.x : c.x + c.w][c.mask] = c.label
    _save(out / "05_segments.png", (colorize(seg_map) * 255).astype(np.uint8))

    overlay = img.copy()
    lw = max(2, img.shape[1] // 500)
    for c in crops:
        cv2.rectangle(overlay, (c.x, c.y), (c.x + c.w, c.y + c.h), RED, lw)
        cv2.putText(overlay, str(c.label), (c.x, max(c.y - lw, 0)), cv2.FONT_HERSHEY_SIMPLEX, lw / 2, RED, lw)
        _save(out / f"crop_{c.label}.png", c.img)
    _save(out / "06_overlay.png", overlay)

    return {
        "edge_px": int(edge.sum()),
        "seeds": len(seeds),
        "segments": len(np.unique(seg_map)) - 1,
        "crops": len(crops),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Стенд сегментации на виды")
    ap.add_argument("src", type=Path, help="папка с изображениями")
    ap.add_argument("--out", type=Path, default=Path("research/results/seg_stand"))
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in EXTS)
    images = {}
    for path in files:
        images[path.name] = stand_one(load_image(str(path)), args.out / path.stem, PARAMS)
        print(path.name, images[path.name])

    summary = {"canon_version": CANON, "requirements": REQS, "params": PARAMS, "images": images}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
