"""Тесты загрузки, масштабирования и бинаризации."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from draftslice.image_preparation import (
    binarize_strokes,
    load_bgr_image,
    resize_to_long_side,
    scale_image,
    scale_mask,
)


def build_drawing(height: int = 120, width: int = 200) -> np.ndarray:
    """Белый лист с одной темной линией."""
    image = np.full((height, width, 3), 245, np.uint8)
    cv2.line(image, (20, 60), (180, 60), (30, 30, 30), 3)
    return image


def test_load_bgr_image_reads_cyrillic_path(tmp_path: Path) -> None:
    image_path = tmp_path / "чертеж детали.png"
    cv2.imencode(".png", build_drawing())[1].tofile(str(image_path))

    image = load_bgr_image(image_path)

    assert image.shape == (120, 200, 3)


def test_load_bgr_image_reports_unreadable_file(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.png"
    broken_path.write_bytes(b"not an image")

    with pytest.raises(ValueError):
        load_bgr_image(broken_path)


def test_resize_to_long_side_keeps_aspect_ratio() -> None:
    image = build_drawing(120, 200)

    enlarged = resize_to_long_side(image, 400)
    reduced = resize_to_long_side(image, 100)

    assert enlarged.shape[:2] == (240, 400)
    assert reduced.shape[:2] == (60, 100)


def test_resize_to_long_side_returns_same_image_when_size_matches() -> None:
    image = build_drawing(120, 200)

    assert resize_to_long_side(image, 200) is image


def test_scale_mask_grows_by_one_pixel_along_border() -> None:
    mask = np.zeros((60, 100), dtype=bool)
    mask[20:40, 30:70] = True
    ideal_area = 4 * int(mask.sum())
    scaled_perimeter = 2 * (2 * 40 + 2 * 20)

    scaled = scale_mask(mask, 2.0)

    assert scaled.shape == scale_image(np.zeros((60, 100), np.uint8), 2.0).shape
    assert ideal_area <= int(scaled.sum()) <= ideal_area + scaled_perimeter + 20


def test_binarize_strokes_marks_dark_ink_only() -> None:
    image = build_drawing()

    strokes = binarize_strokes(image)

    assert strokes.strokes_mask.dtype == np.bool_
    assert 30.0 <= strokes.threshold_value < 245.0
    assert bool(strokes.strokes_mask[60, 100])
    assert not bool(strokes.strokes_mask[10, 10])
    assert float(strokes.strokes_mask.mean()) < 0.05
