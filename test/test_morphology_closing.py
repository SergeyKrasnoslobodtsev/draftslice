"""Тесты морфологии на синтетических масках с известным ответом."""

import cv2
import numpy as np

from draftslice.common_types import Mask
from draftslice.morphology_closing import (
    build_line_kernel,
    close_along_orientations,
    fill_internal_holes,
    suggest_kernel_length,
)


def draw_mask(height: int, width: int) -> np.ndarray:
    """Пустой холст под рисование фигур средствами cv2."""
    return np.zeros((height, width), dtype=np.uint8)


def count_components(strokes_mask: Mask) -> int:
    """Число связных кусков маски без учета фона."""
    return int(cv2.connectedComponents(strokes_mask.astype(np.uint8), connectivity=8)[0]) - 1


def test_fill_internal_holes_takes_ring_interior() -> None:
    canvas = draw_mask(20, 20)
    cv2.rectangle(canvas, (4, 4), (15, 15), 1, 1)
    ring = canvas.astype(bool)

    holes = fill_internal_holes(ring) & ~ring

    assert int(holes.sum()) == 100


def test_fill_internal_holes_ignores_diagonal_across_bbox() -> None:
    canvas = draw_mask(20, 20)
    cv2.line(canvas, (0, 0), (19, 19), 1, 1)
    diagonal = canvas.astype(bool)

    holes = fill_internal_holes(diagonal) & ~diagonal

    assert int(holes.sum()) == 0


def test_fill_internal_holes_ignores_open_wedge() -> None:
    canvas = draw_mask(30, 30)
    cv2.line(canvas, (2, 15), (27, 5), 1, 1)
    cv2.line(canvas, (2, 15), (27, 25), 1, 1)
    wedge = canvas.astype(bool)

    holes = fill_internal_holes(wedge) & ~wedge

    assert int(holes.sum()) == 0


def test_build_line_kernel_is_odd_and_centered() -> None:
    kernel = build_line_kernel(8, 0.0)

    assert kernel.shape == (9, 9)
    assert int(kernel[4].sum()) == 9
    assert int(kernel.sum()) == 9


def test_close_along_orientations_joins_broken_line() -> None:
    canvas = draw_mask(60, 120)
    cv2.line(canvas, (20, 30), (54, 30), 1, 1)
    cv2.line(canvas, (62, 30), (100, 30), 1, 1)
    broken = canvas.astype(bool)
    assert count_components(broken) == 2

    closed = close_along_orientations(broken, kernel_length=11, orientation_count=8)

    assert count_components(closed) == 1


def test_close_along_orientations_keeps_parallel_lines_apart() -> None:
    canvas = draw_mask(60, 120)
    cv2.line(canvas, (20, 22), (100, 22), 1, 1)
    cv2.line(canvas, (20, 38), (100, 38), 1, 1)
    parallel = canvas.astype(bool)

    closed = close_along_orientations(parallel, kernel_length=11, orientation_count=8)

    assert count_components(closed) == 2


def test_suggest_kernel_length_follows_thickness_edge() -> None:
    assert suggest_kernel_length(3.06) == 3
    assert suggest_kernel_length(17.4) == 17
    assert suggest_kernel_length(11.6) == 13


def test_suggest_kernel_length_stays_odd_and_bounded() -> None:
    assert suggest_kernel_length(0.4) == 3
    assert suggest_kernel_length(120.0) == 31
    assert all(suggest_kernel_length(edge) % 2 == 1 for edge in (2.0, 4.0, 6.0, 8.0, 10.0))
