"""Тесты перерисовки сохраненных линий."""

import cv2
import numpy as np

from draftslice.common_types import Mask
from draftslice.stroke_graph import build_stroke_graph
from draftslice.stroke_painting import paint_selected_paths


def build_graph(canvas: np.ndarray):
    """Граф по нарисованному кадру с пустой текстовой областью."""
    strokes_mask: Mask = canvas.astype(bool)
    return strokes_mask, build_stroke_graph(strokes_mask, np.zeros_like(strokes_mask))


def select_horizontal_paths(graph) -> Mask:
    """Ребра, вытянутые вдоль оси x."""
    kept = np.zeros(graph.path_count, dtype=bool)
    for path_index in range(graph.path_count):
        points = graph.path_points(path_index)
        kept[path_index] = np.ptp(points[:, 0]) > np.ptp(points[:, 1])
    return kept


def test_paint_keeps_selected_line_and_drops_other() -> None:
    canvas = np.zeros((160, 400), np.uint8)
    cv2.line(canvas, (30, 50), (370, 50), 1, 5)
    cv2.line(canvas, (30, 120), (370, 120), 1, 5)
    strokes_mask, graph = build_graph(canvas)

    kept = np.zeros(graph.path_count, dtype=bool)
    kept[0] = True
    painted = paint_selected_paths(graph, kept, strokes_mask)

    assert painted.sum() > 0
    assert painted[:85].any()
    assert not painted[100:].any()


def test_paint_does_not_bite_out_crossing() -> None:
    canvas = np.zeros((200, 400), np.uint8)
    cv2.line(canvas, (30, 100), (370, 100), 1, 7)
    horizontal_only = canvas.astype(bool)
    cv2.line(canvas, (200, 20), (200, 180), 1, 3)
    strokes_mask, graph = build_graph(canvas)

    painted = paint_selected_paths(graph, select_horizontal_paths(graph), strokes_mask)

    assert bool((horizontal_only & ~painted).sum() == 0)
    assert not painted[:40].any()


def test_paint_returns_empty_mask_when_nothing_selected() -> None:
    canvas = np.zeros((120, 200), np.uint8)
    cv2.line(canvas, (20, 60), (180, 60), 1, 3)
    strokes_mask, graph = build_graph(canvas)

    painted = paint_selected_paths(graph, np.zeros(graph.path_count, dtype=bool), strokes_mask)

    assert int(painted.sum()) == 0
