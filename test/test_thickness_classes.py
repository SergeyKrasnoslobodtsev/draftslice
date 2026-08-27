"""Тесты классов толщины."""

import cv2
import numpy as np
import pytest

from draftslice.chain_merging import ChainMergeParameters, build_stroke_chains
from draftslice.common_types import Mask
from draftslice.stroke_graph import build_stroke_graph
from draftslice.thickness_classes import fit_thickness_classes, select_chains_by_class


def build_chains(canvas: np.ndarray, text_canvas: np.ndarray | None = None):
    """Цепи по нарисованному кадру."""
    strokes_mask: Mask = canvas.astype(bool)
    text_mask: Mask = np.zeros_like(strokes_mask) if text_canvas is None else text_canvas.astype(bool)
    graph = build_stroke_graph(strokes_mask, text_mask)
    return build_stroke_chains(graph, ChainMergeParameters())


def draw_lines(canvas: np.ndarray, rows: tuple[int, ...], line_width: int) -> None:
    """Набор горизонтальных линий одной толщины."""
    for row in rows:
        cv2.line(canvas, (20, row), (380, row), 1, line_width)


def test_classes_split_thin_and_thick_lines() -> None:
    canvas = np.zeros((320, 400), np.uint8)
    draw_lines(canvas, (40, 80, 120), 3)
    draw_lines(canvas, (200, 240, 280), 11)

    chains = build_chains(canvas)
    classes = fit_thickness_classes(chains, class_count=2)

    assert classes.centers[0] < classes.edges[0] < classes.centers[1]
    assert classes.centers[0] == pytest.approx(5.0, abs=1.0)
    assert classes.centers[1] == pytest.approx(13.0, abs=1.0)
    assert set(classes.label.tolist()) == {0, 1}


def test_text_chains_stay_out_of_fitting() -> None:
    canvas = np.zeros((360, 400), np.uint8)
    draw_lines(canvas, (40, 80, 120), 3)
    draw_lines(canvas, (200, 240, 280), 11)
    draw_lines(canvas, (330,), 1)
    text_canvas = np.zeros_like(canvas)
    text_canvas[310:350, :] = 1

    chains = build_chains(canvas, text_canvas)
    classes = fit_thickness_classes(chains, class_count=2)

    assert int((classes.label < 0).sum()) == 1
    assert classes.centers[0] == pytest.approx(5.0, abs=1.0)
    assert classes.centers[1] == pytest.approx(13.0, abs=1.0)


def test_select_chains_by_class_keeps_thick_only() -> None:
    canvas = np.zeros((320, 400), np.uint8)
    draw_lines(canvas, (40, 80, 120), 3)
    draw_lines(canvas, (200, 240, 280), 11)

    chains = build_chains(canvas)
    classes = fit_thickness_classes(chains, class_count=2)
    kept = select_chains_by_class(chains, classes, minimum_class=1)

    assert int(kept.sum()) == 3
    assert float(chains.thickness[kept].min()) > float(classes.edges[0])


def test_fit_reports_when_everything_is_text() -> None:
    canvas = np.zeros((200, 400), np.uint8)
    draw_lines(canvas, (60, 120), 3)
    text_canvas = np.ones_like(canvas)

    chains = build_chains(canvas, text_canvas)

    with pytest.raises(ValueError):
        fit_thickness_classes(chains, class_count=2)
