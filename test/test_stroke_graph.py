"""Тесты графа скелета на игрушечных масках с известным ответом."""

import cv2
import numpy as np

from draftslice.common_types import Mask
from draftslice.stroke_graph import (
    CYCLE_BRANCH,
    ISOLATED_BRANCH,
    SPUR_BRANCH,
    build_stroke_graph,
    find_hanging_paths,
    label_connected_components,
    measure_thickness_spread,
    median_by_group,
)


def build_graph_from_canvas(canvas: np.ndarray, text_canvas: np.ndarray | None = None):
    """Граф по нарисованной фигуре, текстовая область пустая, если не задана."""
    strokes_mask: Mask = canvas.astype(bool)
    text_mask: Mask = np.zeros_like(strokes_mask) if text_canvas is None else text_canvas.astype(bool)
    return build_stroke_graph(strokes_mask, text_mask)


def test_median_by_group_takes_upper_median_and_marks_empty() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    labels = np.array([0, 0, 0, 1, 1])

    medians = median_by_group(values, labels, 3)

    assert medians[0] == 2.0
    assert medians[1] == 20.0
    assert np.isnan(medians[2])


def test_straight_line_gives_one_isolated_path_with_real_thickness() -> None:
    canvas = np.zeros((60, 120), np.uint8)
    cv2.line(canvas, (20, 30), (100, 30), 1, 3)
    real_width = int(canvas[:, 60].sum())

    graph = build_graph_from_canvas(canvas)

    assert graph.path_count == 1
    assert graph.branch_type.tolist() == [ISOLATED_BRANCH]
    assert graph.thickness[0] == real_width
    assert graph.length[0] > 70.0


def test_cross_gives_four_spurs_around_one_junction() -> None:
    canvas = np.zeros((80, 80), np.uint8)
    cv2.line(canvas, (10, 40), (70, 40), 1, 3)
    cv2.line(canvas, (40, 10), (40, 70), 1, 3)

    graph = build_graph_from_canvas(canvas)
    hanging = find_hanging_paths(graph, np.ones(graph.path_count, dtype=bool))

    assert graph.path_count == 4
    assert graph.branch_type.tolist() == [SPUR_BRANCH] * 4
    assert int(graph.node_degree.max()) == 4
    assert int(hanging.sum()) == 4


def test_circle_gives_one_cycle_without_hanging_paths() -> None:
    canvas = np.zeros((80, 80), np.uint8)
    cv2.circle(canvas, (40, 40), 25, 1, 3)

    graph = build_graph_from_canvas(canvas)
    hanging = find_hanging_paths(graph, np.ones(graph.path_count, dtype=bool))

    assert graph.path_count == 1
    assert graph.branch_type.tolist() == [CYCLE_BRANCH]
    assert int(hanging.sum()) == 0


def test_separate_shapes_fall_into_separate_components() -> None:
    canvas = np.zeros((60, 140), np.uint8)
    cv2.line(canvas, (10, 20), (50, 20), 1, 3)
    cv2.line(canvas, (80, 40), (130, 40), 1, 3)

    graph = build_graph_from_canvas(canvas)
    components = label_connected_components(graph, np.ones(graph.path_count, dtype=bool))

    assert graph.path_count == 2
    assert len(set(components.tolist())) == 2


def test_text_share_matches_covered_part_of_line() -> None:
    canvas = np.zeros((60, 120), np.uint8)
    cv2.line(canvas, (20, 30), (100, 30), 1, 3)
    text_canvas = np.zeros_like(canvas)
    text_canvas[:, :60] = 1

    graph = build_graph_from_canvas(canvas, text_canvas)

    assert 0.45 <= float(graph.text_share[0]) <= 0.55


def test_thickness_spread_separates_wedge_from_even_line() -> None:
    line_canvas = np.zeros((60, 120), np.uint8)
    cv2.line(line_canvas, (20, 30), (100, 30), 1, 3)
    wedge_canvas = np.zeros((60, 120), np.uint8)
    cv2.fillPoly(wedge_canvas, [np.array([[20, 30], [100, 16], [100, 44]])], 1)

    line_graph = build_graph_from_canvas(line_canvas)
    wedge_graph = build_graph_from_canvas(wedge_canvas)
    line_spread = measure_thickness_spread(line_graph, np.arange(line_graph.path_count))
    wedge_spread = measure_thickness_spread(wedge_graph, np.arange(wedge_graph.path_count))

    assert float(line_spread.max()) < 1.2
    assert float(wedge_spread.max()) > 2.0


def test_measure_thickness_spread_skips_paths_out_of_selection() -> None:
    canvas = np.zeros((80, 80), np.uint8)
    cv2.line(canvas, (10, 40), (70, 40), 1, 3)
    cv2.line(canvas, (40, 10), (40, 70), 1, 3)

    graph = build_graph_from_canvas(canvas)
    spread = measure_thickness_spread(graph, np.array([0]))

    assert np.isfinite(spread[0])
    assert bool(np.isnan(spread[1:]).all())
