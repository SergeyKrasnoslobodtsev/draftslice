"""Тесты метрик этапов."""

import cv2
import numpy as np
from stage_metrics import measure_graph_metrics, measure_mask_metrics

from draftslice.stroke_graph import build_stroke_graph


def test_mask_metrics_count_pieces_and_holes() -> None:
    canvas = np.zeros((120, 240), np.uint8)
    cv2.circle(canvas, (60, 60), 30, 1, 3)
    cv2.line(canvas, (150, 40), (220, 40), 1, 3)
    mask = canvas.astype(bool)

    metrics = measure_mask_metrics(mask)

    assert metrics.component_count == 2
    assert metrics.hole_count == 1
    assert metrics.ink_share == float(mask.mean())


def test_graph_metrics_split_branches_by_type() -> None:
    canvas = np.zeros((160, 160), np.uint8)
    cv2.line(canvas, (20, 80), (140, 80), 1, 3)
    cv2.line(canvas, (80, 20), (80, 140), 1, 3)
    mask = canvas.astype(bool)

    graph = build_stroke_graph(mask, np.zeros_like(mask))
    metrics = measure_graph_metrics(graph)

    assert metrics.path_count == 4
    assert metrics.spur_count == 4
    assert metrics.junction_count == 0
    assert metrics.free_end_count == 4
    assert metrics.branching_node_count == 1
    assert metrics.median_thickness > 0.0


def test_graph_metrics_count_text_paths() -> None:
    canvas = np.zeros((120, 240), np.uint8)
    cv2.line(canvas, (20, 60), (100, 60), 1, 3)
    cv2.line(canvas, (140, 60), (220, 60), 1, 3)
    mask = canvas.astype(bool)
    text_mask = np.zeros_like(mask)
    text_mask[:, 120:] = True

    graph = build_stroke_graph(mask, text_mask)
    metrics = measure_graph_metrics(graph)

    assert metrics.path_count == 2
    assert metrics.text_path_count == 1
