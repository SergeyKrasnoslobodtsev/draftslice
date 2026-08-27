"""Тесты метрик этапов."""

import cv2
import numpy as np
import pytest
from stage_metrics import (
    measure_chain_metrics,
    measure_graph_metrics,
    measure_mask_metrics,
    measure_thickness_class_metrics,
)

from draftslice.chain_merging import ChainMergeParameters, build_stroke_chains
from draftslice.stroke_graph import build_stroke_graph
from draftslice.thickness_classes import fit_thickness_classes


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


def test_chain_metrics_report_merge_result() -> None:
    canvas = np.zeros((160, 400), np.uint8)
    cv2.line(canvas, (20, 80), (380, 80), 1, 3)
    cv2.line(canvas, (200, 80), (200, 150), 1, 3)
    mask = canvas.astype(bool)

    graph = build_stroke_graph(mask, np.zeros_like(mask))
    chains = build_stroke_chains(graph, ChainMergeParameters())
    metrics = measure_chain_metrics(chains)

    assert metrics.chain_count == 2
    assert metrics.linked_path_count == 3
    assert metrics.links_per_chain == pytest.approx(1.5)
    assert metrics.maximum_length > 350.0
    assert metrics.text_chain_count == 0


def test_thickness_class_metrics_split_length_between_classes() -> None:
    canvas = np.zeros((320, 400), np.uint8)
    for row in (40, 80, 120):
        cv2.line(canvas, (20, row), (380, row), 1, 3)
    for row in (200, 240, 280):
        cv2.line(canvas, (20, row), (380, row), 1, 11)
    mask = canvas.astype(bool)

    graph = build_stroke_graph(mask, np.zeros_like(mask))
    chains = build_stroke_chains(graph, ChainMergeParameters())
    classes = fit_thickness_classes(chains, class_count=2)
    metrics = measure_thickness_class_metrics(chains, classes)

    assert metrics.chain_counts.tolist() == [3, 3]
    assert metrics.length_shares.sum() == pytest.approx(1.0)
    assert metrics.unclassified_count == 0
