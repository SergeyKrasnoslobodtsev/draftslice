"""Тесты склейки ребер скелета в цепи."""

import cv2
import numpy as np

from draftslice.chain_merging import (
    ChainMergeParameters,
    MergeGates,
    build_stroke_chains,
    measure_end_tangents,
    weighted_median,
)
from draftslice.common_types import Mask
from draftslice.stroke_graph import build_stroke_graph


def build_graph(canvas: np.ndarray):
    """Граф по нарисованному кадру с пустой текстовой областью."""
    strokes_mask: Mask = canvas.astype(bool)
    return build_stroke_graph(strokes_mask, np.zeros_like(strokes_mask))


def test_weighted_median_follows_weights() -> None:
    values = np.array([1.0, 10.0])

    assert weighted_median(values, np.array([1.0, 100.0])) == 10.0
    assert weighted_median(values, np.array([100.0, 1.0])) == 1.0
    assert np.isnan(weighted_median(np.array([np.nan]), np.array([1.0])))


def test_end_tangents_point_into_the_path() -> None:
    canvas = np.zeros((120, 300), np.uint8)
    cv2.line(canvas, (20, 60), (280, 60), 1, 3)

    graph = build_graph(canvas)
    tangents = measure_end_tangents(graph, tangent_length_pixels=12)

    assert tangents.shape == (1, 2, 2)
    assert abs(float(tangents[0, 0, 1])) > 0.9
    assert float(tangents[0, 0, 1]) * float(tangents[0, 1, 1]) < 0


def test_collinear_paths_merge_into_one_chain() -> None:
    canvas = np.zeros((160, 400), np.uint8)
    cv2.line(canvas, (20, 80), (380, 80), 1, 3)
    cv2.line(canvas, (200, 80), (200, 150), 1, 3)

    graph = build_graph(canvas)
    chains = build_stroke_chains(graph, ChainMergeParameters())

    assert graph.path_count == 3
    assert chains.chain_count == 2
    assert float(chains.length.max()) > 350.0


def test_right_angle_is_not_merged() -> None:
    canvas = np.zeros((200, 200), np.uint8)
    cv2.line(canvas, (20, 100), (100, 100), 1, 3)
    cv2.line(canvas, (100, 20), (100, 180), 1, 3)

    graph = build_graph(canvas)
    chains = build_stroke_chains(graph, ChainMergeParameters())

    assert graph.path_count == 3
    assert chains.chain_count == 2
    assert sorted(np.bincount(chains.chain_label).tolist()) == [1, 2]


def test_thickness_gate_blocks_merge_of_different_widths() -> None:
    canvas = np.zeros((200, 400), np.uint8)
    cv2.line(canvas, (20, 100), (200, 100), 1, 3)
    cv2.line(canvas, (200, 100), (380, 100), 1, 11)
    cv2.line(canvas, (200, 100), (200, 180), 1, 3)

    graph = build_graph(canvas)
    strict_chains = build_stroke_chains(graph, ChainMergeParameters(gates=MergeGates(maximum_thickness_ratio=1.6)))
    loose_chains = build_stroke_chains(graph, ChainMergeParameters(gates=MergeGates(maximum_thickness_ratio=5.0)))

    assert strict_chains.chain_count > loose_chains.chain_count


def test_expand_selection_to_paths_maps_chains_back() -> None:
    canvas = np.zeros((160, 400), np.uint8)
    cv2.line(canvas, (20, 80), (380, 80), 1, 3)
    cv2.line(canvas, (200, 80), (200, 150), 1, 3)

    graph = build_graph(canvas)
    chains = build_stroke_chains(graph, ChainMergeParameters())
    longest_chain = np.zeros(chains.chain_count, dtype=bool)
    longest_chain[int(np.argmax(chains.length))] = True

    kept_paths = chains.expand_selection_to_paths(longest_chain)

    assert len(kept_paths) == graph.path_count
    assert int(kept_paths.sum()) == 2
