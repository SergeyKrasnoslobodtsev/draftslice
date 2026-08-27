"""Тесты признаков компонент на синтетических фигурах с известным смыслом."""

import cv2
import numpy as np
import pytest

from draftslice.common_types import Mask
from draftslice.features import (
    build_component_features,
    expand_selection_to_paths,
    find_enclosed_components,
    find_host_contours,
    find_repeated_components,
    find_text_sized_components,
    group_similar_components,
    measure_text_component_scale,
    select_components,
    trim_short_twigs,
)
from draftslice.stroke_graph import build_stroke_graph


def draw_ring(canvas: np.ndarray, center_x: int, center_y: int) -> None:
    """Замкнутая фигура: у нее есть внутренняя площадь."""
    cv2.circle(canvas, (center_x, center_y), 40, 1, 3)


def draw_wedge(canvas: np.ndarray, tip_x: int, tip_y: int) -> None:
    """Наконечник стрелки: толщина растет от острия к основанию."""
    points = np.array([[tip_x, tip_y], [tip_x + 80, tip_y - 14], [tip_x + 80, tip_y + 14]])
    cv2.fillPoly(canvas, [points], 1)


def draw_zigzag(canvas: np.ndarray, left_x: int, top_y: int) -> None:
    """Профиль резьбы: ровная толщина при высокой извилистости."""
    teeth = np.array([[left_x + 20 * index, top_y if index % 2 else top_y + 40] for index in range(9)])
    cv2.polylines(canvas, [teeth], False, 1, 3)


def build_features(canvas: np.ndarray, text_canvas: np.ndarray | None = None):
    """Признаки компонент по нарисованному кадру."""
    strokes_mask: Mask = canvas.astype(bool)
    text_mask: Mask = np.zeros_like(strokes_mask) if text_canvas is None else text_canvas.astype(bool)
    graph = build_stroke_graph(strokes_mask, text_mask)
    alive_paths = np.ones(graph.path_count, dtype=bool)
    return graph, build_component_features(graph, alive_paths, text_mask)


def test_ring_has_inside_area_and_survives() -> None:
    canvas = np.zeros((160, 160), np.uint8)
    draw_ring(canvas, 80, 80)

    _, features = build_features(canvas)

    assert len(features) == 1
    assert int(features.inside_area[0]) > 3000
    assert bool(select_components(features)[0])


def test_wedge_is_recognised_by_thickness_spread() -> None:
    canvas = np.zeros((160, 260), np.uint8)
    draw_wedge(canvas, 40, 80)

    _, features = build_features(canvas)

    assert float(features.thickness_spread[0]) > 2.5
    assert int(features.inside_area[0]) == 0
    assert not bool(select_components(features)[0])


def test_zigzag_survives_by_tortuosity() -> None:
    canvas = np.zeros((240, 260), np.uint8)
    draw_zigzag(canvas, 40, 120)

    _, features = build_features(canvas)

    assert float(features.tortuosity[0]) > 1.6
    assert float(features.thickness_spread[0]) < 2.5
    assert int(features.inside_area[0]) == 0
    assert bool(select_components(features)[0])


def test_straight_line_is_dropped() -> None:
    canvas = np.zeros((120, 260), np.uint8)
    cv2.line(canvas, (30, 60), (220, 60), 1, 3)

    _, features = build_features(canvas)

    assert float(features.tortuosity[0]) < 1.2
    assert not bool(select_components(features)[0])


def test_text_inside_kills_closed_component() -> None:
    canvas = np.zeros((160, 160), np.uint8)
    draw_ring(canvas, 80, 80)
    text_canvas = np.zeros_like(canvas)
    cv2.rectangle(text_canvas, (60, 60), (100, 100), 1, cv2.FILLED)

    _, features = build_features(canvas, text_canvas)

    assert float(features.inside_text_share[0]) > 0.10
    assert not bool(select_components(features)[0])


def test_repeated_wedges_are_marked_as_template() -> None:
    canvas = np.zeros((320, 320), np.uint8)
    draw_ring(canvas, 60, 60)
    draw_wedge(canvas, 30, 160)
    draw_wedge(canvas, 30, 220)
    draw_wedge(canvas, 30, 280)

    _, features = build_features(canvas)
    groups = group_similar_components(features)
    repeated = find_repeated_components(features, groups)

    assert int(repeated.sum()) == 3
    assert not bool(repeated[features.inside_area > 0][0])


def test_trim_short_twigs_removes_spur_and_keeps_line() -> None:
    canvas = np.zeros((160, 400), np.uint8)
    cv2.line(canvas, (30, 80), (370, 80), 1, 3)
    cv2.line(canvas, (200, 80), (200, 62), 1, 3)

    graph, features = build_features(canvas)
    alive_paths = np.ones(graph.path_count, dtype=bool)
    kept = trim_short_twigs(graph, alive_paths, features.path_component)

    assert graph.path_count == 3
    assert int(kept.sum()) == 2
    assert float(graph.length[kept].sum()) > 300.0


def test_expand_selection_to_paths_maps_components_back() -> None:
    canvas = np.zeros((200, 400), np.uint8)
    draw_ring(canvas, 70, 90)
    cv2.line(canvas, (200, 90), (370, 90), 1, 3)

    graph, features = build_features(canvas)
    kept_paths = expand_selection_to_paths(features, select_components(features))

    assert len(features) == 2
    assert int(kept_paths.sum()) == 1
    assert graph.branch_type[kept_paths][0] == 3


def test_text_component_scale_uses_quantile() -> None:
    canvas = np.zeros((200, 400), np.uint8)
    cv2.rectangle(canvas, (20, 20), (50, 40), 1, cv2.FILLED)
    cv2.rectangle(canvas, (100, 20), (140, 60), 1, cv2.FILLED)
    cv2.rectangle(canvas, (200, 20), (320, 120), 1, cv2.FILLED)
    text_mask: Mask = canvas.astype(bool)

    median_scale = measure_text_component_scale(text_mask, 0.5)
    maximum_scale = measure_text_component_scale(text_mask, 1.0)

    assert median_scale == pytest.approx(np.hypot(41, 41), abs=1.0)
    assert maximum_scale == pytest.approx(np.hypot(121, 101), abs=1.0)
    assert measure_text_component_scale(np.zeros((10, 10), dtype=bool), 0.5) == 0.0


def test_text_sized_components_drop_small_leftovers() -> None:
    canvas = np.zeros((320, 420), np.uint8)
    draw_ring(canvas, 90, 120)
    cv2.line(canvas, (250, 60), (280, 60), 1, 3)
    cv2.line(canvas, (250, 200), (300, 240), 1, 3)

    _, features = build_features(canvas)
    text_sized = find_text_sized_components(features, reference_diagonal=70.0)

    assert len(features) == 3
    assert int(text_sized.sum()) == 2
    assert not bool(text_sized[np.argmax(features.axis_length)])
    assert int(find_text_sized_components(features, reference_diagonal=0.0).sum()) == 0


def test_enclosed_components_are_protected_from_text_size_rule() -> None:
    canvas = np.zeros((400, 600), np.uint8)
    cv2.rectangle(canvas, (40, 40), (300, 340), 1, 3)
    cv2.circle(canvas, (170, 190), 18, 1, 3)
    cv2.circle(canvas, (480, 190), 18, 1, 3)

    graph, features = build_features(canvas)
    hosts = features.inside_area >= 1000
    enclosed = find_enclosed_components(graph, features, hosts)
    text_sized = find_text_sized_components(features, reference_diagonal=70.0)

    assert int(hosts.sum()) == 1
    assert int(enclosed.sum()) == 1
    assert int(text_sized.sum()) == 2
    assert int((text_sized & ~enclosed).sum()) == 1


def test_host_contours_take_body_and_skip_small_frames() -> None:
    canvas = np.zeros((400, 700), np.uint8)
    cv2.rectangle(canvas, (40, 40), (340, 340), 1, 3)
    cv2.rectangle(canvas, (420, 60), (520, 120), 1, 3)
    cv2.rectangle(canvas, (420, 200), (470, 240), 1, 3)

    _, features = build_features(canvas)
    hosts = find_host_contours(features, minimum_area_share=0.2)

    assert len(features) == 3
    assert int(hosts.sum()) == 1
    assert float(features.inside_area[hosts][0]) == float(features.inside_area.max())
