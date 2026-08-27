"""Тесты подготовки входа детектора и постобработки карты вероятностей."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from draftslice.text_detection import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TextProbabilityDetector,
    accept_candidates,
    build_detector_blob,
    build_text_region_mask,
    compute_unclip_distance,
    dilate_contour_mask,
    find_text_candidates,
    measure_box_score,
    pad_to_stride,
    resize_probability_map,
    take_accepted_kernels,
)


def build_probability_map(shape: tuple[int, int] = (120, 200)) -> np.ndarray:
    """Карта с двумя пятнами: уверенным и слабым."""
    probability_map = np.zeros(shape, dtype=np.float64)
    probability_map[20:40, 30:90] = 0.9
    probability_map[70:90, 120:180] = 0.2
    return probability_map


def test_pad_to_stride_makes_sides_divisible_and_pads_white() -> None:
    image = np.full((100, 150, 3), 10, np.uint8)

    padded = pad_to_stride(image)

    assert padded.image.shape[:2] == (128, 160)
    assert padded.added_rows == 28
    assert padded.added_cols == 10
    assert int(padded.image[120, 155, 0]) == 255


def test_pad_to_stride_keeps_image_when_already_divisible() -> None:
    image = np.zeros((64, 96, 3), np.uint8)

    padded = pad_to_stride(image)

    assert padded.image is image
    assert (padded.added_rows, padded.added_cols) == (0, 0)


def test_build_detector_blob_normalises_white_pixel() -> None:
    image = np.full((32, 32, 3), 255, np.uint8)

    blob = build_detector_blob(image)
    expected_red = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]

    assert blob.shape == (1, 3, 32, 32)
    assert blob.dtype == np.float32
    assert float(blob[0, 0, 0, 0]) == pytest.approx(expected_red, rel=1e-5)


def test_resize_probability_map_changes_size_only() -> None:
    probability_map = build_probability_map()

    enlarged = resize_probability_map(probability_map, (240, 400))

    assert enlarged.shape == (240, 400)
    assert 0.0 <= float(enlarged.min()) <= float(enlarged.max()) <= 1.0


def test_measure_box_score_averages_inside_box() -> None:
    probability_map = build_probability_map()
    box = np.array([[30.0, 20.0], [89.0, 20.0], [89.0, 39.0], [30.0, 39.0]])

    assert measure_box_score(probability_map, box) == pytest.approx(0.9, abs=0.05)


def test_candidates_accept_strong_spot_and_reject_weak_one() -> None:
    probability_map = build_probability_map()

    candidates = find_text_candidates(probability_map, binary_threshold=0.1)
    accepted = accept_candidates(candidates, box_threshold=0.5, minimum_short_side=3.0)
    kernels = take_accepted_kernels(candidates, accepted)

    assert len(candidates) == 2
    assert int(accepted.sum()) == 1
    assert len(kernels) == 1
    assert float(kernels.area[0]) > 1000.0


def test_candidates_reject_thin_spot_by_short_side() -> None:
    probability_map = np.zeros((120, 200), dtype=np.float64)
    probability_map[50:52, 30:150] = 0.9

    candidates = find_text_candidates(probability_map, binary_threshold=0.1)

    assert int(accept_candidates(candidates, box_threshold=0.5, minimum_short_side=3.0).sum()) == 0
    assert int(accept_candidates(candidates, box_threshold=0.5, minimum_short_side=1.0).sum()) == 1


def test_compute_unclip_distance_follows_area_over_perimeter() -> None:
    assert compute_unclip_distance(area=400.0, perimeter=80.0, unclip_ratio=1.5) == 8
    assert compute_unclip_distance(area=400.0, perimeter=0.0, unclip_ratio=1.5) == 0


def test_dilate_contour_mask_grows_square_by_distance() -> None:
    canvas = np.zeros((120, 120), np.uint8)
    cv2.rectangle(canvas, (40, 40), (59, 59), 1, cv2.FILLED)
    contour = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][0]

    region = dilate_contour_mask(contour, distance=5, shape=canvas.shape)
    corner_loss = (4 - np.pi) * 5**2

    assert int(region.sum()) == pytest.approx(30 * 30 - corner_loss, abs=40)


def test_build_text_region_mask_unites_kernels() -> None:
    canvas = np.zeros((160, 240), np.uint8)
    cv2.rectangle(canvas, (30, 30), (69, 49), 1, cv2.FILLED)
    cv2.rectangle(canvas, (150, 100), (189, 119), 1, cv2.FILLED)
    contours = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    candidates = find_text_candidates(canvas.astype(np.float64), binary_threshold=0.5)

    region = build_text_region_mask(
        take_accepted_kernels(candidates, np.ones(len(candidates), dtype=bool)), unclip_ratio=1.5, shape=canvas.shape
    )

    assert len(contours) == 2
    assert int(region.sum()) > int(canvas.sum())
    assert bool(region[40, 50]) and bool(region[110, 170])


def test_detector_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TextProbabilityDetector(tmp_path / "no_such_model.onnx")
