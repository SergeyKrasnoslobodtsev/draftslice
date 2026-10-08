import numpy as np
import pytest

from draftslice.core.exceptions import DraftsliceTypeError, DraftsliceValueError
from draftslice.core.preprocess import preprocess


def _sheet(t: int, size: int = 400) -> np.ndarray:
    """Белый лист: прямоугольный контур шириной ровно t px и три тонкие линии по 1 px внутри."""
    img = np.full((size, size, 3), 255, np.uint8)
    lo, hi = 60, size - 60
    img[lo : lo + t, lo:hi] = 0
    img[hi - t : hi, lo:hi] = 0
    img[lo:hi, lo : lo + t] = 0
    img[lo:hi, hi - t : hi] = 0
    for y in (120, 180, 240):
        img[y, lo:hi] = 0
    return img


def _three_classes(size: int = 400) -> np.ndarray:
    """Три группы толщины: контур 8 px (две стороны), выноски 4 px (шесть линий, их больше), тонкие по 1 px."""
    img = np.full((size, size, 3), 255, np.uint8)
    img[60:68, 60:340] = 0
    img[332:340, 60:340] = 0
    for y in (100, 130, 160, 190, 220, 250):
        img[y : y + 4, 60:340] = 0
    for y in (280, 290, 300):
        img[y, 60:340] = 0
    return img


@pytest.mark.parametrize(
    "color",
    [(0, 0, 0), (0, 0, 255), (255, 165, 0), (255, 255, 0)],
    ids=["чёрный", "синий", "оранжевый", "жёлтый"],
)
def test_colored_line_becomes_dark(color):
    img = np.full((3, 3, 3), 255, np.uint8)
    img[1, 1] = color

    gray = preprocess(img, target=None).gray

    assert gray[1, 1] == min(color)
    assert gray[0, 0] == 255


def test_grayscale_input_raises():
    with pytest.raises(DraftsliceTypeError):
        preprocess(np.zeros((3, 3), np.uint8))


@pytest.mark.req("SCALE-01")
def test_without_target_keeps_scale():
    img = _sheet(6)

    res = preprocess(img, target=None)

    assert res.img is img
    assert res.scale == 1.0
    assert res.t_c is None
    assert res.binary[62, 200] == 255
    assert res.binary[10, 10] == 0


@pytest.mark.parametrize("t", [4, 6, 10, 20])
def test_thickness_ignores_thin_lines(t):
    assert preprocess(_sheet(t)).t_c == pytest.approx(t, abs=1.0)


def test_thickness_takes_top_hump_not_medium_lines():
    assert preprocess(_three_classes()).t_c == pytest.approx(8, abs=1.0)


def test_no_ink_raises():
    with pytest.raises(DraftsliceValueError):
        preprocess(np.full((50, 50, 3), 255, np.uint8))


@pytest.mark.req("SCALE-01")
def test_within_tolerance_returns_input():
    img = _sheet(6)

    res = preprocess(img)

    assert res.scale == 1.0
    assert res.img is img


@pytest.mark.req("SCALE-01")
def test_upscales_thin_contour():
    img = _sheet(4)

    res = preprocess(img)

    assert res.scale == pytest.approx(1.5, abs=0.2)
    assert res.img.shape[0] == pytest.approx(img.shape[0] * res.scale, abs=1)
    assert preprocess(res.img, target=6.0).t_c == pytest.approx(6.0, abs=1.5)


@pytest.mark.req("SCALE-01")
def test_downscales_thick_contour():
    res = preprocess(_sheet(20))

    assert res.scale < 1.0
    assert res.t_c == pytest.approx(20, abs=3)


@pytest.mark.parametrize("t", [4, 20], ids=["увеличен", "уменьшен"])
def test_gray_and_binary_match_image_size(t):
    res = preprocess(_sheet(t))

    assert res.gray.shape == res.img.shape[:2]
    assert res.binary.shape == res.img.shape[:2]


def test_caps_upscale():
    assert preprocess(_sheet(1), max_scale=3.0).scale <= 3.0


@pytest.mark.parametrize(
    "kwargs",
    [{"target": 0}, {"tol": -0.1}, {"min_t": 0}, {"max_scale": 0.5}, {"top_share": 0}, {"top_share": 1}],
)
def test_bad_params_raise(kwargs):
    with pytest.raises(DraftsliceValueError):
        preprocess(_sheet(6), **kwargs)
