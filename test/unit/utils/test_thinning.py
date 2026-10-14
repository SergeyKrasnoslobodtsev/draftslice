import cv2
import numpy as np
import pytest

from draftslice.core.utils.thinning import thinning


def _cv2(mask):
    return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


@pytest.mark.req("KEEP-01", "SCALE-01")
@pytest.mark.parametrize("seed", range(50))
def test_matches_opencv_on_random_blobs(seed):
    """Пятна разной толщины, в том числе на краю изображения: скелет совпадает с OpenCV пиксель в пиксель."""
    rng = np.random.default_rng(seed)
    noise = rng.random((60, 80)).astype(np.float32)
    mask = (cv2.GaussianBlur(noise, (0, 0), rng.uniform(0.5, 3)) > rng.uniform(0.45, 0.55)).astype(np.uint8) * 255
    np.testing.assert_array_equal(thinning(mask), _cv2(mask))


@pytest.mark.req("KEEP-01", "SCALE-01")
def test_matches_opencv_on_lines():
    mask = np.zeros((120, 160), np.uint8)
    cv2.line(mask, (10, 10), (150, 100), 255, 9)
    cv2.circle(mask, (60, 70), 35, 255, 6)
    cv2.rectangle(mask, (100, 15), (150, 60), 255, 4)
    np.testing.assert_array_equal(thinning(mask), _cv2(mask))


@pytest.mark.req("KEEP-01", "SCALE-01")
def test_empty_and_input_untouched():
    mask = np.zeros((10, 10), np.uint8)
    assert not thinning(mask).any()
    mask[3:7, 2:8] = 1  # не 0/255: краска - любые ненулевые
    before = mask.copy()
    out = thinning(mask)
    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(out, _cv2(mask * 255))


@pytest.mark.req("KEEP-01", "SCALE-01")
def test_rejects_non_gray_uint8():
    with pytest.raises(TypeError):
        thinning(np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(TypeError):
        thinning(np.zeros((10, 10), np.float32))
