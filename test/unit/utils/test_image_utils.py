import numpy as np
import pytest

from draftslice.core.exceptions import DraftsliceTypeError
from draftslice.core.utils.image_utils import rgb_to_grayscale


@pytest.mark.parametrize(
    "color",
    [(0, 0, 0), (0, 0, 255), (255, 165, 0), (255, 255, 0)],
    ids=["чёрный", "синий", "оранжевый", "жёлтый"],
)
def test_colored_line_becomes_dark(color):
    img = np.full((3, 3, 3), 255, np.uint8)
    img[1, 1] = color

    gray = rgb_to_grayscale(img)

    assert gray[1, 1] == min(color)
    assert gray[0, 0] == 255


def test_grayscale_input_raises():
    with pytest.raises(DraftsliceTypeError):
        rgb_to_grayscale(np.zeros((3, 3), np.uint8))
