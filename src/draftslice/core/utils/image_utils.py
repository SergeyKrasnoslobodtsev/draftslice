import enum

import cv2

from draftslice.core.types import Mat
from draftslice.core.validators import validate_image


class MorphShape(enum.IntEnum):
    """Форма структурного элемента для морфологических операций."""

    RECT = cv2.MORPH_RECT
    ELLIPSE = cv2.MORPH_ELLIPSE
    CROSS = cv2.MORPH_CROSS


@validate_image(channels=3)
def rgb_to_grayscale(image_rgb: Mat) -> Mat:
    """Преобразует RGB изображение в градации серого."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


@validate_image(channels=1)
def binarize_image(grayscale_image: Mat) -> Mat:
    """Бинаризует изображение порогом Otsu с инверсией."""
    return cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


@validate_image(channels=1)
def dilate_image(binary_image: Mat, shape: MorphShape, kernel_size: int, iterations: int = 1) -> Mat:
    """Дилатирует бинарное изображение ядром заданной формы и размера."""
    kernel = cv2.getStructuringElement(shape.value, (kernel_size, kernel_size))
    return cv2.dilate(binary_image, kernel, iterations=iterations)
