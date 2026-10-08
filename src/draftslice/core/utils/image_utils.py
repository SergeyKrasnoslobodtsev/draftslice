import enum

import cv2
import numpy as np

from draftslice.core.exceptions import InvalidImageAngleError
from draftslice.core.types import Mat
from draftslice.core.validators import validate_image


class MorphShape(enum.IntEnum):
    """Форма структурного элемента для морфологических операций."""

    RECT = cv2.MORPH_RECT
    ELLIPSE = cv2.MORPH_ELLIPSE
    CROSS = cv2.MORPH_CROSS


@validate_image(channels=1)
def dilate_image(binary_image: Mat, shape: MorphShape, kernel_size: int, iterations: int = 1) -> Mat:
    """Дилатирует бинарное изображение ядром заданной формы и размера."""
    kernel = cv2.getStructuringElement(shape.value, (kernel_size, kernel_size))
    return cv2.dilate(binary_image, kernel, iterations=iterations)


def rotate_image(image: Mat, angle: float) -> tuple[Mat, np.ndarray]:
    """Поворачивает изображение на заданный угол в градусах."""
    if angle < 0 or angle >= 360:
        raise InvalidImageAngleError("Параметр `angle` должен быть в диапазоне [0, 360).")

    if angle < 1e-7:
        return image, np.eye(2, 3, dtype=np.float32)

    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2
    dst_size = (new_w, new_h)

    rotated = cv2.warpAffine(image, mat, dst_size, flags=cv2.INTER_CUBIC)
    return rotated, mat
