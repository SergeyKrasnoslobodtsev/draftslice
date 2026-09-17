from typing import TypeAlias, TypeVar

import cv2.typing as cvt
import numpy as np
import numpy.typing as npt

Mat: TypeAlias = cvt.MatLike
"""Тип для представления матрицы/изображения в OpenCV."""

DType = TypeVar("DType", bound=np.generic)

Array: TypeAlias = npt.NDArray[DType]
"""Массив произвольной формы и размерности."""

Array2D: TypeAlias = npt.NDArray[DType]
"""Двумерный массив (матрица/таблица, напр. stats, маска)."""
