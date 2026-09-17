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

Vec: TypeAlias = npt.NDArray[DType]
"""Вектор произвольной размерности — направление или смещение."""

Vec2D: TypeAlias = npt.NDArray[DType]
"""Вектор из 2 чисел (dx, dy)."""

Vec3D: TypeAlias = npt.NDArray[DType]
"""Вектор из 3 чисел (dx, dy, dz)."""

Point: TypeAlias = npt.NDArray[DType]
"""Координата произвольной размерности."""

Point2D: TypeAlias = npt.NDArray[DType]
"""Координата на плоскости (x, y)."""
