"""Проверки входа, как `ultralytics.utils.checks`: явный вызов в начале функции, ошибка - встроенное исключение."""

import numpy as np


def check_image(img: np.ndarray, channels: int | None = None, dtype: type | None = None) -> np.ndarray:
    """Проверяет число каналов и dtype изображения.

    Parameters
    ----------
    img : np.ndarray
        Изображение `(H, W)` или `(H, W, C)`.
    channels : int | None
        Нужное число каналов; `None` - не проверять. У `(H, W)` один канал.
    dtype : type | None
        Нужный dtype; `None` - не проверять.

    Returns
    -------
    np.ndarray
        То же изображение.

    Raises
    ------
    TypeError
        `img` не массив, другое число каналов или другой dtype.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Изображение должно быть np.ndarray, получено: {type(img).__name__}")
    n = img.shape[2] if img.ndim == 3 else 1
    if channels is not None and n != channels:
        raise TypeError(f"Изображение должно иметь {channels} канал(ов), получено: {n}")
    if dtype is not None and img.dtype != dtype:
        raise TypeError(f"Изображение должно иметь dtype={np.dtype(dtype)}, получено: {img.dtype}")
    return img
