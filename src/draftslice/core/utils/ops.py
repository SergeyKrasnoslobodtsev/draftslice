import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Поворачивает изображение на заданный угол в градусах."""
    if angle < 0 or angle >= 360:
        raise ValueError("Параметр `angle` должен быть в диапазоне [0, 360).")

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


def scale_mask(
    mask: np.ndarray, shape: tuple[int, int], scale: float, box: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """Маска рабочего масштаба в исходном: пиксель (Y, X) берёт пиксель (floor(Y * scale), floor(X * scale)).

    Целая часть, а не округление: иначе нижний ряд выходит за маску.

    Parameters
    ----------
    mask : np.ndarray
        Маска `(h, w)` рабочего масштаба.
    shape : tuple[int, int]
        `(H, W)` исходного листа.
    scale : float
        Во сколько раз рабочий масштаб больше исходного.
    box : tuple[int, int, int, int] | None
        `x0 y0 x1 y1` рамки в исходных координатах; `None` - весь лист.

    Returns
    -------
    np.ndarray
        Маска размера рамки (или `shape`) того же dtype.
    """
    x0, y0, x1, y1 = box if box is not None else (0, 0, shape[1], shape[0])
    ys = np.minimum((np.arange(y0, y1) * scale).astype(np.int64), mask.shape[0] - 1)
    xs = np.minimum((np.arange(x0, x1) * scale).astype(np.int64), mask.shape[1] - 1)
    return mask[ys][:, xs]
