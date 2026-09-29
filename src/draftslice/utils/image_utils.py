import cv2
import numpy as np

from draftslice.core.common.types import Floating, Mat


def bgr_to_rgb(bgr: Mat) -> Mat:
    return cv2.cvtColor(bgr, code=cv2.COLOR_BGR2RGB)

def draw_border(img: Mat, rows: int, cols: int) -> Mat:
    return cv2.copyMakeBorder(img,
                              0,
                              rows,
                              0,
                              cols,
                              cv2.BORDER_CONSTANT,
                              value=(255, 255, 255),
                              )

def resize_to_long_side(image: Mat, long_side: int) -> Mat:
    """Привести длинную сторону кадра к заданному размеру.

    Parameters
    ----------
    image : Mat
        Исходный кадр.
    long_side : int
        Требуемый размер длинной стороны в пикселях.

    Returns
    -------
    Mat
        Кадр с сохраненными пропорциями.
    """
    current_side = max(image.shape[:2])
    if current_side == long_side:
        return image

    factor = long_side / float(current_side)
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    width = round(image.shape[1] * factor)
    height = round(image.shape[0] * factor)
    return cv2.resize(image, (width, height), interpolation=interpolation)

def resize_prob_map(predict: Floating, shape: tuple[int, int]):
    """Привести карту вероятностей к заданному размеру.
    """
    interpolation = cv2.INTER_AREA if predict.shape[0] > shape[0] else cv2.INTER_LINEAR
    resized = cv2.resize(predict, (shape[1], shape[0]), interpolation=interpolation)
    return resized.astype(np.float32)
