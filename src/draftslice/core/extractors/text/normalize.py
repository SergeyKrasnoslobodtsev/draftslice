from dataclasses import dataclass

import numpy as np

from draftslice.core.common.types import Floating, Mat
from draftslice.utils.image_utils import bgr_to_rgb, draw_border


@dataclass(frozen=True, eq=False)
class DBNetImage:
    """Изображение нормализованное под сеть детекции текста

    Attributes
    ----------
    image : Mat
        Кадр после паддинга.
    added_rows : int
        Сколько строк дописано снизу.
    added_cols : int
        Сколько столбцов дописано справа.
    """

    image: Mat
    added_rows: int
    added_cols: int

    @classmethod
    def create(cls, image: Mat, stride: int = 32):
        height, width = image.shape[:2]
        added_rows = _add(height, stride)
        added_cols = _add(width, stride)

        if added_rows > 0 or added_cols > 0:
            padded = draw_border(image, added_rows, added_cols)
        else:
            padded = image.copy()

        norm_img = _image_normalize(padded)

        return cls(
            image=norm_img,
            added_rows=added_rows,
            added_cols=added_cols,
        )

    def unpad(self, predictions: Floating) -> Floating:
        """Обрезает паддинг, возвращая карту предсказаний исходного размера."""
        if predictions.shape[1] != 1:
            raise ValueError(f"Ожидается 1 канал, получено {predictions.shape[1]}")

        height, width = predictions.shape[-2:]
        return predictions[0, 0, : height - self.added_rows, : width - self.added_cols]


def _add(len: int, stride: int = 32):
    return (-len) % stride


_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _image_normalize(image: Mat):
    """Подготовить вход сети: нормализация ImageNet и порядок осей NCHW.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.

    Returns
    -------
    npt.NDArray[np.float32]
        Массив формы (1, 3, высота, ширина).
    """
    rgb = bgr_to_rgb(image).astype(np.float32)
    rgb /= 255.0
    normalized = (rgb - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)[np.newaxis, ...]
