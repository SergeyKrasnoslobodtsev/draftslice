"""Алгоритм детекции текста использует модель DBNet, также известная при использовании
PaddleOcr. Нормализация изображение перед подачей на вход сети была заимствована
из исходников PaddleOcr.

Можно работать с одним типом изображений, но предпочтительнее использовать
функцию `detect_multiscale`. При стандартном подходе сеть может не увидеть
маленькие буквы или слишком большие буквы типа (А - А).

Важным моментом является то, что окончательно принимать решение текст это или нет
по картам предсказаний - нельзя. Основной проблемой является сеть может уверено
спутать отверстия как текст (особенно круговые) и часто пропускает букву "Г".

"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from draftslice.core.common.types import Floating, Mat
from draftslice.core.extractors.text.normalize import DBNetImage
from draftslice.utils.image_utils import resize_prob_map, resize_to_long_side


class TextProbabilityDetector:
    """Находит текст на изображении."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        self._session = ort.InferenceSession(str(model_path))

    def _run(self, image: Mat):
        """Детектор текста

        Parameters
        ----------
        image : Mat
            Входное изображение в формате BGR (по умолчанию OpenCV).

        Returns
        -------
        Floating
            Возвращает карту предсказаний в формате float32

        """

        norm_img = DBNetImage.create(image)

        predict = np.array(
            self._session.run(None, {self._session.get_inputs()[0].name: norm_img.image})[0],
            dtype=np.float32,
        )
        unpadded = norm_img.unpad(predict)
        uncliped = ...
        return uncliped

    def detect_multiscale(self, image: Mat, long_sides: tuple[int, ...]):
        """Детектор текста на изображении с разным разрешением.

        Parameters
        ----------
        image : Mat
            Входное изображение в формате BGR (по умолчанию OpenCV).
        long_sides : tuple[int, ...]
            Размер длинной стороны изображения.

        Returns
        -------
        NDArray[Floating]
            Вернет массив карт предсказаний.

        Note
        ----
        Изменяет размер исходного изображения и вызывает метод `detect`.
        Также карты предсказаний на выходе преобразуются к оргинальному
        размеру изображения, так что вам не нужно беспокоиться об этом
        после вызова функции.
        """

        maps: list[Floating] = []

        for long_side in long_sides:
            resize_img = resize_to_long_side(image, long_side)
            predict = self._run(resize_img)
            maps.append(resize_prob_map(predict, image.shape[:2]))

        return np.maximum.reduce(maps)
