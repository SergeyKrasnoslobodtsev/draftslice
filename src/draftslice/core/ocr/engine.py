from dataclasses import dataclass

import cv2
import numpy as np
from paddleocr import PaddleOCR

from draftslice.core.types import Array2D, Mat
from draftslice.core.utils.image_utils import rotate_image


@dataclass
class PaddleOCRParameters:
    """Параметры для настройки PaddleOCR.

    Attributes:
    ----------
    text_rec_score_thresh : float
        Порог достоверности для распознавания текста.
    text_detection_model_dir : str
        Директория с моделью для обнаружения текста.
    text_recognition_model_dir : str
        Директория с моделью для распознавания текста.
    text_detection_model_name : str
        Имя модели для обнаружения текста.
    text_recognition_model_name : str
        Имя модели для распознавания текста.
    angles : tuple[float, ...]
        Углы поворота листа в градусах, на которых ищем текст. Детектор надёжно находит
        только текст, близкий к горизонтальному, а размерные надписи на чертеже часто идут
        вертикально вдоль размерных линий - поэтому распознавание прогоняется на нескольких
        поворотах всего листа, а не один раз.
    """

    text_rec_score_thresh: float = 0.3
    text_detection_model_dir: str = "../models/PP-OCRv6_medium_det_infer"
    text_recognition_model_dir: str = "../models/cyrillic_PP-OCRv5_mobile_rec_infer"
    text_detection_model_name: str = "PP-OCRv6_medium_det"
    text_recognition_model_name: str = "cyrillic_PP-OCRv5_mobile_rec"
    angles: tuple[float, ...] = (0, 90, 270)


@dataclass
class OcrResult:
    """Один результат распознавания текста.

    Attributes
    ----------
    text : str
        Распознанный текст.
    confidence : float
        Достоверность распознавания.
    poly : Array2D[np.float32]
        Четырёхугольник текстовой строки (4 точки, строка/столбец) в системе координат
        исходного изображения, переданного в `OCREngine.recognize`.
    """

    text: str
    confidence: float
    poly: Array2D[np.float32]


def _polys_to_original(polys: list[Array2D[np.float32]], mat: Array2D[np.float32]) -> list[Array2D[np.float32]]:
    """Переводит точки полигонов из системы координат повёрнутого изображения в исходную."""
    inv = cv2.invertAffineTransform(mat)
    return [cv2.transform(np.array([poly], dtype=np.float32), inv)[0] for poly in polys]


class OCREngine:
    """Распознавание текста на чертеже через PaddleOCR."""

    def __init__(self, params: PaddleOCRParameters, **kwargs) -> None:
        """Создаёт движок распознавания.

        Parameters
        ----------
        params : PaddleOCRParameters
            Параметры моделей детекции/распознавания и углы поворота листа.
        **kwargs
            Дополнительные параметры, переданные в `PaddleOCR` как есть.
        """
        self._params = params
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_model_dir=params.text_recognition_model_dir,
            text_detection_model_dir=params.text_detection_model_dir,
            text_recognition_model_name=params.text_recognition_model_name,
            text_detection_model_name=params.text_detection_model_name,
            text_rec_score_thresh=params.text_rec_score_thresh,
            **kwargs,
        )

    def recognize(self, image: Mat) -> list[OcrResult]:
        """Распознаёт текст на изображении, прогоняя несколько поворотов листа.

        Parameters
        ----------
        image : Mat
            Исходное изображение чертежа (RGB).

        Returns
        -------
        list[OcrResult]
            Сырые результаты распознавания по всем поворотам из `params.angles`, в
            координатах `image`. Могут содержать дубли (один и тот же текст, найденный в
            нескольких поворотах) и нетекстовый мусор - это разбирает
            `result_filtering.filter_results`.
        """
        results: list[OcrResult] = []
        for angle in self._params.angles:
            rotated, mat = rotate_image(image, angle)
            res = self._ocr.predict(rotated)[0]
            polys = _polys_to_original(res["rec_polys"], mat)
            for text, confidence, poly in zip(res["rec_texts"], res["rec_scores"], polys, strict=True):
                results.append(OcrResult(text=text, confidence=confidence, poly=poly))
        return results
