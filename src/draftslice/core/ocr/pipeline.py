"""Единая точка входа пайплайна распознавания текста на чертеже.

`run()` разбивает распознавание на два шага:

1. `OCREngine.recognize` - распознавание на нескольких поворотах листа, чтобы находить не
   только горизонтальный, но и вертикальный текст (размерные надписи вдоль линий).
2. `result_filtering.filter_results` - отбрасывает нетекстовый мусор и дубли одной и той же
   надписи, найденной в нескольких поворотах.

`OCREngine` с загруженными моделями PaddleOCR создаётся один раз на пачку изображений и передаётся в `run()`.
"""

import numpy as np

from draftslice.core.ocr.engine import OCREngine, OcrResult
from draftslice.core.ocr.result_filtering import filter_results
from draftslice.core.utils.checks import check_image


def run(image: np.ndarray, engine: OCREngine, iou_threshold: float = 0.3) -> list[OcrResult]:
    """Распознаёт текст на чертеже.

    Parameters
    ----------
    image : np.ndarray[uint8]
        Изображение чертежа RGB `(H, W, 3)`.
    engine : OCREngine
        Заранее созданный движок распознавания.
    iou_threshold : float
        Порог перекрытия боксов, выше которого результаты из разных поворотов считаются
        одной и той же надписью - передаётся в `result_filtering.filter_results`.

    Returns
    -------
    list[OcrResult]
        Распознанный текст без мусора и дублей, координаты полигонов - в системе `image`.

    Raises
    ------
    TypeError
        `image` не `(H, W, 3)` uint8.
    """
    check_image(image, channels=3, dtype=np.uint8)
    raw_results = engine.recognize(image)
    return filter_results(raw_results, iou_threshold=iou_threshold)
