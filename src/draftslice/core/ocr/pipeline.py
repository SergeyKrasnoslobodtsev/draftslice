"""Единая точка входа пайплайна распознавания текста на чертеже.

`run()` разбивает распознавание на два шага:

1. `OCREngine.recognize` - распознавание на нескольких поворотах листа, чтобы находить не
   только горизонтальный, но и вертикальный текст (размерные надписи вдоль линий).
2. `result_filtering.filter_results` - отбрасывает нетекстовый мусор и дубли одной и той же
   надписи, найденной в нескольких поворотах.

`OCREngine` держит инициализированные модели PaddleOCR, поэтому создаётся и передаётся вызывающим
кодом один раз на пачку изображений, а не пересоздаётся внутри `run()` на каждый вызов.
"""

from draftslice.core.ocr.engine import OCREngine, OcrResult
from draftslice.core.ocr.result_filtering import filter_results
from draftslice.core.types import Mat


def run(image: Mat, engine: OCREngine, iou_threshold: float = 0.3) -> list[OcrResult]:
    """Распознаёт текст на чертеже.

    Parameters
    ----------
    image : Mat
        Изображение чертежа (RGB).
    engine : OCREngine
        Заранее созданный движок распознавания.
    iou_threshold : float
        Порог перекрытия боксов, выше которого результаты из разных поворотов считаются
        одной и той же надписью - передаётся в `result_filtering.filter_results`.

    Returns
    -------
    list[OcrResult]
        Распознанный текст без мусора и дублей, координаты полигонов - в системе `image`.
    """
    raw_results = engine.recognize(image)
    return filter_results(raw_results, iou_threshold=iou_threshold)
