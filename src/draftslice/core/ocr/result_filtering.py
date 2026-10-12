from draftslice.core.ocr.engine import OcrResult


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _poly_bbox(poly) -> tuple[float, float, float, float]:
    return poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_results(results: list[OcrResult], iou_threshold: float = 0.3) -> list[OcrResult]:
    """Убирает нетекстовый мусор и дубли между поворотами листа.

    `OCREngine.recognize` прогоняет распознавание на нескольких поворотах одного и того же
    листа, поэтому один и тот же текст может встретиться несколько раз - в координатах
    исходного изображения такие результаты дают перекрывающиеся полигоны.

    Parameters
    ----------
    results : list[OcrResult]
        Сырые результаты `OCREngine.recognize`.
    iou_threshold : float
        Порог перекрытия (IoU осевых прямоугольников полигонов), выше которого два
        результата считаются одной и той же надписью.

    Returns
    -------
    list[OcrResult]
        Результаты без нетекстового мусора (не содержащих ни одной буквы/цифры) и без
        дублей - среди перекрывающихся результатов оставлен только самый уверенный.

    Notes
    -----
    Перекрытие считается по осевому прямоугольнику полигона (IoU), не по самому
    четырёхугольнику - для различения "одна и та же надпись" от "разные надписи рядом" этого
    достаточно, а точное пересечение произвольных четырёхугольников не даёт здесь выигрыша.
    """
    textual = [r for r in results if _has_alnum(r.text)]

    ordered = sorted(textual, key=lambda r: r.confidence, reverse=True)
    kept: list[OcrResult] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    for result in ordered:
        box = _poly_bbox(result.poly)
        if any(_bbox_iou(box, kept_box) > iou_threshold for kept_box in kept_boxes):
            continue
        kept.append(result)
        kept_boxes.append(box)
    return kept
