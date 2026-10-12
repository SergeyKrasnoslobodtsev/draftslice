"""Текст с маски краски: компоненты, которые OCR накрыл полигоном."""

import cv2
import numpy as np


def clear_text(binary: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    """Снимает с маски краски компоненты текста по полигонам OCR.

    Компонента связности краски - текст, если больше половины её площади лежит в полигонах OCR или она целиком
    лежит в рамке одного полигона, расширенной на полвысоты строки с каждой стороны (хвосты букв, индексы, точки
    за краем полигона). Линия чертежа, которую полигон задел, остаётся: у неё в полигоне малая доля площади, и
    в рамку она целиком не входит.

    Parameters
    ----------
    binary : np.ndarray
        Маска краски `(H, W)`, краска - ненулевые пиксели.
    polys : list[np.ndarray]
        Полигоны OCR `(n, 2)` (x, y) в координатах `binary`.

    Returns
    -------
    np.ndarray[uint8]
        Маска краски `0/255` `(H, W)` без компонент текста.
    """
    on = binary > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(on.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    tight = np.zeros(on.shape, np.uint8)
    for p in polys:
        cv2.fillConvexPoly(tight, np.round(p).astype(np.int32), 1)
    text = 2 * np.bincount(lab[on & (tight > 0)], minlength=n) > area
    for p in polys:
        text |= _inside(lab, on, _wide_box(p), area)
    text[0] = False  # фон
    return (on & ~text[lab]).astype(np.uint8) * 255


def _wide_box(poly):
    """Повёрнутая рамка полигона, расширенная на полвысоты строки с каждой стороны."""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(poly, np.float32))
    m = min(w, h) / 2
    return np.round(cv2.boxPoints(((cx, cy), (w + 2 * m, h + 2 * m), a))).astype(np.int32)


def _inside(lab, on, box, area):
    """(N,) компоненты, целиком лежащие в рамке box."""
    img_h, img_w = on.shape
    x0, y0 = np.maximum(box.min(0), 0)
    x1, y1 = np.minimum(box.max(0) + 1, (img_w, img_h))
    out = np.zeros(len(area), bool)
    if x0 >= x1 or y0 >= y1:
        return out
    roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(roi, box - (x0, y0), 1)
    cnt = np.bincount(lab[y0:y1, x0:x1][(roi > 0) & on[y0:y1, x0:x1]], minlength=len(area))
    out[(cnt > 0) & (cnt == area)] = True
    return out
