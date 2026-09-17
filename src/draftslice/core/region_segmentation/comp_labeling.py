"""Поиск связанных компонент дилатированной маски чертежа.

На начальном этапе идея заключается в том, чтобы найти все
внешние контуры дилатированной маски и затем определить
связанные компоненты на основе этих контуров. Это немного грубо
если посмотреть отладочные результаты, но дает хороший старт
для дальнейшей обработки.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from draftslice.core.exceptions import DraftsliceValueError
from draftslice.core.types import Array, Array2D, Mat


@dataclass(frozen=True)
class ComponentLabels:
    """Результат поиска связанных компонент дилатированной маски чертежа.

    Attributes
    ----------
    labels : Array2D[np.intp]
        Метки каждой компоненты.
    area : Array[np.float32]
        Площадь каждой компоненты.
    left : Array[np.int32]
        Координаты левой границы каждой компоненты.
    top : Array[np.int32]
        Координаты верхней границы каждой компоненты.
    right : Array[np.int32]
        Координаты правой границы каждой компоненты.
    bottom : Array[np.int32]
        Координаты нижней границы каждой компоненты.
    width : Array[np.int32]
        Ширина каждой компоненты.
    height : Array[np.int32]
        Высота каждой компоненты.

    """

    labels: Array2D[np.intp]
    area: Array[np.float32]
    left: Array[np.int32]
    top: Array[np.int32]
    right: Array[np.int32]
    bottom: Array[np.int32]
    width: Array[np.int32]
    height: Array[np.int32]


def _get_filled_mask(
    image_size: tuple[int, int],
    contours: Sequence[Array2D[np.int32]],
    outer_indices: Array[np.intp],
) -> Mat:
    """Возвращает маску с залитыми внешними контурами чертежа."""
    filled = np.zeros(image_size, dtype=np.uint8)
    for idx in outer_indices:
        cv2.drawContours(filled, contours, int(idx), 255, cv2.FILLED)
    return filled


def find_connected_components(dilated_mask: Mat) -> ComponentLabels:
    """Находит связанные компоненты дилатированной маски чертежа.

    Parameters
    ----------
    dilated_mask : Mat
        Дилатированная бинарная маска чертежа, значения 0 или 255.

    Returns
    -------
    ComponentLabels
        Координаты границ, ширина и высота каждой компоненты.

    Raises
    ------
    DraftsliceValueError
        Если во входной маске нет ни одного foreground пикселя.
    """
    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise DraftsliceValueError("Маска не содержит ни одного foreground пикселя.")

    outer_indices = np.flatnonzero(hierarchy[0][:, 3] == -1)
    image_size = (dilated_mask.shape[0], dilated_mask.shape[1])
    contours = cast(Sequence[Array2D[np.int32]], contours)
    filled_mask = _get_filled_mask(image_size, contours, outer_indices)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)

    return ComponentLabels(
        labels=labels.astype(np.intp),
        area=stats[1:, cv2.CC_STAT_AREA].astype(np.float32),
        left=stats[1:, cv2.CC_STAT_LEFT].astype(np.int32),
        top=stats[1:, cv2.CC_STAT_TOP].astype(np.int32),
        right=(stats[1:, cv2.CC_STAT_LEFT] + stats[1:, cv2.CC_STAT_WIDTH]).astype(np.int32),
        bottom=(stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]).astype(np.int32),
        width=stats[1:, cv2.CC_STAT_WIDTH].astype(np.int32),
        height=stats[1:, cv2.CC_STAT_HEIGHT].astype(np.int32),
    )
