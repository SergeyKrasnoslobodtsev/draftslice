"""Классификация связанных компонент чертежа на мелкие, seed'ы и краевые артефакты.

Данный алгоритм нацелен для получения маленьких компонентов чертежа (текст, сноски и шум).
Изначально на чертеже может быть очень много компонентов и нам нужно их классифицировать
на три класса: мелкие компоненты, seed-компоненты и краевые артефакты.
Основным критерием классификации является площадь компоненты и касание её границ с краем изображения.
Мы берем среднюю площадь всех компонентов и используем её как порог для классификации.
Как правило компоненты чертежа имеют наибольшую площадь среди всех компонентов.
При этом текст как правило находится среди мелких компонентов.
"""

from dataclasses import dataclass

import numpy as np

from draftslice.core.region_segmentation.comp_labeling import ComponentLabels
from draftslice.core.types import Array


@dataclass(frozen=True)
class ComponentClassification:
    """Разбиение меток связанных компонент на три непересекающихся класса.

    Attributes
    ----------
    small_labels : Array[np.intp]
        Метки компонент с площадью меньше средней, не касающихся края листа.
    seed_labels : Array[np.intp]
        Метки компонент с площадью не меньше средней, не касающихся края листа.
    edge_labels : Array[np.intp]
        Метки компонент, чей bbox касается границы изображения.
    """

    small_labels: Array[np.intp]
    seed_labels: Array[np.intp]
    edge_labels: Array[np.intp]


def classify_components(labels: ComponentLabels, image_height: int, image_width: int) -> ComponentClassification:
    """Классифицирует связанные компоненты чертежа по площади и касанию края листа.

    Parameters
    ----------
    labels : ComponentLabels
        Результат поиска связанных компонент.
    image_height : int
        Высота исходного изображения в пикселях.
    image_width : int
        Ширина исходного изображения в пикселях.

    Returns
    -------
    ComponentClassification
        Метки мелких компонент, seed-компонент и краевых артефактов.

    Notes
    -----
    `mean_area` используется как единственный порог одновременно для отбора
    кандидатов на объединение в чанки и для отбора seed'ов - мелкое само не может
    стать seed'ом. Считается по площадям всех компонент, без исключения
    `touches_edge`. Если компонентов нет вообще (`len(areas) == 0`), функция сразу
    возвращает пустой результат, не вычисляя `np.mean` на пустом массиве (иначе
    `RuntimeWarning` и `NaN`).
    """
    if len(labels.area) == 0:
        empty_labels = np.empty(0, dtype=np.intp)
        return ComponentClassification(small_labels=empty_labels, seed_labels=empty_labels, edge_labels=empty_labels)

    touches_edge: Array[np.bool_] = (
        (labels.left <= 0) | (labels.top <= 0) | (labels.right >= image_width) | (labels.bottom >= image_height)
    )

    mean_area: np.float64 = np.mean(labels.area)

    small_labels: Array[np.intp] = np.flatnonzero((labels.area < mean_area) & ~touches_edge) + 1
    seed_labels: Array[np.intp] = np.flatnonzero((labels.area >= mean_area) & ~touches_edge) + 1
    edge_labels: Array[np.intp] = np.flatnonzero(touches_edge) + 1

    return ComponentClassification(small_labels=small_labels, seed_labels=seed_labels, edge_labels=edge_labels)
