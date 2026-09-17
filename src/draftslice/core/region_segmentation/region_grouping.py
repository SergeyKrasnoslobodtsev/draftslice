"""Группировка чанков чертежа по ближайшему seed-компоненту.

Принцип алгоритма состоит в том, что каждый чанк чертежа присваивается
ближайшему seed-компоненту. Для этого используется distance transform: сначала
создается карта меток, где оставлены только seed-пиксели, затем для каждого пикселя
вычисляются координаты ближайшего seed-пикселя. Чанк получает группу того seed'а,
который чаще всего оказывается ближайшим среди его пикселей.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from draftslice.core.exceptions import ArrayNullError
from draftslice.core.region_segmentation.comp_classification import ComponentClassification
from draftslice.core.region_segmentation.comp_labeling import ComponentLabels
from draftslice.core.types import Array, Array2D


@dataclass(frozen=True)
class Groups:
    """Результат группировки чанков чертежа по ближайшему seed-компоненту.

    Attributes
    ----------
    groups : Array[np.int32]
        Таблица меток чанков в метки их groups-seed'а.
    group_image : Array2D[np.int32]
        Финальная карта регионов на всем изображении.
    """

    groups: Array[np.int32]
    group_image: Array2D[np.int32]


def group_regions(
    labels: ComponentLabels, classification: ComponentClassification, merged_labels: Array2D[np.int32]
) -> Groups:
    """Группирует чанки чертежа по ближайшему seed-компоненту через distance transform.

    Parameters
    ----------
    labels : ComponentLabels
        Метки компонентов и их статистики, используется только число компонент.
    classification : ComponentClassification
        Классификация компонент, используются `seed_labels`.
    merged_labels : Array[np.int32]
        Карта меток после объединения мелких компонент в чанки.

    Returns
    -------
    Groups
        Таблица групп чанков и финальная карта регионов.
    """
    number_of_components: int = len(labels.area)

    unique_labels = np.unique(merged_labels)
    unique_labels = unique_labels[unique_labels != 0]

    is_seed_lookup = np.zeros(number_of_components + 1, dtype=bool)
    is_seed_lookup[classification.seed_labels] = True
    markers = np.where(is_seed_lookup[merged_labels], merged_labels, 0)

    if not np.any(markers):
        raise ArrayNullError("Нет seed-компонентов для группировки.")

    _, nearest_seed_pixel_indices = distance_transform_edt(markers == 0, return_indices=True)
    nearest_marker = markers[nearest_seed_pixel_indices[0], nearest_seed_pixel_indices[1]]

    groups = np.zeros(number_of_components + 1, dtype=np.int32)
    groups[classification.seed_labels] = classification.seed_labels
    for label in unique_labels:
        if is_seed_lookup[label]:
            continue
        votes = nearest_marker[merged_labels == label]
        groups[label] = np.bincount(votes).argmax()

    group_image = groups[merged_labels]
    return Groups(groups=groups, group_image=group_image)
