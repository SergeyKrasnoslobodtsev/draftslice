"""Единая точка входа пайплайна разметки регионов чертежа.

`run()` разбивает растровый чертеж на регионы (виды, подписи, таблицы) шестью
последовательными шагами:

1. Бинаризация чертежа порогом Otsu (`THRESH_BINARY_INV + THRESH_OTSU`) - линии и
   текст чертежа становятся белыми на черном фоне.
2. Дилатация бинарной маски эллиптическим структурным элементом, размер которого
   пропорционален ширине изображения, что смыкает микроразрывы контура детали и
   подтягивает текст к линиям выносок.
3. Поиск связанных компонент дилатированной маски.
4. Классификация компонент на мелкие, seed'ы (крупные значимые компоненты) и
   краевые артефакты по их геометрии.
5. Объединение мелких компонент в чанки через Union-Find по условию совпадения
   строки или столбца, с занулением краевых меток.
6. Группировка чанков по ближайшему seed'у через `scipy.ndimage.distance_transform_edt`
   и построение финальной карты регионов изображения.

`run()` возвращает `Groups` - только финальную карту регионов,
без промежуточных артефактов, готовый для использования будущим CLI.

Notes
-----
`extract_region_crops` - отдельная функция, не входящая в `run()`, вызывается
отдельно с исходным изображением и результатом `run()`. Для каждой ненулевой
группы финальной карты регионов она вырезает прямоугольный кроп из исходного
изображения по bounding box группы и закрашивает внутри этого кропа белым (255)
пиксели, принадлежащие другой ненулевой группе; пиксели фона внутри bbox не трогает.
"""

from dataclasses import dataclass

import numpy as np

from draftslice.core.region_segmentation.comp_classification import classify_components
from draftslice.core.region_segmentation.comp_labeling import (
    find_connected_components,
)
from draftslice.core.region_segmentation.preprocess import preprocess_drawing
from draftslice.core.region_segmentation.region_grouping import Groups, group_regions
from draftslice.core.region_segmentation.region_merging import merge_into_chunks
from draftslice.core.types import Mat


@dataclass(frozen=True)
class RegionCrop:
    """Прямоугольный кроп исходного изображения по одной группе регионов.

    Attributes
    ----------
    group_label : int
        Метка группы (совпадает со значением в `group_image` результата `run()`).
    image : Mat
        Кроп исходного изображения по bounding box группы, пиксели чужих групп
        внутри этого прямоугольника закрашены белым (255).
    left : int
        Левая граница bounding box в пикселях исходного изображения.
    top : int
        Верхняя граница bounding box в пикселях исходного изображения.
    width : int
        Ширина bounding box в пикселях.
    height : int
        Высота bounding box в пикселях.
    """

    group_label: int
    image: Mat
    left: int
    top: int
    width: int
    height: int


def run(image_rgb: Mat, dilation_kernel_scale_divisor: int = 300) -> Groups:
    """Разбивает растровый чертеж на регионы (виды, подписи, таблицы)."""
    mask = preprocess_drawing(image_rgb, dilation_kernel_scale_divisor)
    labels = find_connected_components(mask)

    img_h, img_w = image_rgb.shape[:2]
    classification = classify_components(labels, img_h, img_w)

    merged_labels = merge_into_chunks(labels, classification)
    groups = group_regions(labels, classification, merged_labels)

    return groups


def extract_region_crops(image_rgb: Mat, result: Groups) -> list[RegionCrop]:
    """Вырезает из исходного изображения по одному кропу на каждую группу регионов."""
    group_image = result.group_image

    crops: list[RegionCrop] = []
    group_ids = np.unique(group_image)
    group_ids = group_ids[group_ids != 0]

    for label in group_ids:
        ys, xs = np.nonzero(group_image == label)
        top = int(ys.min())
        left = int(xs.min())
        bottom = int(ys.max()) + 1
        right = int(xs.max()) + 1

        crop = image_rgb[top:bottom, left:right].copy()
        group_crop = group_image[top:bottom, left:right]
        foreign_mask = (group_crop != label) & (group_crop != 0)
        crop[foreign_mask] = 255

        crops.append(
            RegionCrop(
                group_label=int(label),
                image=crop,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )

    return crops
