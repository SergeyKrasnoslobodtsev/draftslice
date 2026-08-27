"""Перерисовка сохраненных линий по их осям.

Удалять пиксели по принадлежности ближайшей оси нельзя: там, где тонкая линия упирается в контур,
часть пикселей контура оказывается ближе к оси тонкой линии, и наивное вычитание выкусывает из
контура клин. Поэтому итоговая маска не вычитается из исходной, а рисуется заново: каждая точка
сохраненной оси раздувается на собственный радиус из distance transform, а удаленное исчезает
просто потому, что его никто не рисует.

Радиус берется в ближайшей осевой точке через distanceTransformWithLabels: сначала считается
расстояние от каждого пикселя до множества сохраненных осей вместе с меткой ближайшей точки, затем
метка переводится в радиус этой точки. Пиксель остается, если расстояние до своей оси не превышает
ее радиус с небольшим допуском.
"""

import cv2
import numpy as np

from draftslice.common_types import Mask
from draftslice.stroke_graph import StrokeGraph


def paint_selected_paths(
    graph: StrokeGraph, kept_paths: Mask, strokes_mask: Mask, radius_tolerance: float = 0.5
) -> Mask:
    """Нарисовать сохраненные линии от их осей.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета, из которого берутся оси и радиусы.
    kept_paths : Mask
        Маска сохраненных ребер.
    strokes_mask : Mask
        Маска штрихов, из которой берутся пиксели. Результат всегда является ее подмножеством.
    radius_tolerance : float
        Допуск в пикселях, добавляемый к радиусу оси.

    Returns
    -------
    Mask
        Маска сохраненных линий.

    Notes
    -----
    Источником пикселей может быть как замкнутая маска, так и оригинальная бинаризация. Во втором
    случае искусственные перемычки замыкания в результат не попадут, и там, где контур был порван,
    он снова окажется порванным.
    """
    kept_axis = (graph.path_id_image >= 0) & kept_paths[np.maximum(graph.path_id_image, 0)]
    if not kept_axis.any():
        return np.zeros_like(strokes_mask)

    distance, labels = cv2.distanceTransformWithLabels(
        (~kept_axis).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    radius = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    radius[labels[kept_axis]] = graph.distance_map[kept_axis]
    return strokes_mask & (distance <= radius[labels] + radius_tolerance)
