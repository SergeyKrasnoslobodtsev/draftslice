"""Разбиение очищенного листа на отдельные виды.

`part_mask` содержит только линии детали, но одним растром на весь лист: несколько ортогональных
проекций, разнесенных пустым полем. Куски, из которых состоит один вид, сами по себе не всегда
связны — разрыв размерной выноски, отверстие внутри контура или условный обрыв длинной детали
(волнистая линия) режут один вид на несколько кусков растра. Решение о том, что считать одним видом,
принимается по геометрии кусков и толщине линии на листе, а не по содержимому.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from draftslice.common_types import IntMap, Ints, Mask, Mat
from draftslice.image_preparation import scale_mask

Contour = np.ndarray
"""Один контур из cv2.findContours: массив точек (N, 1, 2)."""


@dataclass(frozen=True)
class ViewGroups:
    """Связные группы кусков `part_mask` после слияния близких по зазору.

    Attributes
    ----------
    labels : IntMap
        Номер группы на пиксель `part_mask`, 0 вне маски.
    group_count : int
        Число групп после слияния.
    raw_component_count : int
        Число кусков (контуров) до слияния, для сравнения.
    """

    labels: IntMap
    group_count: int
    raw_component_count: int


@dataclass(frozen=True)
class ContourBox:
    """Габарит контура, посчитанный один раз и переиспользуемый для всех пар.

    Attributes
    ----------
    left, top, right, bottom : int
        Границы габаритного прямоугольника.
    """

    left: int
    top: int
    right: int
    bottom: int


def find_piece_contours(mask: Mask) -> list[Contour]:
    """Найти внешние контуры кусков `part_mask`.

    Parameters
    ----------
    mask : Mask
        Маска детали (`part_mask`).

    Returns
    -------
    list[Contour]
        Контуры кусков, один на связный кусок растра.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def measure_contour_box(contour: Contour) -> ContourBox:
    """Габаритный прямоугольник одного контура.

    Parameters
    ----------
    contour : Contour
        Контур куска.

    Returns
    -------
    ContourBox
        Границы габарита.
    """
    left, top, width, height = cv2.boundingRect(contour)
    return ContourBox(left=left, top=top, right=left + width - 1, bottom=top + height - 1)


def is_nested(inner: ContourBox, outer: ContourBox) -> bool:
    """Проверить, лежит ли габарит `inner` целиком внутри габарита `outer`.

    Parameters
    ----------
    inner : ContourBox
        Габарит куска-кандидата на вложенность.
    outer : ContourBox
        Габарит потенциального хозяина.

    Returns
    -------
    bool
        True, если `inner` целиком внутри `outer`.
    """
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def measure_nearest_point_gap(contour_a: Contour, contour_b: Contour) -> float:
    """Минимальное расстояние между точками двух контуров.

    Parameters
    ----------
    contour_a, contour_b : Contour
        Контуры кусков.

    Returns
    -------
    float
        Расстояние между ближайшей парой точек, в пикселях.
    """
    points_a = contour_a.reshape(-1, 2).astype(np.float64)
    points_b = contour_b.reshape(-1, 2).astype(np.float64)
    return float(cdist(points_a, points_b).min())


def contours_belong_together(
    contour_a: Contour, box_a: ContourBox, contour_b: Contour, box_b: ContourBox, merge_gap: float
) -> bool:
    """Решить, относятся ли два куска к одному виду.

    Parameters
    ----------
    contour_a, contour_b : Contour
        Контуры кусков.
    box_a, box_b : ContourBox
        Их предпосчитанные габариты.
    merge_gap : float
        Наибольший зазор между кусками в пикселях.

    Returns
    -------
    bool
        True, если куски нужно слить в одну группу.

    Notes
    -----
    Вложенный габарит — это всегда один и тот же вид (отверстие внутри контура), порог тут ни при
    чем.
    """
    if is_nested(box_a, box_b) or is_nested(box_b, box_a):
        return True
    return measure_nearest_point_gap(contour_a, contour_b) <= merge_gap


def merge_pieces(contours: list[Contour], merge_gap: float) -> Ints:
    """Слить куски в группы через попарное решение о принадлежности одному виду.

    Parameters
    ----------
    contours : list[Contour]
        Контуры кусков.
    merge_gap : float
        Наибольший зазор между кусками в пикселях.

    Returns
    -------
    Ints
        Номер группы (с нуля) для каждого контура, по порядку `contours`.

    Notes
    -----
    Слияние транзитивно: если кусок A рядом с B, а B рядом с C, то A, B и C — одна группа, даже если
    A и C далеко друг от друга напрямую. Это union-find по попарным решениям `contours_belong_together`.
    """
    boxes = [measure_contour_box(contour) for contour in contours]
    parent = list(range(len(contours)))

    def find_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            if not contours_belong_together(contours[i], boxes[i], contours[j], boxes[j], merge_gap):
                continue
            root_i, root_j = find_root(i), find_root(j)
            if root_i != root_j:
                parent[root_i] = root_j

    roots = [find_root(index) for index in range(len(contours))]
    group_of_root = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.array([group_of_root[root] for root in roots], dtype=np.int64)


def render_group_labels(mask: Mask, contours: list[Contour], group_of_contour: Ints) -> IntMap:
    """Растеризовать группы кусков обратно на пиксели `part_mask`.

    Parameters
    ----------
    mask : Mask
        Маска детали (`part_mask`), задает итоговый набор пикселей.
    contours : list[Contour]
        Контуры кусков, тот же порядок, что при слиянии.
    group_of_contour : Ints
        Номер группы (с нуля) для каждого контура.

    Returns
    -------
    IntMap
        Номер группы на пиксель, 0 вне маски.

    Notes
    -----
    Контуры заливаются от большего к меньшему, чтобы вложенный контур (отверстие) перекрывал
    заливку хозяина в своей области и получал в итоге собственный номер группы, а не номер хозяина
    по ошибке порядка отрисовки.
    """
    draw_order = sorted(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]), reverse=True)
    raw_canvas: IntMap = np.zeros(mask.shape, dtype=np.int32)
    for index in draw_order:
        cv2.drawContours(raw_canvas, contours, index, index + 1, thickness=cv2.FILLED)

    group_of_piece = np.concatenate(([0], group_of_contour + 1)).astype(np.int32)
    return np.where(mask, group_of_piece[raw_canvas], 0).astype(np.int32)


def group_mask_components(mask: Mask, line_thickness: float, merge_gap_thickness: float) -> ViewGroups:
    """Слить куски `part_mask` в группы, соответствующие отдельным видам.

    Parameters
    ----------
    mask : Mask
        Маска детали (`part_mask`).
    line_thickness : float
        Толщина линии на листе, например `strokes_graph.median_thickness`.
    merge_gap_thickness : float
        Наибольший зазор между кусками как число толщин линии.

    Returns
    -------
    ViewGroups
        Разметка по группам вместе со счетчиками кусков до и после слияния.

    Notes
    -----
    Зазор задается не в абсолютных пикселях и не долей габарита куска, а числом толщин линии: разрыв
    размерной выноски или обрыв длинной детали рисуется пропорционально толщине пера, а не размеру
    куска, поэтому именно толщина линии — устойчивый ориентир, общий для всего листа.
    """
    contours = find_piece_contours(mask)
    if not contours:
        return ViewGroups(labels=np.zeros(mask.shape, dtype=np.int32), group_count=0, raw_component_count=0)

    merge_gap = merge_gap_thickness * line_thickness
    group_of_contour = merge_pieces(contours, merge_gap)
    labels = render_group_labels(mask, contours, group_of_contour)

    return ViewGroups(
        labels=labels, group_count=int(group_of_contour.max()) + 1, raw_component_count=len(contours)
    )


def measure_group_box(labels: IntMap, group_id: int) -> ContourBox:
    """Габаритный прямоугольник одной группы на разметке.

    Parameters
    ----------
    labels : IntMap
        Номер группы на пиксель, как в `ViewGroups.labels`.
    group_id : int
        Номер группы, от 1 до `ViewGroups.group_count`.

    Returns
    -------
    ContourBox
        Границы габарита группы.
    """
    rows, cols = np.nonzero(labels == group_id)
    return ContourBox(left=int(cols.min()), top=int(rows.min()), right=int(cols.max()), bottom=int(rows.max()))


def pad_box(box: ContourBox, padding: float, shape: tuple[int, int]) -> ContourBox:
    """Расширить габарит на отступ и обрезать по границам кадра.

    Parameters
    ----------
    box : ContourBox
        Исходный габарит.
    padding : float
        Отступ в пикселях со всех сторон.
    shape : tuple[int, int]
        Размер кадра (высота, ширина), которым обрезается результат.

    Returns
    -------
    ContourBox
        Расширенный и обрезанный по кадру габарит.
    """
    height, width = shape
    return ContourBox(
        left=max(0, int(box.left - padding)),
        top=max(0, int(box.top - padding)),
        right=min(width - 1, int(box.right + padding)),
        bottom=min(height - 1, int(box.bottom + padding)),
    )


def crop_box(mask: Mask, box: ContourBox) -> Mask:
    """Вырезать прямоугольную область маски.

    Parameters
    ----------
    mask : Mask
        Маска, из которой вырезается область.
    box : ContourBox
        Границы вырезаемой области.

    Returns
    -------
    Mask
        Вырезанный фрагмент маски.
    """
    return mask[box.top : box.bottom + 1, box.left : box.right + 1]


def extract_view_crops(mask: Mask, groups: ViewGroups, padding: float) -> list[Mask]:
    """Вырезать по одному кропу на группу из чистой маски детали.

    Parameters
    ----------
    mask : Mask
        Маска детали (`part_mask`), из которой вырезаются кропы.
    groups : ViewGroups
        Группы кусков той же маски.
    padding : float
        Отступ вокруг габарита группы в пикселях.

    Returns
    -------
    list[Mask]
        Кропы видов в порядке номеров групп, с первой по `groups.group_count`.
    """
    height, width = mask.shape
    crops = []
    for group_id in range(1, groups.group_count + 1):
        box = measure_group_box(groups.labels, group_id)
        box = pad_box(box, padding, (height, width))
        crops.append(crop_box(mask, box))
    return crops


def fit_scale(shape: tuple[int, int], target_size: int) -> float:
    """Множитель, чтобы бОльшая сторона кадра стала равна `target_size`.

    Parameters
    ----------
    shape : tuple[int, int]
        Размер кадра (высота, ширина).
    target_size : int
        Целевой размер большей стороны в пикселях.

    Returns
    -------
    float
        Множитель масштаба.
    """
    height, width = shape
    return target_size / max(height, width)


def pad_to_square(mask: Mask, target_size: int) -> Mask:
    """Дополнить маску пустым полем до квадрата, отцентровав содержимое.

    Parameters
    ----------
    mask : Mask
        Маска, не крупнее `target_size` по обеим сторонам.
    target_size : int
        Сторона итогового квадрата в пикселях.

    Returns
    -------
    Mask
        Маска `target_size` x `target_size` с исходным содержимым по центру.
    """
    height, width = mask.shape
    canvas: Mask = np.zeros((target_size, target_size), dtype=bool)
    top, left = (target_size - height) // 2, (target_size - width) // 2
    canvas[top : top + height, left : left + width] = mask
    return canvas


def resize_view_crop(crop: Mask, scale: float, target_size: int) -> Mask:
    """Отмасштабировать кроп на заданный множитель и дополнить до квадрата.

    Parameters
    ----------
    crop : Mask
        Кроп вида.
    scale : float
        Множитель масштаба — общий для всех видов одного листа, см. `measure_shared_scale`.
    target_size : int
        Сторона итогового квадрата в пикселях.

    Returns
    -------
    Mask
        Кроп после масштабирования, вписанный в квадрат `target_size` x `target_size`.

    Notes
    -----
    Масштаб приходит извне, а не считается по своему собственному габариту: у видов одного листа
    он общий, иначе крупный и мелкий вид после вписывания в квадрат окажутся одного размера.
    """
    return pad_to_square(scale_mask(crop, scale), target_size)


def box_area(box: ContourBox) -> int:
    """Площадь габарита в пикселях.

    Parameters
    ----------
    box : ContourBox
        Габарит.

    Returns
    -------
    int
        Площадь габарита.
    """
    return (box.right - box.left + 1) * (box.bottom - box.top + 1)


def measure_shared_scale(padded_boxes: list[ContourBox], target_size: int) -> float:
    """Общий для листа масштаб: у самого крупного вида бОльшая сторона станет равна `target_size`.

    Parameters
    ----------
    padded_boxes : list[ContourBox]
        Габариты видов листа, уже с добавленным отступом.
    target_size : int
        Сторона итогового квадрата в пикселях.

    Returns
    -------
    float
        Единый множитель масштаба для всех видов листа.
    """
    largest_side = max(max(box.bottom - box.top + 1, box.right - box.left + 1) for box in padded_boxes)
    return fit_scale((largest_side, largest_side), target_size)


def normalize_view_crops(mask: Mask, groups: ViewGroups, padding: float, target_size: int) -> list[Mask]:
    """Вырезать виды листа в едином масштабе и отсортировать от крупного к мелкому.

    Parameters
    ----------
    mask : Mask
        Маска детали (`part_mask`).
    groups : ViewGroups
        Группы кусков той же маски.
    padding : float
        Отступ вокруг габарита группы в пикселях, тот же, что и при обычной вырезке.
    target_size : int
        Сторона итогового квадрата в пикселях.

    Returns
    -------
    list[Mask]
        Кропы видов, все в едином масштабе и вписанные в квадрат `target_size`, от самого крупного
        вида к самому мелкому, а при близкой площади — сверху вниз и слева направо по расположению
        на листе.

    Notes
    -----
    Масштаб общий для всех видов одного листа (см. `measure_shared_scale`): маленькая выноска,
    отмасштабированная под свой собственный квадрат, выглядела бы такого же размера, как длинный
    вал детали, хотя на самом чертеже она в разы меньше.
    """
    height, width = mask.shape
    boxes = [measure_group_box(groups.labels, group_id) for group_id in range(1, groups.group_count + 1)]
    padded_boxes = [pad_box(box, padding, (height, width)) for box in boxes]
    scale = measure_shared_scale(padded_boxes, target_size)

    crops = [resize_view_crop(crop_box(mask, padded_box), scale, target_size) for padded_box in padded_boxes]
    order = sorted(
        range(len(boxes)), key=lambda index: (-box_area(boxes[index]), boxes[index].top, boxes[index].left)
    )
    return [crops[index] for index in order]


def render_view(crop: Mask) -> Mat:
    """Отрисовать кроп вида как RGB-картинку: черные линии на белом поле.

    Parameters
    ----------
    crop : Mask
        Кроп вида, True — чернила.

    Returns
    -------
    Mat
        RGB-картинка того же размера, три одинаковых канала.
    """
    gray = np.where(crop, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
