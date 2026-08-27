"""Признаки связных компонент и правила отбора того, что относится к детали.

Тип ребра плохо предсказывает, нужен ли объект на чертеже: зигзаг резьбы попадает в изоляты только
потому, что не связан с деталью, а рамка вида становится набором шпор только потому, что где-то
разомкнута. Поэтому решение принимается не по ребру, а по связной компоненте, и опирается на ее
измеримые свойства.

Компонента описывается тремя группами признаков. Размер и форма: суммарная длина осей, диагональ
габарита и их отношение, то есть извилистость. Профиль толщины: медиана и отношение процентилей
вдоль оси, которое отделяет клин наконечника от линии постоянной ширины. Топология: внутренняя
площадь, доля этой площади под текстом и вклад висячих ветвей.

Правила поверх признаков такие. Компонента живет, если у нее есть внутренняя площадь, либо она
ровная по толщине и при этом извилистая; текст внутри контура убивает компоненту в любом случае.
Отдельно снимается повторяющийся мусор: деталь на листе одна, а размерная стрелка нарисована
десятки раз одинаково, поэтому группа одинаковых по размеру незамкнутых компонент это шаблон.
Наконечники, приклеенные к детали, компонентой не отделяются и срезаются как хвосты, короткие по
сравнению со своей компонентой.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from draftslice.common_types import Floats, Ints, Mask
from draftslice.morphology_closing import fill_internal_holes
from draftslice.stroke_graph import (
    StrokeGraph,
    find_hanging_paths,
    label_connected_components,
    measure_thickness_spread,
)


@dataclass(frozen=True)
class ComponentShape:
    """Пиксельные признаки одной компоненты, снятые в ее габарите.

    Attributes
    ----------
    component_id : int
        Номер компоненты.
    bounding_diagonal : float
        Диагональ габаритного прямоугольника.
    median_thickness : float
        Медиана толщины по пикселям оси.
    thickness_spread : float
        Отношение девяностого процентиля толщины к десятому.
    inside_area : int
        Площадь внутренних пустот в пикселях.
    inside_text_share : float
        Доля внутренней площади, попавшая в маску текста.
    """

    component_id: int
    bounding_diagonal: float
    median_thickness: float
    thickness_spread: float
    inside_area: int
    inside_text_share: float


@dataclass(frozen=True, eq=False)
class ComponentFeatures:
    """Признаки всех связных компонент графа.

    Массивы одной длины описывают компоненты, кроме path_component и hanging_path, которые заданы по
    ребрам графа.

    Attributes
    ----------
    path_component : Ints
        Номер компоненты для каждого ребра графа.
    hanging_path : Mask
        Ребро имеет свободный конец.
    component_id : Ints
        Номера компонент в порядке строк остальных массивов.
    path_count : Ints
        Число ребер в компоненте.
    axis_length : Floats
        Суммарная длина осей компоненты.
    bounding_diagonal : Floats
        Диагональ габарита компоненты.
    tortuosity : Floats
        Отношение длины осей к диагонали габарита.
    median_thickness : Floats
        Медиана толщины по пикселям оси.
    thickness_spread : Floats
        Отношение процентилей толщины, признак клина.
    inside_area : Ints
        Площадь внутренних пустот.
    inside_text_share : Floats
        Доля внутренней площади под текстом.
    twig_count : Ints
        Число висячих ребер в компоненте.
    twig_length_share : Floats
        Доля длины компоненты, приходящаяся на висячие ребра.
    longest_twig_length : Floats
        Длина самого длинного висячего ребра.
    maximum_twig_spread : Floats
        Наибольшее отношение процентилей толщины среди висячих ребер.
    """

    path_component: Ints
    hanging_path: Mask
    component_id: Ints
    path_count: Ints
    axis_length: Floats
    bounding_diagonal: Floats
    tortuosity: Floats
    median_thickness: Floats
    thickness_spread: Floats
    inside_area: Ints
    inside_text_share: Floats
    twig_count: Ints
    twig_length_share: Floats
    longest_twig_length: Floats
    maximum_twig_spread: Floats

    def __len__(self) -> int:
        return len(self.component_id)


def build_component_axis_image(graph: StrokeGraph, alive_paths: Mask, path_component: Ints) -> Ints:
    """Построить карту пикселей осей в номера компонент.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    alive_paths : Mask
        Маска ребер, которые считаются существующими.
    path_component : Ints
        Номер компоненты для каждого ребра.

    Returns
    -------
    Ints
        Карта номеров компонент на пикселях осей, минус единица вне осей.
    """
    axis_image = np.full(graph.path_id_image.shape, -1, dtype=np.int64)
    on_axis = (graph.path_id_image >= 0) & alive_paths[np.maximum(graph.path_id_image, 0)]
    axis_image[on_axis] = path_component[graph.path_id_image[on_axis]]
    return axis_image


def measure_component_shape(
    component_id: int, rows: Ints, cols: Ints, distance_map: Floats, text_region_mask: Mask
) -> ComponentShape:
    """Снять пиксельные признаки компоненты в ее габарите.

    Parameters
    ----------
    component_id : int
        Номер компоненты.
    rows, cols : Ints
        Координаты пикселей оси компоненты.
    distance_map : Floats
        Distance transform маски штрихов.
    text_region_mask : Mask
        Маска текстовой области.

    Returns
    -------
    ComponentShape
        Габарит, профиль толщины и внутренняя площадь компоненты.

    Notes
    -----
    Расчет идет в габарите компоненты, а не по всему кадру, поэтому стоимость не зависит от размера
    листа.
    """
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1

    axis_mask = np.zeros((bottom - top, right - left), dtype=bool)
    axis_mask[rows - top, cols - left] = True
    holes = fill_internal_holes(axis_mask) & ~axis_mask

    widths = 2 * distance_map[rows, cols] - 1
    low, middle, high = np.percentile(widths, [10, 50, 90])
    return ComponentShape(
        component_id=component_id,
        bounding_diagonal=float(np.hypot(bottom - top, right - left)),
        median_thickness=float(middle),
        thickness_spread=float(high / max(low, 1e-6)),
        inside_area=int(holes.sum()),
        inside_text_share=float(text_region_mask[top:bottom, left:right][holes].mean()) if holes.any() else 0.0,
    )


def build_component_features(graph: StrokeGraph, alive_paths: Mask, text_region_mask: Mask) -> ComponentFeatures:
    """Снять признаки со всех связных компонент графа.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    alive_paths : Mask
        Маска ребер, которые считаются существующими.
    text_region_mask : Mask
        Маска текстовой области.

    Returns
    -------
    ComponentFeatures
        Признаки компонент вместе с разметкой ребер по компонентам.
    """
    path_component = label_connected_components(graph, alive_paths)
    component_count = int(path_component.max()) + 1
    hanging_path = find_hanging_paths(graph, alive_paths)
    twig_spread = np.nan_to_num(measure_thickness_spread(graph, np.flatnonzero(hanging_path)))

    path_count = np.bincount(path_component, minlength=component_count)
    axis_length = np.bincount(path_component, weights=graph.length, minlength=component_count)
    twig_count = np.bincount(path_component[hanging_path], minlength=component_count)
    twig_length = np.bincount(path_component, weights=graph.length * hanging_path, minlength=component_count)
    longest_twig = np.zeros(component_count)
    maximum_spread = np.zeros(component_count)
    np.maximum.at(longest_twig, path_component[hanging_path], graph.length[hanging_path])
    np.maximum.at(maximum_spread, path_component[hanging_path], twig_spread[hanging_path])

    axis_image = build_component_axis_image(graph, alive_paths, path_component)
    rows, cols = np.nonzero(axis_image >= 0)
    labels = axis_image[rows, cols]
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    shapes = [
        measure_component_shape(int(labels[group[0]]), rows[group], cols[group], graph.distance_map, text_region_mask)
        for group in groups
    ]

    component_id = np.array([shape.component_id for shape in shapes], dtype=np.int64)
    bounding_diagonal = np.array([shape.bounding_diagonal for shape in shapes])
    return ComponentFeatures(
        path_component=path_component,
        hanging_path=hanging_path,
        component_id=component_id,
        path_count=path_count[component_id],
        axis_length=axis_length[component_id],
        bounding_diagonal=bounding_diagonal,
        tortuosity=axis_length[component_id] / np.maximum(bounding_diagonal, 1.0),
        median_thickness=np.array([shape.median_thickness for shape in shapes]),
        thickness_spread=np.array([shape.thickness_spread for shape in shapes]),
        inside_area=np.array([shape.inside_area for shape in shapes], dtype=np.int64),
        inside_text_share=np.array([shape.inside_text_share for shape in shapes]),
        twig_count=twig_count[component_id],
        twig_length_share=twig_length[component_id] / np.maximum(axis_length[component_id], 1.0),
        longest_twig_length=longest_twig[component_id],
        maximum_twig_spread=maximum_spread[component_id],
    )


def select_components(
    features: ComponentFeatures,
    minimum_inside_area: int = 1000,
    maximum_thickness_spread: float = 2.5,
    minimum_tortuosity: float = 1.6,
    maximum_inside_text_share: float = 0.10,
) -> Mask:
    """Отобрать компоненты, относящиеся к детали.

    Parameters
    ----------
    features : ComponentFeatures
        Признаки компонент.
    minimum_inside_area : int
        Внутренняя площадь, начиная с которой компонента считается замкнутой фигурой.
    maximum_thickness_spread : float
        Отношение процентилей толщины, выше которого компонента считается клином.
    minimum_tortuosity : float
        Извилистость, начиная с которой незамкнутая компонента считается осмысленной линией.
    maximum_inside_text_share : float
        Доля внутренней площади под текстом, начиная с которой компонента считается выноской.

    Returns
    -------
    Mask
        Маска компонент, которые остаются на чертеже.
    """
    closed = features.inside_area >= minimum_inside_area
    even_and_winding = (features.thickness_spread <= maximum_thickness_spread) & (
        features.tortuosity >= minimum_tortuosity
    )
    return (closed | even_and_winding) & (features.inside_text_share < maximum_inside_text_share)


def group_similar_components(features: ComponentFeatures, tolerance: float = 0.05) -> Ints:
    """Сгруппировать компоненты, совпадающие по размеру и форме.

    Parameters
    ----------
    features : ComponentFeatures
        Признаки компонент.
    tolerance : float
        Относительный допуск на совпадение длины, габарита и извилистости.

    Returns
    -------
    Ints
        Номер группы для каждой компоненты.

    Notes
    -----
    Размеры сравниваются в логарифмах, поэтому допуск относительный. Группы собираются связностью, а
    не округлением по сетке: цепочка близких размеров не должна рассыпаться на границе ячейки.
    """
    signature = np.stack(
        [
            np.log(np.maximum(features.axis_length, 1.0)),
            np.log(np.maximum(features.bounding_diagonal, 1.0)),
            np.log(np.maximum(features.tortuosity, 1.0)),
        ],
        axis=1,
    )
    close = np.abs(signature[:, None, :] - signature[None, :, :]).max(axis=2) <= tolerance
    return connected_components(coo_matrix(close), directed=False)[1]


def find_repeated_components(features: ComponentFeatures, groups: Ints, minimum_repeat_count: int = 3) -> Mask:
    """Найти незамкнутые компоненты, повторяющиеся на листе как шаблон.

    Parameters
    ----------
    features : ComponentFeatures
        Признаки компонент.
    groups : Ints
        Номер группы для каждой компоненты.
    minimum_repeat_count : int
        Размер группы, начиная с которого компоненты считаются шаблонными.

    Returns
    -------
    Mask
        Маска компонент, признанных повторяющимся мусором.

    Notes
    -----
    Замкнутые компоненты исключены: два одинаковых отверстия на детали не должны считаться мусором.
    """
    group_size = np.bincount(groups)
    return (group_size[groups] >= minimum_repeat_count) & (features.inside_area == 0)


def expand_selection_to_paths(features: ComponentFeatures, kept_components: Mask) -> Mask:
    """Развернуть решение по компонентам в маску ребер.

    Parameters
    ----------
    features : ComponentFeatures
        Признаки компонент.
    kept_components : Mask
        Маска компонент, которые остаются.

    Returns
    -------
    Mask
        Маска ребер графа.
    """
    lookup = np.zeros(int(features.path_component.max()) + 1, dtype=bool)
    lookup[features.component_id] = kept_components
    return lookup[features.path_component]


def trim_short_twigs(
    graph: StrokeGraph, alive_paths: Mask, path_component: Ints, twig_length_share: float = 0.05
) -> Mask:
    """Срезать хвосты, короткие по сравнению со своей компонентой.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    alive_paths : Mask
        Маска ребер, с которой начинается обрезка.
    path_component : Ints
        Номер компоненты для каждого ребра.
    twig_length_share : float
        Доля длины компоненты, ниже которой висячее ребро считается хвостом.

    Returns
    -------
    Mask
        Маска ребер после обрезки.

    Notes
    -----
    Порог относительный, поэтому переносится между кадрами разного разрешения. Длина компоненты
    пересчитывается на каждой итерации по выжившим ребрам, а сама обрезка идет до неподвижной точки:
    после удаления наконечника его корешок сам становится хвостом.
    """
    component_count = int(path_component.max()) + 1
    kept = alive_paths.copy()
    while True:
        component_length = np.bincount(path_component, weights=graph.length * kept, minlength=component_count)
        cut = find_hanging_paths(graph, kept) & (graph.length < twig_length_share * component_length[path_component])
        if not cut.any():
            return kept
        kept &= ~cut
