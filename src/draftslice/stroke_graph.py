"""Граф скелета чертежа: ребра, их геометрия и связи.

Скелетизация превращает маску штрихов в линии толщиной в один пиксель, а skan режет их на ребра в
каждом узле. Ребро это участок оси между двумя узлами, между узлом и свободным концом или замкнутая
петля. Такое разбиение нужно потому, что решения о том, что оставить на чертеже, принимаются целыми
ребрами: пиксельные правила рвут линии в местах пересечений, а реберные сохраняют их целиком.

Толщина штриха берется из distance transform: значение в точке оси равно расстоянию до фона, то
есть половине локальной ширины. Значения снимаются только со срединной части ребра, потому что у
узлов и свободных концов оценка занижена по построению и смазывает распределение толщин.

Доля текста считается сразу по пикселям ребра: маска текста нужна не для того, чтобы что-то стереть,
а чтобы буквы не участвовали в оценке границ классов толщины.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from draftslice.common_types import FloatMap, Floats, IntMap, Ints, Mask

ISOLATED_BRANCH = 0
"""Ребро между двумя свободными концами."""

SPUR_BRANCH = 1
"""Ребро между узлом и свободным концом."""

JUNCTION_BRANCH = 2
"""Ребро между двумя узлами."""

CYCLE_BRANCH = 3
"""Замкнутая петля."""


@dataclass(frozen=True, eq=False)
class StrokeGraph:
    """Скелет чертежа: ребра, их геометрия и связи.

    Attributes
    ----------
    skeleton : Skeleton
        Разбиение скелета на ребра, объект skan.
    distance_map : FloatMap
        Distance transform маски штрихов, половина локальной ширины в каждой точке.
    path_id_image : IntMap
        Карта пикселей скелета в номер ребра, минус единица вне скелета.
    thickness : Floats
        Толщина каждого ребра в пикселях.
    length : Floats
        Длина каждого ребра вдоль оси.
    pixel_count : Ints
        Число пикселей оси каждого ребра.
    branch_type : Ints
        Тип ребра: изолят, шпора, узел-узел или цикл.
    source_node, target_node : Ints
        Номера узлов на концах ребра.
    node_degree : Ints
        Степень каждого узла скелета.
    text_share : Floats
        Доля пикселей ребра, попавших в маску текста.
    """

    skeleton: Skeleton
    distance_map: FloatMap
    path_id_image: IntMap
    thickness: Floats
    length: Floats
    pixel_count: Ints
    branch_type: Ints
    source_node: Ints
    target_node: Ints
    node_degree: Ints
    text_share: Floats

    @property
    def path_count(self) -> int:
        """Число ребер скелета."""
        return int(self.skeleton.n_paths)

    @property
    def node_count(self) -> int:
        """Число узлов скелета, включая свободные концы."""
        return len(self.skeleton.coordinates)

    @property
    def median_thickness(self) -> float:
        """Медиана толщины по всем ребрам."""
        return float(np.nanmedian(self.thickness))

    def path_points(self, path_index: int) -> Floats:
        """Точки оси ребра в порядке x, y.

        Parameters
        ----------
        path_index : int
            Номер ребра.

        Returns
        -------
        Floats
            Массив формы (n, 2) с координатами в порядке, удобном для matplotlib.
        """
        return self.skeleton.path_coordinates(path_index)[:, ::-1]


def median_by_group(values: Floats, group_labels: Ints, group_count: int) -> Floats:
    """Медиана значений внутри каждой группы.

    Parameters
    ----------
    values : Floats
        Значения, по одному на элемент.
    group_labels : Ints
        Номер группы для каждого элемента.
    group_count : int
        Общее число групп.

    Returns
    -------
    Floats
        Медиана каждой группы, пустая группа дает nan.
    """
    order = np.lexsort((values, group_labels))
    sorted_values = values[order]
    sorted_labels = group_labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(group_count), side="left")
    stops = np.searchsorted(sorted_labels, np.arange(group_count), side="right")

    medians = np.full(group_count, np.nan)
    filled = stops > starts
    medians[filled] = sorted_values[((starts + stops) // 2)[filled]]
    return medians


def measure_path_thickness(skeleton: Skeleton, distance_map: FloatMap, trim_share: float = 0.25) -> Floats:
    """Измерить толщину каждого ребра по срединной части его оси.

    Parameters
    ----------
    skeleton : Skeleton
        Разбиение скелета на ребра.
    distance_map : FloatMap
        Distance transform маски штрихов.
    trim_share : float
        Доля длины, отбрасываемая с каждого конца ребра.

    Returns
    -------
    Floats
        Толщина каждого ребра, вычисленная как медиана удвоенного расстояния до фона.

    Notes
    -----
    У узлов и свободных концов distance transform занижает ширину, поэтому концы ребра в оценке не
    участвуют.
    """
    pixels_per_path = np.diff(skeleton.paths.indptr)
    path_labels = np.repeat(np.arange(skeleton.n_paths), pixels_per_path)
    position = np.arange(len(path_labels)) - np.repeat(skeleton.paths.indptr[:-1], pixels_per_path)
    path_length = np.repeat(pixels_per_path, pixels_per_path)
    inner = (position >= trim_share * path_length) & (position < (1.0 - trim_share) * path_length)

    coordinates = skeleton.coordinates[skeleton.paths.indices].astype(int)
    widths = 2 * distance_map[coordinates[:, 0], coordinates[:, 1]] - 1
    return median_by_group(widths[inner], path_labels[inner], skeleton.n_paths)


def build_path_id_image(skeleton: Skeleton, shape: tuple[int, int]) -> IntMap:
    """Построить карту пикселей скелета в номера ребер.

    Parameters
    ----------
    skeleton : Skeleton
        Разбиение скелета на ребра.
    shape : tuple[int, int]
        Размер кадра в порядке высота, ширина.

    Returns
    -------
    IntMap
        Карта номеров ребер, минус единица вне скелета.
    """
    path_ids = np.full(shape, -1, dtype=np.int32)
    coordinates = skeleton.coordinates[skeleton.paths.indices].astype(int)
    path_ids[coordinates[:, 0], coordinates[:, 1]] = np.repeat(
        np.arange(skeleton.n_paths), np.diff(skeleton.paths.indptr)
    )
    return path_ids


def build_owner_image(path_id_image: IntMap, strokes_mask: Mask) -> IntMap:
    """Отнести каждый пиксель штриха к ближайшему ребру скелета.

    Parameters
    ----------
    path_id_image : IntMap
        Карта пикселей скелета в номера ребер.
    strokes_mask : Mask
        Маска штрихов.

    Returns
    -------
    IntMap
        Номер ребра для каждого пикселя штриха, минус единица вне штрихов.

    Notes
    -----
    Нужна только для метрик: показывает, сколько пикселей выкусил бы наивный подход, удаляющий
    пиксели по принадлежности ближайшей оси. Для построения итоговой маски применяется перерисовка
    от оси, а не эта карта.
    """
    _, labels = cv2.distanceTransformWithLabels(
        (path_id_image < 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    on_skeleton = path_id_image >= 0
    lookup = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    lookup[labels[on_skeleton]] = path_id_image[on_skeleton]
    return np.where(strokes_mask, lookup[labels], -1).astype(np.int32)


def measure_path_text_share(skeleton: Skeleton, text_region_mask: Mask) -> Floats:
    """Измерить долю пикселей каждого ребра, попавших в маску текста.

    Parameters
    ----------
    skeleton : Skeleton
        Разбиение скелета на ребра.
    text_region_mask : Mask
        Маска текстовой области.

    Returns
    -------
    Floats
        Доля пикселей внутри маски текста для каждого ребра.
    """
    coordinates = skeleton.coordinates[skeleton.paths.indices].astype(int)
    path_labels = np.repeat(np.arange(skeleton.n_paths), np.diff(skeleton.paths.indptr))
    inside = text_region_mask[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
    inside_count = np.bincount(path_labels, weights=inside, minlength=skeleton.n_paths)
    return inside_count / np.maximum(np.diff(skeleton.paths.indptr), 1)


def build_stroke_graph(strokes_mask: Mask, text_region_mask: Mask) -> StrokeGraph:
    """Построить граф скелета по маске штрихов.

    Parameters
    ----------
    strokes_mask : Mask
        Бинарная маска штрихов, True это чернила.
    text_region_mask : Mask
        Маска текстовой области того же размера.

    Returns
    -------
    StrokeGraph
        Ребра скелета с толщиной, длиной, типом ветвления и долей текста.
    """
    if strokes_mask.shape != text_region_mask.shape:
        raise ValueError(f"размеры масок не совпадают: {strokes_mask.shape} и {text_region_mask.shape}")

    skeleton_mask = skeletonize(strokes_mask)
    distance_map = cv2.distanceTransform(strokes_mask.astype(np.uint8) * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    skeleton = Skeleton(skeleton_mask)
    # На macOS numpy использует Accelerate вместо OpenBLAS, и его matmul поднимает
    # ложные RuntimeWarning ("divide by zero", "overflow" и т.п.) на вырожденных
    # входах (ветви-циклы skan с совпадающими src/dst, большие координаты).
    # Результат от этого не портится, поэтому глушим все категории FPE здесь.
    with np.errstate(all="ignore"):
        summary = summarize(skeleton, separator="-")
    source_node = summary["node-id-src"].to_numpy()
    target_node = summary["node-id-dst"].to_numpy()

    return StrokeGraph(
        skeleton=skeleton,
        distance_map=distance_map,
        path_id_image=build_path_id_image(skeleton, skeleton_mask.shape),
        thickness=measure_path_thickness(skeleton, distance_map),
        length=summary["branch-distance"].to_numpy(),
        pixel_count=np.diff(skeleton.paths.indptr),
        branch_type=summary["branch-type"].to_numpy(),
        source_node=source_node,
        target_node=target_node,
        node_degree=np.bincount(np.concatenate([source_node, target_node]), minlength=len(skeleton.coordinates)),
        text_share=measure_path_text_share(skeleton, text_region_mask),
    )


def label_connected_components(graph: StrokeGraph, alive_paths: Mask) -> Ints:
    """Пометить связные компоненты, считая связи только по живым ребрам.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    alive_paths : Mask
        Маска ребер, которые считаются существующими.

    Returns
    -------
    Ints
        Номер компоненты для каждого ребра графа.
    """
    node_count = graph.node_count
    links = coo_matrix(
        (np.ones(int(alive_paths.sum())), (graph.source_node[alive_paths], graph.target_node[alive_paths])),
        shape=(node_count, node_count),
    )
    return connected_components(links, directed=False)[1][graph.source_node]


def find_hanging_paths(graph: StrokeGraph, alive_paths: Mask) -> Mask:
    """Найти ребра со свободным концом среди живых.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    alive_paths : Mask
        Маска ребер, которые считаются существующими.

    Returns
    -------
    Mask
        Маска ребер, у которых хотя бы один узел имеет степень один.

    Notes
    -----
    Степень считается заново по живым ребрам, поэтому после удаления хвоста его корешок становится
    висячим и попадает в результат следующего вызова.
    """
    degree = np.bincount(
        np.concatenate([graph.source_node[alive_paths], graph.target_node[alive_paths]]), minlength=graph.node_count
    )
    free_end = (degree[graph.source_node] == 1) | (degree[graph.target_node] == 1)
    return alive_paths & free_end & (graph.source_node != graph.target_node)


def measure_thickness_spread(graph: StrokeGraph, path_indices: Ints) -> Floats:
    """Измерить неравномерность толщины вдоль оси ребра.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    path_indices : Ints
        Номера ребер, которые надо измерить.

    Returns
    -------
    Floats
        Отношение девяностого процентиля толщины к десятому для каждого ребра графа, nan для ребер
        вне path_indices.

    Notes
    -----
    Наконечник размерной стрелки это клин: его толщина растет от острия к основанию, и отношение
    процентилей у него в разы больше единицы. У линии постоянной ширины отношение близко к единице.
    """
    spread = np.full(graph.path_count, np.nan)
    for path_index in path_indices:
        coordinates = graph.skeleton.path_coordinates(path_index).astype(int)
        widths = 2 * graph.distance_map[coordinates[:, 0], coordinates[:, 1]] - 1
        low, high = np.percentile(widths, [10, 90])
        spread[path_index] = high / max(low, 1e-6)
    return spread
