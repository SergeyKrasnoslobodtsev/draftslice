"""Склейка ребер скелета обратно в цепи.

Скелет рвет линию на ребра в каждом пересечении, поэтому длинный контур, через который проходит
десяток размерных линий, превращается в десяток коротких ребер. Для оценки толщины это плохо: медиана
по короткому ребру шумит, а решение о том, что оставить, принимается по толщине.

Цепь собирает ребра обратно. Два конца, сходящиеся в одном узле, склеиваются, если проходят трое
ворот: направления продолжают друг друга, концы лежат на одной прямой, толщины сопоставимы. Близкие
стыки предварительно схлопываются в один супер-узел, иначе пара соседних пересечений разрывает
склейку на ровном месте. Пары упорядочиваются по прямизне и разбираются жадно, чтобы каждый конец
участвовал ровно один раз.

Самокольца в выборке остаются: иначе теряются буквы О и кружки позиций, которые сами по себе замкнуты
и ни с чем не склеиваются.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from draftslice.common_types import Floats, Ints, Mask
from draftslice.stroke_graph import StrokeGraph


@dataclass(frozen=True)
class MergeGates:
    """Ворота склейки пары концов.

    Attributes
    ----------
    maximum_deviation_degrees : float
        Допустимое отклонение пары от прямой.
    maximum_offset_pixels : float
        Допустимый поперечный сдвиг одного конца относительно прямой другого.
    maximum_thickness_ratio : float
        Допустимое отношение большей толщины к меньшей.
    """

    maximum_deviation_degrees: float = 20.0
    maximum_offset_pixels: float = 7.5
    maximum_thickness_ratio: float = 3.0


@dataclass(frozen=True)
class ChainMergeParameters:
    """Параметры шага склейки.

    Attributes
    ----------
    node_radius_pixels : float
        Длина ребра, ниже которой стык схлопывается в супер-узел.
    tangent_length_pixels : int
        Длина участка оси, по которому оценивается касательная в конце ребра.
    gates : MergeGates
        Ворота склейки.
    """

    node_radius_pixels: float = 3.0
    tangent_length_pixels: int = 12
    gates: MergeGates = MergeGates()


@dataclass(frozen=True, eq=False)
class SuperNodes:
    """Стыки, соединенные коротким ребром, схлопнутые в один узел.

    Attributes
    ----------
    label : Ints
        Метка супер-узла для каждого узла скелета.
    absorbed : Mask
        Ребра, поглощенные супер-узлом.
    """

    label: Ints
    absorbed: Mask


@dataclass(frozen=True, eq=False)
class PathEnds:
    """Концы ребер, участвующих в склейке. Все массивы одной длины.

    Attributes
    ----------
    path : Ints
        Номер ребра, которому принадлежит конец.
    node : Ints
        Метка супер-узла, в котором лежит конец.
    degree : Ints
        Степень исходного узла скелета.
    point : Floats
        Координаты конца в порядке строка, столбец.
    direction : Floats
        Касательная в конце, направленная внутрь ребра.
    thickness : Floats
        Толщина ребра.
    """

    path: Ints
    node: Ints
    degree: Ints
    point: Floats
    direction: Floats
    thickness: Floats

    def __len__(self) -> int:
        return len(self.path)


@dataclass(frozen=True, eq=False)
class StrokeChains:
    """Цепи: ребра, собранные обратно в линии.

    Attributes
    ----------
    graph_path_count : int
        Число ребер исходного графа, нужно для разворота решений в маску ребер.
    path_indices : Ints
        Номера ребер, попавших в цепи.
    chain_label : Ints
        Метка цепи для каждого элемента path_indices.
    length : Floats
        Суммарная длина каждой цепи.
    thickness : Floats
        Толщина каждой цепи.
    text_share : Floats
        Доля пикселей цепи внутри маски текста.
    """

    graph_path_count: int
    path_indices: Ints
    chain_label: Ints
    length: Floats
    thickness: Floats
    text_share: Floats

    @property
    def chain_count(self) -> int:
        """Число цепей."""
        return len(self.length)

    def expand_selection_to_paths(self, kept_chains: Mask) -> Mask:
        """Развернуть решение по цепям в маску ребер скелета.

        Parameters
        ----------
        kept_chains : Mask
            Маска цепей, которые остаются.

        Returns
        -------
        Mask
            Маска ребер графа.
        """
        mask = np.zeros(self.graph_path_count, dtype=bool)
        mask[self.path_indices] = kept_chains[self.chain_label]
        return mask


def build_super_nodes(graph: StrokeGraph, node_radius_pixels: float) -> SuperNodes:
    """Схлопнуть близкие стыки в супер-узлы.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    node_radius_pixels : float
        Длина ребра, ниже которой стык считается частью того же узла.

    Returns
    -------
    SuperNodes
        Метки супер-узлов и поглощенные ребра.
    """
    absorbed = (
        (graph.length < node_radius_pixels)
        & (graph.node_degree[graph.source_node] >= 3)
        & (graph.node_degree[graph.target_node] >= 3)
    )
    node_count = graph.node_count
    links = coo_matrix(
        (np.ones(int(absorbed.sum())), (graph.source_node[absorbed], graph.target_node[absorbed])),
        shape=(node_count, node_count),
    )
    return SuperNodes(label=connected_components(links, directed=False)[1], absorbed=absorbed)


def measure_end_tangents(graph: StrokeGraph, tangent_length_pixels: int) -> Floats:
    """Оценить направление оси в каждом конце каждого ребра.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    tangent_length_pixels : int
        Сколько точек оси участвует в оценке.

    Returns
    -------
    Floats
        Массив формы (число ребер, 2, 2): для каждого ребра два конца, направление внутрь ребра в
        порядке строка, столбец.

    Notes
    -----
    Направление берется главным вектором сингулярного разложения участка оси, поэтому ступеньки
    растра усредняются и не создают ложных изломов.
    """
    tangents = np.zeros((graph.path_count, 2, 2))
    for path_index in range(graph.path_count):
        coordinates = graph.skeleton.path_coordinates(path_index).astype(float)
        sample_size = max(2, min(int(tangent_length_pixels), len(coordinates)))
        for side in (0, 1):
            points = coordinates[:sample_size] if side == 0 else coordinates[::-1][:sample_size]
            direction = np.linalg.svd(points - points.mean(0), full_matrices=False)[2][0]
            tangents[path_index, side] = direction if direction @ (points[-1] - points[0]) > 0 else -direction
    return tangents


def build_path_ends(graph: StrokeGraph, path_indices: Ints, super_nodes: SuperNodes, tangents: Floats) -> PathEnds:
    """Собрать таблицу концов: по два конца на каждое ребро.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    path_indices : Ints
        Ребра, участвующие в склейке.
    super_nodes : SuperNodes
        Метки супер-узлов.
    tangents : Floats
        Касательные в концах ребер.

    Returns
    -------
    PathEnds
        Концы ребер с координатами, направлениями и толщиной.
    """
    path = np.repeat(path_indices, 2)
    side = np.tile(np.array([0, 1]), len(path_indices))
    raw_node = np.where(side == 0, graph.source_node[path], graph.target_node[path])
    point = np.array(
        [
            graph.skeleton.path_coordinates(path_index)[0 if end_side == 0 else -1]
            for path_index, end_side in zip(path, side, strict=True)
        ],
        dtype=float,
    )
    return PathEnds(
        path=path,
        node=super_nodes.label[raw_node],
        degree=graph.node_degree[raw_node],
        point=point,
        direction=tangents[path, side],
        thickness=graph.thickness[path],
    )


def measure_perpendicular_offset(
    first_point: Floats, first_direction: Floats, second_point: Floats, second_direction: Floats
) -> float:
    """Насколько конец не лежит на прямой другого конца.

    Parameters
    ----------
    first_point, second_point : Floats
        Координаты концов.
    first_direction, second_direction : Floats
        Направления в этих концах.

    Returns
    -------
    float
        Больший из двух поперечных сдвигов в пикселях.
    """
    offset_vector = second_point - first_point
    first_offset = abs(offset_vector[0] * first_direction[1] - offset_vector[1] * first_direction[0])
    second_offset = abs(offset_vector[0] * second_direction[1] - offset_vector[1] * second_direction[0])
    return float(max(first_offset, second_offset))


def thickness_is_comparable(first_thickness: float, second_thickness: float, maximum_ratio: float) -> bool:
    """Сопоставимы ли толщины двух ребер.

    Parameters
    ----------
    first_thickness, second_thickness : float
        Толщины ребер.
    maximum_ratio : float
        Допустимое отношение большей толщины к меньшей.

    Returns
    -------
    bool
        True, если толщины сопоставимы или хотя бы одна не определена.
    """
    if not (np.isfinite(first_thickness) and np.isfinite(second_thickness)):
        return True
    larger = max(first_thickness, second_thickness)
    smaller = max(min(first_thickness, second_thickness), 1e-6)
    return larger <= maximum_ratio * smaller


def measure_pair_deviation(
    ends: PathEnds, first_index: int, second_index: int, gates: MergeGates, offset_scale: float = 1.0
) -> float | None:
    """Отклонение пары концов от прямой, если пара проходит ворота.

    Parameters
    ----------
    ends : PathEnds
        Таблица концов.
    first_index, second_index : int
        Номера концов в таблице.
    gates : MergeGates
        Ворота склейки.
    offset_scale : float
        Множитель допуска на поперечный сдвиг.

    Returns
    -------
    float | None
        Отклонение в градусах или None, если ворота не пройдены.
    """
    cosine = -float(ends.direction[first_index] @ ends.direction[second_index])
    deviation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    offset = measure_perpendicular_offset(
        ends.point[first_index],
        ends.direction[first_index],
        ends.point[second_index],
        ends.direction[second_index],
    )
    if deviation > gates.maximum_deviation_degrees or offset > offset_scale * gates.maximum_offset_pixels:
        return None
    if not thickness_is_comparable(
        ends.thickness[first_index], ends.thickness[second_index], gates.maximum_thickness_ratio
    ):
        return None
    return deviation


def group_ends_by_node(ends: PathEnds) -> list[Ints]:
    """Сгруппировать концы по супер-узлу.

    Parameters
    ----------
    ends : PathEnds
        Таблица концов.

    Returns
    -------
    list[Ints]
        Номера концов, сгруппированные по узлу.
    """
    order = np.argsort(ends.node, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(ends.node[order])) + 1)


def find_candidate_pairs(ends: PathEnds, gates: MergeGates) -> list[tuple[int, int]]:
    """Найти пары концов внутри супер-узла, прошедшие ворота.

    Parameters
    ----------
    ends : PathEnds
        Таблица концов.
    gates : MergeGates
        Ворота склейки.

    Returns
    -------
    list[tuple[int, int]]
        Пары концов, самые прямые первыми.
    """
    scored: list[tuple[float, int, int]] = []
    for group in group_ends_by_node(ends):
        for first_index, second_index in combinations(group.tolist(), 2):
            if ends.path[first_index] == ends.path[second_index]:
                continue
            deviation = measure_pair_deviation(ends, first_index, second_index, gates)
            if deviation is not None:
                scored.append((deviation, first_index, second_index))
    scored.sort()
    return [(first_index, second_index) for _, first_index, second_index in scored]


def match_pairs_greedily(pairs: list[tuple[int, int]], end_count: int) -> list[tuple[int, int]]:
    """Разобрать пары жадно, чтобы каждый конец участвовал один раз.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Пары концов в порядке предпочтения.
    end_count : int
        Общее число концов.

    Returns
    -------
    list[tuple[int, int]]
        Выбранные пары.
    """
    taken = np.zeros(end_count, dtype=bool)
    matched: list[tuple[int, int]] = []
    for first_index, second_index in pairs:
        if not (taken[first_index] or taken[second_index]):
            taken[first_index] = taken[second_index] = True
            matched.append((first_index, second_index))
    return matched


def weighted_median(values: Floats, weights: Floats) -> float:
    """Медиана, взвешенная весами.

    Parameters
    ----------
    values : Floats
        Значения, нечисловые пропускаются.
    weights : Floats
        Веса той же длины.

    Returns
    -------
    float
        Взвешенная медиана или nan, если значений нет.
    """
    finite = np.isfinite(values)
    finite_values, finite_weights = values[finite], weights[finite]
    if not len(finite_values):
        return float("nan")

    order = np.argsort(finite_values)
    cumulative = np.cumsum(finite_weights[order])
    return float(finite_values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def label_chains(pairs: list[tuple[int, int]], ends: PathEnds, path_indices: Ints, graph_path_count: int) -> Ints:
    """Превратить склеенные пары в метки цепей.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Выбранные пары концов.
    ends : PathEnds
        Таблица концов.
    path_indices : Ints
        Ребра, участвующие в склейке.
    graph_path_count : int
        Число ребер графа.

    Returns
    -------
    Ints
        Метка цепи для каждого элемента path_indices.
    """
    position = np.full(graph_path_count, -1)
    position[path_indices] = np.arange(len(path_indices))
    rows = [position[ends.path[first_index]] for first_index, _ in pairs]
    cols = [position[ends.path[second_index]] for _, second_index in pairs]
    links = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(path_indices), len(path_indices)))
    return connected_components(links, directed=False)[1]


def measure_chain_thickness(chain_label: Ints, thickness: Floats, length: Floats, chain_count: int) -> Floats:
    """Толщина цепи как медиана толщин ее звеньев, взвешенная длиной.

    Parameters
    ----------
    chain_label : Ints
        Метка цепи для каждого звена.
    thickness : Floats
        Толщина каждого звена.
    length : Floats
        Длина каждого звена.
    chain_count : int
        Число цепей.

    Returns
    -------
    Floats
        Толщина каждой цепи.
    """
    thicknesses = np.full(chain_count, np.nan)
    for chain_index in range(chain_count):
        members = chain_label == chain_index
        thicknesses[chain_index] = weighted_median(thickness[members], length[members])
    return thicknesses


def measure_chain_text_share(chain_label: Ints, text_share: Floats, pixel_count: Ints, chain_count: int) -> Floats:
    """Доля пикселей цепи внутри маски текста.

    Parameters
    ----------
    chain_label : Ints
        Метка цепи для каждого звена.
    text_share : Floats
        Доля текста у каждого звена.
    pixel_count : Ints
        Число пикселей оси каждого звена.
    chain_count : int
        Число цепей.

    Returns
    -------
    Floats
        Доля текста для каждой цепи.

    Notes
    -----
    Порог применяется один раз и уже к цепи: короткое ребро, целиком попавшее под надпись, не должно
    объявлять текстом всю линию.
    """
    inside = np.bincount(chain_label, weights=text_share * pixel_count, minlength=chain_count)
    total = np.bincount(chain_label, weights=pixel_count.astype(np.float64), minlength=chain_count)
    return inside / np.maximum(total, 1.0)


def build_stroke_chains(graph: StrokeGraph, parameters: ChainMergeParameters) -> StrokeChains:
    """Склеить ребра скелета в цепи.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    parameters : ChainMergeParameters
        Параметры склейки.

    Returns
    -------
    StrokeChains
        Цепи с длиной, толщиной и долей текста.
    """
    super_nodes = build_super_nodes(graph, parameters.node_radius_pixels)
    path_indices = np.flatnonzero(~super_nodes.absorbed)
    ends = build_path_ends(
        graph, path_indices, super_nodes, measure_end_tangents(graph, parameters.tangent_length_pixels)
    )
    matched = match_pairs_greedily(find_candidate_pairs(ends, parameters.gates), len(ends))
    chain_label = label_chains(matched, ends, path_indices, graph.path_count)
    chain_count = int(chain_label.max()) + 1

    return StrokeChains(
        graph_path_count=graph.path_count,
        path_indices=path_indices,
        chain_label=chain_label,
        length=np.bincount(chain_label, weights=graph.length[path_indices], minlength=chain_count),
        thickness=measure_chain_thickness(
            chain_label, graph.thickness[path_indices], graph.length[path_indices], chain_count
        ),
        text_share=measure_chain_text_share(
            chain_label, graph.text_share[path_indices], graph.pixel_count[path_indices], chain_count
        ),
    )
