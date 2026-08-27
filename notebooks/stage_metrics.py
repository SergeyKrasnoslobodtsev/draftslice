"""Метрики этапов пайплайна.

Метрики нужны для двух разных задач. Первая это подбор параметров на одном кадре: видно, что
меняется, когда двигаешь порог. Вторая это прогон по датасету, когда глазами смотреть уже нельзя и
о качестве судят по таблице чисел.

Поэтому функции здесь только считают и возвращают датаклассы, а печать и форматирование остаются на
стороне блокнота. Метрики подобраны так, чтобы ловить известные нам виды поломок: доля чернил и
число кусков показывают, не заплыла ли маска после замыкания, число свободных концов показывает,
насколько разорван контур, а число замкнутых пустот растет скачком, когда замыкание начинает сшивать
соседние параллельные линии.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from draftslice.common_types import Mask
from draftslice.morphology_closing import fill_internal_holes
from draftslice.stroke_graph import (
    CYCLE_BRANCH,
    ISOLATED_BRANCH,
    JUNCTION_BRANCH,
    SPUR_BRANCH,
    StrokeGraph,
)


@dataclass(frozen=True)
class MaskMetrics:
    """Метрики бинарной маски.

    Attributes
    ----------
    ink_share : float
        Доля пикселей кадра, занятых чернилами.
    component_count : int
        Число связных кусков маски.
    hole_count : int
        Число замкнутых пустот внутри кусков.
    """

    ink_share: float
    component_count: int
    hole_count: int


@dataclass(frozen=True)
class GraphMetrics:
    """Метрики графа скелета.

    Attributes
    ----------
    path_count : int
        Число ребер.
    isolated_count, spur_count, junction_count, cycle_count : int
        Число ребер каждого типа ветвления.
    free_end_count : int
        Число узлов со степенью один, то есть свободных концов.
    branching_node_count : int
        Число узлов со степенью три и выше.
    median_thickness : float
        Медиана толщины ребер в пикселях.
    total_length : float
        Суммарная длина осей.
    text_path_count : int
        Число ребер, у которых доля текста выше порога.
    """

    path_count: int
    isolated_count: int
    spur_count: int
    junction_count: int
    cycle_count: int
    free_end_count: int
    branching_node_count: int
    median_thickness: float
    total_length: float
    text_path_count: int


def measure_mask_metrics(mask: Mask) -> MaskMetrics:
    """Снять метрики бинарной маски.

    Parameters
    ----------
    mask : Mask
        Бинарная маска штрихов.

    Returns
    -------
    MaskMetrics
        Доля чернил, число кусков и число замкнутых пустот.

    Notes
    -----
    Число пустот это индикатор слипания: когда замыкание начинает сшивать соседние параллельные
    линии, каждая ячейка между ними становится пустотой, и счетчик растет скачком.
    """
    component_count = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]) - 1
    holes = fill_internal_holes(mask) & ~mask
    hole_count = int(cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)[0]) - 1
    return MaskMetrics(ink_share=float(mask.mean()), component_count=component_count, hole_count=hole_count)


def measure_graph_metrics(graph: StrokeGraph, text_share_limit: float = 0.5) -> GraphMetrics:
    """Снять метрики графа скелета.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    text_share_limit : float
        Доля пикселей ребра в маске текста, выше которой ребро считается текстовым.

    Returns
    -------
    GraphMetrics
        Состав ребер по типам, статистика узлов и толщины.

    Notes
    -----
    Отношение числа шпор к числу ребер узел-узел показывает, насколько связен контур: когда контур
    разорван, его куски получают свободные концы и попадают в шпоры.
    """
    degree = graph.node_degree[graph.node_degree > 0]
    return GraphMetrics(
        path_count=graph.path_count,
        isolated_count=int((graph.branch_type == ISOLATED_BRANCH).sum()),
        spur_count=int((graph.branch_type == SPUR_BRANCH).sum()),
        junction_count=int((graph.branch_type == JUNCTION_BRANCH).sum()),
        cycle_count=int((graph.branch_type == CYCLE_BRANCH).sum()),
        free_end_count=int((degree == 1).sum()),
        branching_node_count=int((degree >= 3).sum()),
        median_thickness=graph.median_thickness,
        total_length=float(graph.length.sum()),
        text_path_count=int((graph.text_share > text_share_limit).sum()),
    )
