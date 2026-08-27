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

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd

from draftslice.chain_merging import StrokeChains
from draftslice.common_types import Floats, Ints, Mask
from draftslice.features import ComponentFeatures
from draftslice.morphology_closing import fill_internal_holes
from draftslice.pipeline import PipelineArtifacts
from draftslice.stroke_graph import (
    CYCLE_BRANCH,
    ISOLATED_BRANCH,
    JUNCTION_BRANCH,
    SPUR_BRANCH,
    StrokeGraph,
)
from draftslice.thickness_classes import ThicknessClasses


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


@dataclass(frozen=True)
class ChainMetrics:
    """Метрики склейки ребер в цепи.

    Attributes
    ----------
    chain_count : int
        Число цепей.
    linked_path_count : int
        Число ребер, попавших в цепи.
    links_per_chain : float
        Среднее число звеньев в цепи.
    median_length : float
        Медианная длина цепи.
    maximum_length : float
        Длина самой длинной цепи.
    text_chain_count : int
        Число цепей, у которых доля текста выше порога.
    undefined_thickness_count : int
        Число цепей с неопределенной толщиной.
    """

    chain_count: int
    linked_path_count: int
    links_per_chain: float
    median_length: float
    maximum_length: float
    text_chain_count: int
    undefined_thickness_count: int


@dataclass(frozen=True)
class ThicknessClassMetrics:
    """Метрики разбиения цепей на классы толщины.

    Attributes
    ----------
    centers : Floats
        Центры классов в пикселях.
    edges : Floats
        Границы между соседними классами.
    chain_counts : Ints
        Число цепей в каждом классе.
    length_shares : Floats
        Доля суммарной длины, приходящаяся на каждый класс.
    unclassified_count : int
        Число цепей без класса: текст и неопределенная толщина.
    """

    centers: Floats
    edges: Floats
    chain_counts: Ints
    length_shares: Floats
    unclassified_count: int


def measure_chain_metrics(chains: StrokeChains, text_share_limit: float = 0.5) -> ChainMetrics:
    """Снять метрики склейки.

    Parameters
    ----------
    chains : StrokeChains
        Цепи кадра.
    text_share_limit : float
        Доля текста, выше которой цепь считается текстовой.

    Returns
    -------
    ChainMetrics
        Размер цепей и состав выборки.

    Notes
    -----
    Среднее число звеньев в цепи показывает, работает ли склейка вообще: единица означает, что ни
    одна пара концов не прошла ворота.
    """
    return ChainMetrics(
        chain_count=chains.chain_count,
        linked_path_count=len(chains.path_indices),
        links_per_chain=len(chains.path_indices) / max(chains.chain_count, 1),
        median_length=float(np.median(chains.length)) if chains.chain_count else 0.0,
        maximum_length=float(chains.length.max()) if chains.chain_count else 0.0,
        text_chain_count=int((chains.text_share > text_share_limit).sum()),
        undefined_thickness_count=int((~np.isfinite(chains.thickness)).sum()),
    )


def measure_thickness_class_metrics(chains: StrokeChains, classes: ThicknessClasses) -> ThicknessClassMetrics:
    """Снять метрики разбиения на классы толщины.

    Parameters
    ----------
    chains : StrokeChains
        Цепи кадра.
    classes : ThicknessClasses
        Классы толщины.

    Returns
    -------
    ThicknessClassMetrics
        Центры, границы, наполнение классов и доля длины в каждом.
    """
    classified = classes.label >= 0
    total_length = float(chains.length[classified].sum())
    chain_counts = np.array(
        [int((classes.label == class_index).sum()) for class_index in range(classes.class_count)], dtype=np.int64
    )
    length_shares = np.array(
        [
            float(chains.length[classes.label == class_index].sum()) / max(total_length, 1e-6)
            for class_index in range(classes.class_count)
        ]
    )
    return ThicknessClassMetrics(
        centers=classes.centers,
        edges=classes.edges,
        chain_counts=chain_counts,
        length_shares=length_shares,
        unclassified_count=int((~classified).sum()),
    )


def build_component_feature_table(features: ComponentFeatures) -> pd.DataFrame:
    """Собрать таблицу признаков компонент для печати.

    Parameters
    ----------
    features : ComponentFeatures
        Признаки компонент.

    Returns
    -------
    pd.DataFrame
        Таблица, отсортированная по убыванию длины.

    Notes
    -----
    DataFrame используется только как способ печати: расчеты идут по массивам, а здесь собирается
    представление для глаз.
    """
    table = pd.DataFrame(
        {
            "id": features.component_id,
            "ребер": features.path_count,
            "длина": features.axis_length.round(0),
            "габарит": features.bounding_diagonal.round(0),
            "извил": features.tortuosity.round(2),
            "толщина": features.median_thickness.round(1),
            "разброс t": features.thickness_spread.round(2),
            "внутри": features.inside_area,
            "текст": features.inside_text_share.round(2),
            "хвостов": features.twig_count,
            "доля хвостов": features.twig_length_share.round(2),
            "длиннейший хвост": features.longest_twig_length.round(0),
        }
    )
    return table.sort_values("длина", ascending=False)


@dataclass(frozen=True)
class PipelineRow:
    """Сводка одного прогона для таблицы по датасету.

    Attributes
    ----------
    name : str
        Имя файла без расширения.
    source_long_side : int
        Длинная сторона исходника в пикселях.
    text_share : float
        Доля кадра под маской текста.
    strokes_ink_share : float
        Доля чернил после бинаризации.
    strokes_spur_ratio : float
        Доля шпор среди ребер первого графа, признак разорванного контура.
    class_edge : float
        Первая граница классов толщины в пикселях.
    clean_ink_share : float
        Доля чернил, оставшаяся после отбора по толщине.
    closing_kernel_length : int
        Длина ядра замыкания.
    closed_piece_drop : int
        На сколько замыкание уменьшило число кусков маски.
    closed_hole_growth : int
        На сколько замыкание увеличило число замкнутых пустот.
    closed_spur_ratio : float
        Доля шпор среди ребер второго графа.
    component_count : int
        Число компонент замкнутого графа.
    kept_component_count : int
        Сколько компонент прошло правила.
    part_ink_share : float
        Доля чернил в итоговой маске от бинаризации.
    part_piece_count : int
        Число кусков итоговой маски.
    largest_component_length : float
        Длина осей самой длинной компоненты.
    """

    name: str
    source_long_side: int
    text_share: float
    strokes_ink_share: float
    strokes_spur_ratio: float
    class_edge: float
    clean_ink_share: float
    closing_kernel_length: int
    closed_piece_drop: int
    closed_hole_growth: int
    closed_spur_ratio: float
    component_count: int
    kept_component_count: int
    part_ink_share: float
    part_piece_count: int
    largest_component_length: float


def spur_ratio(metrics: GraphMetrics) -> float:
    """Доля шпор среди ребер графа.

    Parameters
    ----------
    metrics : GraphMetrics
        Метрики графа.

    Returns
    -------
    float
        Отношение числа шпор к числу ребер.

    Notes
    -----
    Чем выше доля, тем сильнее разорван контур: куски разорванной линии получают свободные концы и
    попадают в шпоры вместо ребер узел-узел.
    """
    return metrics.spur_count / max(metrics.path_count, 1)


def measure_pipeline_row(name: str, source_long_side: int, artifacts: PipelineArtifacts) -> PipelineRow:
    """Свести один прогон в строку таблицы.

    Parameters
    ----------
    name : str
        Имя файла без расширения.
    source_long_side : int
        Длинная сторона исходника в пикселях.
    artifacts : PipelineArtifacts
        Результаты прогона.

    Returns
    -------
    PipelineRow
        Строка сводной таблицы.
    """
    clean_metrics = measure_mask_metrics(artifacts.clean_mask)
    closed_metrics = measure_mask_metrics(artifacts.closed_mask)
    part_metrics = measure_mask_metrics(artifacts.part_mask)
    strokes_graph_metrics = measure_graph_metrics(artifacts.strokes_graph)
    closed_graph_metrics = measure_graph_metrics(artifacts.closed_graph)

    kept_components = np.unique(artifacts.features.path_component[artifacts.final_paths])
    return PipelineRow(
        name=name,
        source_long_side=source_long_side,
        text_share=float(artifacts.text_region_mask.mean()),
        strokes_ink_share=float(artifacts.strokes_mask.mean()),
        strokes_spur_ratio=spur_ratio(strokes_graph_metrics),
        class_edge=float(artifacts.thickness_classes.edges[0])
        if len(artifacts.thickness_classes.edges)
        else float("nan"),
        clean_ink_share=float(artifacts.clean_mask.sum() / max(artifacts.strokes_mask.sum(), 1)),
        closing_kernel_length=artifacts.closing_kernel_length,
        closed_piece_drop=clean_metrics.component_count - closed_metrics.component_count,
        closed_hole_growth=closed_metrics.hole_count - clean_metrics.hole_count,
        closed_spur_ratio=spur_ratio(closed_graph_metrics),
        component_count=len(artifacts.features),
        kept_component_count=len(kept_components),
        part_ink_share=float(artifacts.part_mask.sum() / max(artifacts.strokes_mask.sum(), 1)),
        part_piece_count=part_metrics.component_count,
        largest_component_length=float(artifacts.features.axis_length.max()) if len(artifacts.features) else 0.0,
    )


def build_dataset_table(rows: list[PipelineRow]) -> pd.DataFrame:
    """Собрать таблицу прогонов по датасету.

    Parameters
    ----------
    rows : list[PipelineRow]
        Строки прогонов.

    Returns
    -------
    pd.DataFrame
        Таблица с читаемыми заголовками.
    """
    table = pd.DataFrame([asdict(row) for row in rows])
    return table.rename(
        columns={
            "name": "файл",
            "source_long_side": "сторона",
            "text_share": "текст",
            "strokes_ink_share": "чернил",
            "strokes_spur_ratio": "шпор до",
            "class_edge": "граница t",
            "clean_ink_share": "после толщины",
            "closing_kernel_length": "ядро",
            "closed_piece_drop": "кусков ушло",
            "closed_hole_growth": "пустот выросло",
            "closed_spur_ratio": "шпор после",
            "component_count": "компонент",
            "kept_component_count": "оставлено",
            "part_ink_share": "итог чернил",
            "part_piece_count": "итог кусков",
            "largest_component_length": "длиннейшая",
        }
    )
