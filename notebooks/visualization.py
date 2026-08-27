"""Отрисовка этапов пайплайна для блокнотов.

Модуль намеренно живет рядом с блокнотами, а не в пакете: библиотека остается свободной от
matplotlib, а картинки нужны только тому, кто подбирает параметры глазами. Все функции здесь только
рисуют и ничего не вычисляют, поэтому их можно менять, не трогая алгоритм.
"""

import cv2
import numpy as np
from ipywidgets import FloatSlider
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from draftslice.common_types import Floats, Ints, Mask, Mat, Window
from draftslice.stroke_graph import (
    CYCLE_BRANCH,
    ISOLATED_BRANCH,
    JUNCTION_BRANCH,
    SPUR_BRANCH,
    StrokeGraph,
)
from draftslice.text_detection import TextCandidates

ACCEPTED_COLOR = (0, 200, 0)
"""Цвет принятого кандидата в BGR."""

REJECTED_COLOR = (0, 0, 255)
"""Цвет отклоненного кандидата в BGR."""


def show_bgr_image(image: Mat, title: str = "", figsize: tuple[float, float] = (16, 8)) -> None:
    """Показать кадр в порядке каналов BGR.

    Parameters
    ----------
    image : Mat
        Кадр.
    title : str
        Заголовок.
    figsize : tuple[float, float]
        Размер фигуры в дюймах.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image[..., ::-1])
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_mask(mask: Mask, title: str = "", figsize: tuple[float, float] = (16, 8)) -> None:
    """Показать бинарную маску, штрих черным на белом.

    Parameters
    ----------
    mask : Mask
        Маска.
    title : str
        Заголовок.
    figsize : tuple[float, float]
        Размер фигуры в дюймах.
    """
    plt.figure(figsize=figsize)
    plt.imshow(np.where(mask, 0, 255).astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_probability_maps(probability_maps: Floats, long_sides: tuple[int, ...]) -> None:
    """Показать карты вероятностей текста, снятые на разных масштабах.

    Parameters
    ----------
    probability_maps : Floats
        Стек карт формы (число масштабов, высота, ширина).
    long_sides : tuple[int, ...]
        Размеры длинной стороны, которым соответствуют карты.
    """
    figure, axes = plt.subplots(1, len(long_sides), figsize=(5 * len(long_sides), 5))
    for axis, probability_map, long_side in zip(np.atleast_1d(axes), probability_maps, long_sides, strict=True):
        axis.imshow(probability_map, cmap="inferno", vmin=0, vmax=1)
        axis.set_title(f"масштаб {long_side}")
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def show_probability_summary(probability_map: Floats) -> None:
    """Показать усредненную карту вероятностей и распределение ее значений.

    Parameters
    ----------
    probability_map : Floats
        Усредненная карта вероятностей.
    """
    figure, (map_axis, histogram_axis) = plt.subplots(1, 2, figsize=(20, 6))
    image = map_axis.imshow(probability_map, cmap="inferno", vmin=0, vmax=1)
    map_axis.set_title("карта вероятностей текста")
    map_axis.axis("off")
    figure.colorbar(image, ax=map_axis, fraction=0.03)
    histogram_axis.hist(probability_map.ravel(), bins=100, log=True)
    histogram_axis.set_title("распределение вероятностей, ось y логарифмическая")
    figure.tight_layout()
    plt.show()


def draw_text_candidates(image: Mat, candidates: TextCandidates, accepted: Mask, title: str) -> None:
    """Показать кандидатов в текстовые ядра: принятые зеленым, отклоненные красным.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.
    candidates : TextCandidates
        Кандидаты.
    accepted : Mask
        Маска принятых кандидатов.
    title : str
        Заголовок.
    """
    canvas = image.copy()
    for contour, is_accepted in zip(candidates.contours, accepted, strict=True):
        cv2.drawContours(canvas, [contour], -1, ACCEPTED_COLOR if is_accepted else REJECTED_COLOR, 2)
    show_bgr_image(canvas, title, figsize=(18, 10))


def draw_text_region(image: Mat, text_region_mask: Mask, title: str) -> None:
    """Показать текстовую область полупрозрачной заливкой поверх кадра.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.
    text_region_mask : Mask
        Маска текстовой области.
    title : str
        Заголовок.
    """
    overlay = image.copy()
    overlay[text_region_mask] = REJECTED_COLOR
    show_bgr_image(cv2.addWeighted(overlay, 0.45, image, 0.55, 0), title, figsize=(18, 10))


BRANCH_NAMES: dict[int, str] = {
    ISOLATED_BRANCH: "конец-конец (изолят)",
    SPUR_BRANCH: "узел-конец (шпора)",
    JUNCTION_BRANCH: "узел-узел",
    CYCLE_BRANCH: "цикл",
}
"""Читаемые названия типов ребер."""

BRANCH_COLORS: dict[int, str] = {
    ISOLATED_BRANCH: "tab:blue",
    SPUR_BRANCH: "tab:green",
    JUNCTION_BRANCH: "orange",
    CYCLE_BRANCH: "tab:red",
}
"""Цвета типов ребер на карте графа."""

NODE_KINDS: tuple[tuple[int, int, str, str, str], ...] = (
    (1, 1, "red", "o", "конец"),
    (2, 2, "yellow", "^", "узел степени 2"),
    (3, 99, "cyan", "s", "стык"),
)
"""Разметка узлов по степени: диапазон, цвет, маркер, подпись."""


def build_view_window(image_shape: tuple[int, int], center_x: float, center_y: float, zoom: float) -> Window:
    """Собрать окно просмотра по относительному центру и увеличению.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Размер кадра в порядке высота, ширина.
    center_x, center_y : float
        Центр окна в долях ширины и высоты.
    zoom : float
        Во сколько раз окно меньше кадра.

    Returns
    -------
    Window
        Границы окна в пикселях.
    """
    height, width = image_shape
    half_side = max(height, width) / (2 * zoom)
    return (
        center_x * width - half_side,
        center_x * width + half_side,
        center_y * height - half_side,
        center_y * height + half_side,
    )


def window_sliders() -> dict[str, FloatSlider]:
    """Одинаковый набор слайдеров окна для любого просмотрщика.

    Returns
    -------
    dict[str, FloatSlider]
        Слайдеры центра и увеличения для передачи в interactive.
    """
    return {
        "center_x": FloatSlider(value=0.5, min=0.0, max=1.0, step=0.02, continuous_update=False, description="центр X"),
        "center_y": FloatSlider(value=0.5, min=0.0, max=1.0, step=0.02, continuous_update=False, description="центр Y"),
        "zoom": FloatSlider(value=1.0, min=1.0, max=12.0, step=0.5, continuous_update=False, description="зум"),
    }


def create_view_axes(
    background_image: Mat, window: Window, title: str, figsize: tuple[float, float] = (18, 11)
) -> Axes:
    """Создать оси с подложкой и выставленным окном просмотра.

    Parameters
    ----------
    background_image : Mat
        Кадр, который показывается под графом.
    window : Window
        Границы окна в пикселях.
    title : str
        Заголовок.
    figsize : tuple[float, float]
        Размер фигуры в дюймах.

    Returns
    -------
    Axes
        Оси, готовые для отрисовки ребер.
    """
    left, right, top, bottom = window
    _, axes = plt.subplots(figsize=figsize)
    axes.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY), cmap="gray", alpha=0.4)
    axes.set_xlim(left, right)
    axes.set_ylim(bottom, top)
    axes.axis("off")
    axes.set_title(title)
    return axes


def draw_graph_paths(
    axes: Axes,
    graph: StrokeGraph,
    path_indices: Ints,
    colors,
    line_width: float = 1.8,
    z_order: int = 2,
    label: str | None = None,
) -> None:
    """Нарисовать выбранные ребра скелета одной коллекцией линий.

    Parameters
    ----------
    axes : Axes
        Оси для отрисовки.
    graph : StrokeGraph
        Граф скелета.
    path_indices : Ints
        Номера ребер.
    colors : Any
        Цвет или массив цветов, как их понимает LineCollection.
    line_width : float
        Толщина линии.
    z_order : int
        Порядок отрисовки.
    label : str | None
        Подпись для легенды.
    """
    axes.add_collection(
        LineCollection(
            [graph.path_points(int(index)) for index in path_indices],
            colors=colors,
            linewidths=line_width,
            zorder=z_order,
            label=label,
        )
    )


def draw_branch_types(graph: StrokeGraph, background_image: Mat, window: Window, title: str) -> None:
    """Показать граф, раскрасив ребра по типу ветвления.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.
    """
    axes = create_view_axes(background_image, window, title)
    for branch_type, name in BRANCH_NAMES.items():
        selected = np.flatnonzero(graph.branch_type == branch_type)
        draw_graph_paths(axes, graph, selected, BRANCH_COLORS[branch_type], z_order=3, label=f"{name}: {len(selected)}")
    axes.legend(loc="upper right", fontsize=8)
    plt.show()


def draw_graph_nodes(axes: Axes, graph: StrokeGraph, window: Window) -> None:
    """Нанести узлы графа, различая их по степени.

    Parameters
    ----------
    axes : Axes
        Оси для отрисовки.
    graph : StrokeGraph
        Граф скелета.
    window : Window
        Границы окна просмотра.
    """
    left, right, top, bottom = window
    rows = graph.skeleton.coordinates[:, 0]
    cols = graph.skeleton.coordinates[:, 1]
    visible = (graph.node_degree > 0) & (cols >= left) & (cols <= right) & (rows >= top) & (rows <= bottom)

    for low, high, color, marker, name in NODE_KINDS:
        selected = np.flatnonzero(visible & (graph.node_degree >= low) & (graph.node_degree <= high))
        axes.scatter(
            cols[selected],
            rows[selected],
            s=26,
            c=color,
            marker=marker,
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
            label=f"{name}: {len(selected)}",
        )
