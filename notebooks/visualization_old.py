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

from draftslice.chain_merging import StrokeChains
from draftslice.common_types import Floats, IntMap, Ints, Mask, Mat, Window
from draftslice.features import ComponentFeatures
from draftslice.stroke_graph import (
    CYCLE_BRANCH,
    ISOLATED_BRANCH,
    JUNCTION_BRANCH,
    SPUR_BRANCH,
    StrokeGraph,
)
from draftslice.text_detection import TextCandidates
from draftslice.thickness_classes import ThicknessClasses

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


CLASS_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.27, 0.51, 1.00),
    (0.24, 0.78, 0.35),
    (1.00, 0.75, 0.16),
    (1.00, 0.27, 0.27),
    (0.78, 0.31, 1.00),
    (0.00, 0.86, 0.86),
)
"""Цвета классов толщины по возрастанию класса."""

UNCLASSIFIED_COLOR = (0.55, 0.55, 0.55)
"""Цвет цепи без класса: текст или неопределенная толщина."""


def draw_chains(graph: StrokeGraph, chains: StrokeChains, background_image: Mat, window: Window, title: str) -> None:
    """Показать цепи, один цвет это одна цепь.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    chains : StrokeChains
        Цепи кадра.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.

    Notes
    -----
    Цвета перемешаны, поэтому соседние цепи различимы, а смена цвета вдоль линии означает, что
    склейка там не сработала.
    """
    axes = create_view_axes(background_image, window, title)
    colors = plt.cm.hsv(np.random.default_rng(1).permutation(chains.chain_count) / max(chains.chain_count - 1, 1))
    draw_graph_paths(axes, graph, chains.path_indices, colors[chains.chain_label], line_width=2.0, z_order=3)
    plt.show()


def draw_thickness_classes(
    graph: StrokeGraph,
    chains: StrokeChains,
    classes: ThicknessClasses,
    background_image: Mat,
    window: Window,
    title: str,
) -> None:
    """Показать цепи, раскрасив их по классу толщины.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    chains : StrokeChains
        Цепи кадра.
    classes : ThicknessClasses
        Классы толщины.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.
    """
    palette = np.array(CLASS_PALETTE[: classes.class_count], dtype=float)
    chain_colors = np.where(
        (classes.label < 0)[chains.chain_label][:, None],
        np.array(UNCLASSIFIED_COLOR),
        palette[classes.label.clip(0)[chains.chain_label]],
    )
    axes = create_view_axes(background_image, window, title)
    draw_graph_paths(axes, graph, chains.path_indices, chain_colors, line_width=1.6, z_order=3)
    plt.show()


GROUP_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.10, 0.29),
    (0.24, 0.78, 0.35),
    (1.00, 0.75, 0.16),
    (0.00, 0.51, 0.78),
    (0.96, 0.51, 0.19),
    (0.57, 0.12, 0.71),
    (0.27, 0.94, 0.94),
    (0.94, 0.20, 0.90),
    (0.82, 0.96, 0.24),
    (0.00, 0.50, 0.50),
)
"""Цвета групп кусков part_mask, циклически повторяются, если групп больше палитры."""


def draw_view_groups(
    labels: IntMap, group_count: int, raw_component_count: int, background_image: Mat, window: Window, title: str
) -> None:
    """Показать группы кусков part_mask после слияния разрывов, каждую своим цветом.

    Parameters
    ----------
    labels : IntMap
        Номер группы на пиксель, 0 вне маски.
    group_count : int
        Число групп после слияния.
    raw_component_count : int
        Число кусков растра до слияния.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.

    Notes
    -----
    Разные виды должны получить заметно разные цвета, а разрыв внутри одного вида, слитый мержем,
    должен оказаться одного цвета по обе стороны от разрыва: по этому и проверяется, не задран ли
    порог разрыва и не слиплись ли два разных вида в один.
    """
    palette = np.array(GROUP_PALETTE, dtype=float)
    overlay = np.zeros((*labels.shape, 4))
    in_mask = labels > 0
    overlay[in_mask, :3] = palette[(labels[in_mask] - 1) % len(GROUP_PALETTE)]
    overlay[in_mask, 3] = 1.0

    axes = create_view_axes(
        background_image, window, f"{title} | кусков было {raw_component_count}, групп стало {group_count}"
    )
    axes.imshow(overlay)
    plt.show()


ADDED_PIXEL_COLOR = (0, 0, 255)
"""Цвет пикселей, добавленных замыканием, в BGR."""


def draw_closing_result(before_mask: Mask, after_mask: Mask, title: str) -> None:
    """Показать, что добавило замыкание, и результат рядом.

    Parameters
    ----------
    before_mask : Mask
        Маска до замыкания.
    after_mask : Mask
        Маска после замыкания.
    title : str
        Заголовок правой панели.
    """
    canvas = np.full((*before_mask.shape, 3), 255, np.uint8)
    canvas[before_mask] = (0, 0, 0)
    canvas[after_mask & ~before_mask] = ADDED_PIXEL_COLOR

    _, (left_axes, right_axes) = plt.subplots(1, 2, figsize=(20, 8))
    left_axes.imshow(canvas[..., ::-1])
    left_axes.set_title("красное это добавленные пиксели")
    right_axes.imshow(np.where(after_mask, 0, 255).astype(np.uint8), cmap="gray")
    right_axes.set_title(title)
    for axes in (left_axes, right_axes):
        axes.axis("off")
    plt.show()


def draw_component_selection(
    graph: StrokeGraph,
    features: ComponentFeatures,
    kept_paths: Mask,
    background_image: Mat,
    window: Window,
    title: str,
    label_length_limit: float = 0.0,
) -> None:
    """Показать решение по компонентам: зеленое остается, красное уходит.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    features : ComponentFeatures
        Признаки компонент.
    kept_paths : Mask
        Маска сохраненных ребер.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.
    label_length_limit : float
        Подписывать номера компонент, чья длина не меньше этого значения. Ноль отключает подписи.
    """
    axes = create_view_axes(background_image, window, title)
    draw_graph_paths(
        axes, graph, np.flatnonzero(kept_paths), "green", z_order=3, label=f"остается: {int(kept_paths.sum())}"
    )
    draw_graph_paths(
        axes, graph, np.flatnonzero(~kept_paths), "red", z_order=4, label=f"уходит: {int((~kept_paths).sum())}"
    )

    if label_length_limit > 0:
        for component_id, axis_length in zip(features.component_id, features.axis_length, strict=True):
            if axis_length < label_length_limit:
                continue
            member_paths = np.flatnonzero(features.path_component == component_id)
            first_point = graph.path_points(int(member_paths[0]))[0]
            axes.text(first_point[0], first_point[1], str(int(component_id)), fontsize=9, color="black")
    axes.legend(loc="upper right", fontsize=8)
    plt.show()


def draw_trimmed_twigs(
    graph: StrokeGraph,
    dropped_paths: Mask,
    trimmed_paths: Mask,
    kept_paths: Mask,
    background_image: Mat,
    window: Window,
    title: str,
) -> None:
    """Показать итог отбора: что осталось, что убрано правилом и что срезано хвостами.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета.
    dropped_paths : Mask
        Ребра, убранные правилом по компонентам.
    trimmed_paths : Mask
        Ребра, срезанные как хвосты.
    kept_paths : Mask
        Ребра, оставшиеся в итоге.
    background_image : Mat
        Кадр-подложка.
    window : Window
        Границы окна просмотра.
    title : str
        Заголовок.
    """
    axes = create_view_axes(background_image, window, title)
    layers = (
        (kept_paths, "green", 3, "остается"),
        (dropped_paths, "red", 4, "убрано правилом"),
        (trimmed_paths, "orange", 5, "срезано хвостами"),
    )
    for mask, color, z_order, name in layers:
        selected = np.flatnonzero(mask)
        draw_graph_paths(axes, graph, selected, color, z_order=z_order, label=f"{name}: {len(selected)}")
    axes.legend(loc="upper right", fontsize=8)
    plt.show()


def show_before_after_list(
    before_masks: list[Mask], after_masks: list[Mask], titles: list[str], row_height: float = 3.4
) -> None:
    """Показать пары было и стало, по строке на файл.

    Parameters
    ----------
    before_masks : list[Mask]
        Маски до очистки.
    after_masks : list[Mask]
        Маски после очистки, того же порядка.
    titles : list[str]
        Подписи строк.
    row_height : float
        Высота одной строки в дюймах.

    Notes
    -----
    Маска результата сама по себе выглядит правдоподобно даже тогда, когда из детали вырезан кусок.
    Сравнение с исходными штрихами показывает, что именно удалено.
    """
    figure, axes_grid = plt.subplots(len(titles), 2, figsize=(13, len(titles) * row_height))
    grid = np.atleast_2d(axes_grid)
    for row_index, (before_mask, after_mask, title) in enumerate(zip(before_masks, after_masks, titles, strict=True)):
        for column_index, (mask, name) in enumerate(((before_mask, "было"), (after_mask, "стало"))):
            axes = grid[row_index, column_index]
            axes.imshow(np.where(mask, 0, 255).astype(np.uint8), cmap="gray")
            axes.set_title(f"{title}, {name}", fontsize=9)
            axes.axis("off")
    figure.tight_layout()
    plt.show()


def show_view_crops(crops: list[Mask], titles: list[str], row_height: float = 3.0) -> None:
    """Показать кропы видов по одному в строке.

    Parameters
    ----------
    crops : list[Mask]
        Кропы видов, вырезанные из part_mask.
    titles : list[str]
        Подписи, тот же порядок, что и `crops`.
    row_height : float
        Высота одной строки в дюймах.
    """
    figure, axes_list = plt.subplots(len(crops), 1, figsize=(13, len(crops) * row_height))
    for axes, crop, title in zip(np.atleast_1d(axes_list), crops, titles, strict=True):
        axes.imshow(np.where(crop, 0, 255).astype(np.uint8), cmap="gray")
        axes.set_title(f"{title} | {crop.shape[1]}x{crop.shape[0]}", fontsize=9)
        axes.axis("off")
    figure.tight_layout()
    plt.show()


def shrink_mask_for_preview(mask: Mask, long_side: int = 900) -> Mask:
    """Уменьшить маску до размера, пригодного для галереи.

    Parameters
    ----------
    mask : Mask
        Исходная маска.
    long_side : int
        Требуемый размер длинной стороны в пикселях.

    Returns
    -------
    Mask
        Уменьшенная маска.

    Notes
    -----
    Нужна для прогона по датасету: держать в памяти полноразмерные маски всех файлов нельзя, кадр в
    двадцать мегапикселей после рабочего масштабирования дает сотни мегабайт на файл.
    """
    current_side = max(mask.shape[:2])
    if current_side <= long_side:
        return mask

    factor = long_side / current_side
    width = max(1, int(mask.shape[1] * factor))
    height = max(1, int(mask.shape[0] * factor))
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA) > 0
