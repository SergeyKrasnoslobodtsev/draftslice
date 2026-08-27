"""Отрисовка этапов пайплайна для блокнотов.

Модуль намеренно живет рядом с блокнотами, а не в пакете: библиотека остается свободной от
matplotlib, а картинки нужны только тому, кто подбирает параметры глазами. Все функции здесь только
рисуют и ничего не вычисляют, поэтому их можно менять, не трогая алгоритм.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from draftslice.common_types import Floats, Mask, Mat
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
