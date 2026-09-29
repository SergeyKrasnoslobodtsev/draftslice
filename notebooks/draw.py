"""Общие функции отрисовки для экспериментов с классификацией линий и сегментацией.

Красят реальные пикселы линии (маску `binary`), а не 1px-скелет: каждому пикселу
маски назначается цвет ближайшей по расстоянию ветки скелета. `plot_label_map` -
отдельная утилита для любой целочисленной карты меток (компоненты, регионы, кластеры).
"""

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt


def paint_lines_by_branch_color(binary, nearest_branch, branch_colors):
    """Закрашивает пикселы `binary` (реальную линию) цветом её ближайшей ветки скелета."""
    canvas = np.zeros((*binary.shape, 3))
    mask = binary.astype(bool)
    canvas[mask] = branch_colors[nearest_branch[mask]]
    return canvas


def colorize(label_image):
    """RGB-картинка (float 0..1) целочисленной карты меток: у каждой метки свой цвет, фон (0) - чёрный.

    Оттенки идут по кругу hue с шагом золотого сечения, а не случайно: соседние по порядку
    метки всегда получают далёкие цвета, и два сегмента не сливаются в один оттенок.
    """
    label_ids = np.unique(label_image)
    label_ids = label_ids[label_ids != 0]
    hues = (np.arange(len(label_ids)) * 0.618033988749895) % 1
    hsv = np.stack([hues, np.full_like(hues, 0.85), np.full_like(hues, 0.95)], axis=1)

    # Палитра по номеру метки - вся карта красится одной индексацией, без цикла по меткам.
    palette = np.zeros((label_image.max() + 1, 3))
    palette[label_ids] = mcolors.hsv_to_rgb(hsv)
    return palette[label_image]


def plot_label_map(label_image, figsize=(20, 20), ax=None):
    """Рисует целочисленную карту меток цветами `colorize`."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(colorize(label_image))
    ax.axis("off")
    return ax
