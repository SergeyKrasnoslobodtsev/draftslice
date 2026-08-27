"""Замыкание разрывов контура и заливка внутренних пустот.

После удаления текста и тонких линий контур детали остается разорванным в тех местах, где его
пересекала размерная линия или накрывала надпись. Разрыв ломает топологию: кусок контура получает
свободный конец и перестает отличаться от обрывка выносной линии, поэтому связность чинится до
того, как по графу принимаются решения.

Изотропное замыкание для этого не годится: круглый структурный элемент склеивает не только концы
одной линии, но и соседние параллельные линии, а на чертеже со штриховкой зазор между штрихами
того же порядка, что и длина разрыва. Поэтому ядро берется линейным и применяется в нескольких
ориентациях, а результаты объединяются: разрыв вдоль линии закрывается ядром, совпадающим с
направлением этой линии, а поперек линия не растет. Длина ядра ограничена снизу длиной разрыва,
сверху минимальным зазором между соседними линиями, и подбирается по кадру.

Заливка внутренних пустот работает от противного: внешний фон заливается от угла рамки, и нулями
остается только то, до чего снаружи не добраться. Рамка обязательна, иначе ось, идущая от края до
края габарита, отрезала бы угол и он был бы засчитан как пустота.
"""

import cv2
import numpy as np

from draftslice.common_types import Mask, Mat


def fill_internal_holes(strokes_mask: Mask) -> Mask:
    """Залить пустоты, окруженные штрихами.

    Parameters
    ----------
    strokes_mask : Mask
        Бинарная маска штрихов, True это чернила.

    Returns
    -------
    Mask
        Маска штрихов вместе с их внутренними пустотами.

    Notes
    -----
    Заливка внешнего фона идет по четырехсвязности, а штрих восьмисвязен, поэтому замкнутый контур
    для заливки герметичен и его внутренность остается неокрашенной.
    """
    padded_height = strokes_mask.shape[0] + 2
    padded_width = strokes_mask.shape[1] + 2
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[1:-1, 1:-1] = strokes_mask.astype(np.uint8) * 255

    flood_mask = np.zeros((padded_height + 2, padded_width + 2), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)
    return strokes_mask | (padded[1:-1, 1:-1] == 0)


def build_line_kernel(kernel_length: int, orientation_degrees: float) -> Mat:
    """Построить линейный структурный элемент заданной длины и наклона.

    Parameters
    ----------
    kernel_length : int
        Длина элемента в пикселях. Четная длина увеличивается на единицу, чтобы у линии был центр.
    orientation_degrees : float
        Наклон линии в градусах, отсчет против часовой стрелки от горизонтали.

    Returns
    -------
    Mat
        Квадратное ядро из нулей и единиц с прочерченной через центр линией.
    """
    if kernel_length < 1:
        raise ValueError(f"длина ядра должна быть положительной, получено {kernel_length}")

    kernel_size = kernel_length + 1 - kernel_length % 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    center = kernel_size // 2
    radians = np.radians(orientation_degrees)
    offset = np.array([np.cos(radians), -np.sin(radians)]) * center

    start_point = np.round(center - offset).astype(int)
    end_point = np.round(center + offset).astype(int)
    cv2.line(kernel, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), 1, 1)
    return kernel


def close_along_orientations(strokes_mask: Mask, kernel_length: int, orientation_count: int) -> Mask:
    """Замкнуть разрывы вдоль линий набором линейных ядер.

    Parameters
    ----------
    strokes_mask : Mask
        Бинарная маска штрихов, True это чернила.
    kernel_length : int
        Длина линейного структурного элемента в пикселях.
    orientation_count : int
        Число равномерных ориентаций в диапазоне от нуля до ста восьмидесяти градусов.

    Returns
    -------
    Mask
        Объединение замыканий по всем ориентациям.

    Notes
    -----
    Ядро, ориентированное поперек линии, способно сшить две соседние параллельные линии, если зазор
    между ними меньше длины ядра. Признак того, что длина завышена, это скачок числа замкнутых
    пустот в маске.

    У самой границы кадра результат шире исходной маски: эрозия у края не срезает то, что нарастила
    дилатация, потому что за границей нет данных. На чертеже это не проявляется, пока штрихи не
    подходят к краю ближе половины длины ядра.
    """
    if orientation_count < 1:
        raise ValueError(f"число ориентаций должно быть положительным, получено {orientation_count}")

    source = strokes_mask.astype(np.uint8)
    closed = np.zeros_like(source)
    for orientation_degrees in np.linspace(0.0, 180.0, orientation_count, endpoint=False):
        kernel = build_line_kernel(kernel_length, float(orientation_degrees))
        closed |= cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)
    return closed > 0
