"""Деление линий графа на толстые и тонкие по среднему уровню толщины.

У каждой линии итогового графа есть уровень толщины (среднее геометрическое профиля без пятен стыков).
Порог между классами — среднее уровней линий, каждая линия считается один раз (без веса по длине). Порог не
требует искать горки распределения, поэтому не зависит от числа промежуточных горок и от шума классов.
"""

from dataclasses import dataclass

import numpy as np
from skan import Skeleton

from draftslice.core.thick_lines.thickness import junction_free, skeleton_thickness
from draftslice.core.types import Array, Array2D


@dataclass(frozen=True, eq=False)
class LineClasses:
    """Классы толщины линий графа и то, по чему они найдены.

    Attributes
    ----------
    thick : Array[np.bool_]
        (P,) линия толстая: её уровень выше порога.
    level : Array[np.float64]
        (P,) уровень толщины линий, px.
    length : Array[np.float64]
        (P,) длина линий, px: вес в распределении.
    thr : float
        Порог уровня толщины, px: среднее уровней линий, умноженное на thr_scale.
    mean : float
        Среднее уровней линий, px, без веса по длине (порог до умножения на thr_scale).
    centers : tuple[float, ...]
        Взвешенные по длине медианы тонких и толстых линий, px; у пустого класса значения нет.
    masses : tuple[float, ...]
        Доли длины тонких и толстых линий; у пустого класса значения нет.
    grid : Array[np.float64]
        Сетка уровней, px, для графика плотности.
    dens : Array[np.float64]
        Плотность распределения уровней на сетке (интеграл по px равен единице).
    """

    thick: Array[np.bool_]
    level: Array[np.float64]
    length: Array[np.float64]
    thr: float
    mean: float
    centers: tuple[float, ...]
    masses: tuple[float, ...]
    grid: Array[np.float64]
    dens: Array[np.float64]


def classify_lines(
    g: Skeleton,
    dist: Array2D[np.float32],
    thr_scale: float = 1.0,
    bandwidth: float = 0.066,
) -> LineClasses:
    """Разделить линии графа на толстые и тонкие по порогу, равному среднему уровню толщины.

    Среднее считается по уровням линий, каждая линия — одно наблюдение. Линия толстая, если её уровень выше
    порога. Длины линий нужны только для медиан и долей классов и для графика плотности (гауссово ядро по
    логарифму уровня).

    Parameters
    ----------
    g : Skeleton
        Итоговый граф (build_graph).
    dist : Array2D[np.float32]
        Расстояние до фона по краске (cv2.distanceTransform), px.
    thr_scale : float
        Множитель порога: больше единицы — меньше толстых, меньше — больше.
    bandwidth : float
        Ширина гауссова ядра в логарифме уровня для графика плотности.

    Returns
    -------
    LineClasses
        Класс каждой линии, порог и параметры распределения.

    Raises
    ------
    ValueError
        thr_scale или bandwidth не положительны.
    """
    if thr_scale <= 0 or bandwidth <= 0:
        raise ValueError(f"thr_scale и bandwidth должны быть положительны: {thr_scale=}, {bandwidth=}")
    level, length = _levels(g, dist)
    if len(level) == 0:
        empty = np.empty(0)
        return LineClasses(np.zeros(0, bool), level, length, 0.0, 0.0, (), (), empty, empty)
    mean = float(np.mean(level))
    thr = mean * thr_scale
    thick = level > thr
    centers = tuple(_wmedian(level[m], length[m]) for m in (~thick, thick) if m.any())
    masses = tuple(float(length[m].sum() / length.sum()) for m in (~thick, thick) if m.any())
    grid, dens = _density(level, length, bandwidth)
    return LineClasses(thick, level, length, thr, mean, centers, masses, grid, dens)


def _levels(g, dist):
    """Уровень толщины и длина каждой линии графа."""
    t_px = skeleton_thickness(g, dist)
    level = np.empty(g.n_paths)
    for i in range(g.n_paths):
        pix = g.path(i)
        t, _ = junction_free(t_px[pix], g.degrees[pix])
        level[i] = np.exp(np.log(np.maximum(t, 1.0)).mean())  # среднее геометрическое: толщина мультипликативна
    return level, g.path_lengths().astype(np.float64)


def _wmedian(x, w):
    """Медиана x с весами w."""
    o = np.argsort(x)
    return float(x[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2)])


def _density(level, length, bandwidth, grid_size=600):
    """Плотность логарифма уровня с весом по длине на сетке уровней (px); интеграл по px равен единице."""
    x = np.log(np.maximum(level, 1.0))
    w = length / length.sum()
    g = np.linspace(x.min() - 3 * bandwidth, x.max() + 3 * bandwidth, grid_size)
    dens = (w[:, None] * np.exp(-0.5 * ((g[None, :] - x[:, None]) / bandwidth) ** 2)).sum(axis=0)
    grid = np.exp(g)
    return grid, dens / np.trapezoid(dens, grid)
