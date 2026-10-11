"""Деление линий графа на толстые и тонкие по горкам распределения уровней толщины.

У каждой линии итогового графа есть уровень толщины (среднее геометрическое профиля без пятен стыков).
Линии одного класса дают горку плотности уровней с весом по длине. Контурная горка — ближайшая к толщине
контура листа (её замеряет preprocess), порог — минимум плотности между ней и ближайшей значимой горкой
ниже. Рамка листа в распределение не входит: она не часть чертежа и сама даёт лишнюю горку.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from skan import Skeleton

from draftslice.core.exceptions import DraftsliceValueError
from draftslice.core.thick_lines.thickness import junction_free, skeleton_thickness
from draftslice.core.types import Array, Array2D


@dataclass(frozen=True, eq=False)
class LineClasses:
    """Классы толщины линий графа и распределение, по которому они найдены.

    Attributes
    ----------
    thick : Array[np.bool_]
        (P,) линия толстая: её уровень выше порога. Если порога нет, все линии толстые.
    level : Array[np.float32]
        (P,) уровень толщины линий, px.
    length : Array[np.float32]
        (P,) длина линий, px: вес в распределении.
    frame : Array[np.bool_]
        (P,) линия — рамка листа; в распределение, горки и массы не входит.
    thr : float | None
        Порог уровня толщины, px: минимум плотности под контурной горкой. None — под контурной горкой
        значимых горок нет, классы не разделены.
    peaks : Array[np.float64]
        (K,) уровни значимых горок, px, по возрастанию.
    weak : Array[np.float64]
        (M,) уровни максимумов плотности, не прошедших по значимости, px.
    valleys : Array[np.float64]
        (K-1,) уровни минимумов плотности между соседними значимыми горками, px.
    masses : Array[np.float64]
        (K,) доли длины линий (без рамки) в каждой горке, между соседними долинами; сумма равна единице.
    contour : int | None
        Индекс контурной горки в peaks. None — нет линий вне рамки.
    grid : Array[np.float64]
        Сетка уровней, px, для графика плотности.
    dens : Array[np.float64]
        Плотность распределения уровней без рамки на сетке (интеграл по px равен единице).
    """

    thick: Array[np.bool_]
    level: Array[np.float32]
    length: Array[np.float32]
    frame: Array[np.bool_]
    thr: float | None
    peaks: Array[np.float64]
    weak: Array[np.float64]
    valleys: Array[np.float64]
    masses: Array[np.float64]
    contour: int | None
    grid: Array[np.float64]
    dens: Array[np.float64]


def classify_lines(
    g: Skeleton,
    dist: Array2D[np.float32],
    t_c: float,
    *,
    frame_span: float = 0.9,
    prominence: float = 0.05,
    bandwidth: float = 0.05,
) -> LineClasses:
    """Разделить линии графа на толстые и тонкие порогом в долине под контурной горкой.

    Рамка — линия, охватывающая не меньше frame_span листа по высоте или ширине. Плотность логарифма
    уровня (гауссово ядро, вес по длине линии) строится по линиям вне рамки. Горка
    значима, если её выступ над окружением не меньше prominence от максимума плотности. Контурная горка —
    значимая, ближайшая к t_c. Порог — минимум плотности между ней и ближайшей значимой горкой ниже;
    линия толстая, если её уровень выше порога. Если ниже контурной горки значимых горок нет, порога нет и
    все линии толстые: потеря контура хуже оставшегося шума.

    Parameters
    ----------
    g : Skeleton
        Итоговый граф (build_graph).
    dist : Array2D[np.float32]
        Расстояние до фона по краске (cv2.distanceTransform), px; его форма — размер листа.
    t_c : float
        Толщина контура в масштабе графа, px (Preprocessed.t_c * Preprocessed.scale).
    frame_span : float
        Доля листа по высоте или ширине, начиная с которой линия — рамка, в (0, 1].
    prominence : float
        Доля максимума плотности, ниже которой выступ максимума не считается горкой, в (0, 1).
    bandwidth : float
        Ширина гауссова ядра в логарифме уровня; горки ближе $e^{4 \\cdot bandwidth}$ раз сливаются.

    Returns
    -------
    LineClasses
        Класс каждой линии, уровни и длины, порог, горки и плотность.

    Raises
    ------
    DraftsliceValueError
        t_c или bandwidth не положительны, prominence вне (0, 1) или frame_span вне (0, 1].
    """
    if t_c <= 0 or bandwidth <= 0 or not 0 < prominence < 1 or not 0 < frame_span <= 1:
        raise DraftsliceValueError(f"Некорректные параметры: {t_c=}, {frame_span=}, {prominence=}, {bandwidth=}")
    level, length = _levels(g, dist)
    frame = _frame(g, dist.shape, frame_span)
    body = ~frame
    if not body.any():
        e = np.empty(0)
        return LineClasses(np.ones(len(level), bool), level, length, frame, None, e, e, e, e, None, e, e)

    w = length[body].astype(np.float64)
    grid, dens = _density(level[body], w, bandwidth)
    pk, prop = find_peaks(dens, prominence=0)
    sig = pk[prop["prominences"] >= prominence * dens.max()]
    hi = int(np.argmin(np.abs(grid[sig] - t_c)))
    vall = np.array([a + np.argmin(dens[a : b + 1]) for a, b in zip(sig[:-1], sig[1:], strict=True)], int)

    thr = float(grid[vall[hi - 1]]) if hi > 0 else None
    thick = level > thr if thr is not None else np.ones(len(level), bool)
    hump = np.searchsorted(grid[vall], level[body], side="right")  # номер горки: между соседними долинами
    masses = np.bincount(hump, weights=w, minlength=len(sig)) / w.sum()
    return LineClasses(
        thick, level, length, frame, thr, grid[sig], grid[np.setdiff1d(pk, sig)], grid[vall], masses, hi, grid, dens
    )


def _levels(g, dist):
    """Уровень толщины и длина каждой линии графа."""
    t_px = skeleton_thickness(g, dist)
    level = np.empty(g.n_paths, np.float32)
    for i in range(g.n_paths):
        pix = g.path(i)
        t, _ = junction_free(t_px[pix], g.degrees[pix])
        level[i] = np.exp(np.log(np.maximum(t, 1.0)).mean())  # среднее геометрическое: толщина мультипликативна
    return level, g.path_lengths().astype(np.float32)


def _frame(g, shape, frame_span):
    """Линии, охватывающие не меньше frame_span листа по высоте или ширине."""
    ext = np.array([np.ptp(g.path_coordinates(i), axis=0) for i in range(g.n_paths)]).reshape(-1, 2)
    return (ext[:, 0] >= frame_span * shape[0]) | (ext[:, 1] >= frame_span * shape[1])


def _density(level, length, bandwidth, grid_size=600):
    """Плотность логарифма уровня с весом по длине на сетке уровней (px); интеграл по px равен единице."""
    x = np.log(np.maximum(level, 1.0).astype(np.float64))  # float32 обнуляет хвосты ядер: порог уезжает к горке
    w = length / length.sum()
    g = np.linspace(x.min() - 3 * bandwidth, x.max() + 3 * bandwidth, grid_size)
    dens = (w[:, None] * np.exp(-0.5 * ((g[None, :] - x[:, None]) / bandwidth) ** 2)).sum(axis=0)
    grid = np.exp(g)
    return grid, dens / np.trapezoid(dens, grid)
