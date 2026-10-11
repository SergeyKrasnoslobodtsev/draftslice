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

from draftslice.core.thick_lines.thickness import junction_free, skeleton_thickness


@dataclass(frozen=True, eq=False)
class LineClasses:
    """Классы толщины линий графа и распределение, по которому они найдены.

    Attributes
    ----------
    thick : np.ndarray[bool]
        (P,) линия толстая: её уровень выше порога или она - разрыв толстой линии (`bridge`). Если порога нет,
        все линии толстые.
    level : np.ndarray[float32]
        (P,) уровень толщины линий, px.
    length : np.ndarray[float32]
        (P,) длина линий, px: вес в распределении.
    frame : np.ndarray[bool]
        (P,) линия — рамка листа; в распределение, горки и массы не входит.
    thr : float | None
        Порог уровня толщины, px: минимум плотности под контурной горкой. None — под контурной горкой
        значимых горок нет, классы не разделены.
    peaks : np.ndarray[float64]
        (K,) уровни значимых горок, px, по возрастанию.
    weak : np.ndarray[float64]
        (M,) уровни максимумов плотности, не прошедших по значимости, px.
    valleys : np.ndarray[float64]
        (K-1,) уровни минимумов плотности между соседними значимыми горками, px.
    masses : np.ndarray[float64]
        (K,) доли длины линий (без рамки) в каждой горке, между соседними долинами; сумма равна единице.
    contour : int | None
        Индекс контурной горки в peaks. None — нет линий вне рамки.
    grid : np.ndarray[float64]
        Сетка уровней, px, для графика плотности.
    dens : np.ndarray[float64]
        Плотность распределения уровней без рамки на сетке (интеграл по px равен единице).
    bridge : np.ndarray[bool]
        (P,) разрыв толстой линии: короткий путь с уровнем ниже порога, который с обоих концов примыкает к толстым
        и хотя бы с одного продолжает толстую почти по прямой; толстый по соседям, а не по своему уровню.
    """

    thick: np.ndarray
    level: np.ndarray
    length: np.ndarray
    frame: np.ndarray
    thr: float | None
    peaks: np.ndarray
    weak: np.ndarray
    valleys: np.ndarray
    masses: np.ndarray
    contour: int | None
    grid: np.ndarray
    dens: np.ndarray
    bridge: np.ndarray


def classify_lines(
    g: Skeleton,
    dist: np.ndarray,
    t_c: float,
    *,
    frame_span: float = 0.9,
    prominence: float = 0.05,
    bandwidth: float = 0.05,
    max_kink_deg: float = 30.0,
) -> LineClasses:
    """Разделить линии графа на толстые и тонкие порогом в долине под контурной горкой.

    Рамка — линия, охватывающая не меньше frame_span листа по высоте или ширине. Плотность логарифма
    уровня (гауссово ядро, вес по длине линии) строится по линиям вне рамки. Горка
    значима, если её выступ над окружением не меньше prominence от максимума плотности. Контурная горка —
    значимая, ближайшая к t_c. Порог — минимум плотности между ней и ближайшей значимой горкой ниже;
    линия толстая, если её уровень выше порога. Если ниже контурной горки значимых горок нет, порога нет и
    все линии толстые: потеря контура хуже оставшегося шума.

    Короткий путь (короче трёх толщин соседних толстых - столько же, сколько нужно разрезу по толщине в
    build_graph) своей надёжной толщины не имеет: между узлами штриховки скелет меряет стык, и кусок контура
    получает уровень тонкой. Такой путь толстый, если с обоих концов к нему примыкают толстые пути и хотя бы с
    одного конца он продолжает толстый путь почти по прямой (излом не больше max_kink_deg): разрыв толстой
    линии, в том числе на её изломе. Штрих штриховки подходит к линиям под углом и тонким остаётся.

    Parameters
    ----------
    g : Skeleton
        Итоговый граф (build_graph).
    dist : np.ndarray[float32]
        Расстояние до фона по краске (cv2.distanceTransform), px; его форма — размер листа.
    t_c : float
        Толщина контура в масштабе графа, px (Preprocessed.t_c * Preprocessed.scale).
    frame_span : float
        Доля листа по высоте или ширине, начиная с которой линия — рамка, в (0, 1].
    prominence : float
        Доля максимума плотности, ниже которой выступ максимума не считается горкой, в (0, 1).
    bandwidth : float
        Ширина гауссова ядра в логарифме уровня; горки ближе $e^{4 \\cdot bandwidth}$ раз сливаются.
    max_kink_deg : float
        Наибольший излом, градусы, при котором короткий путь продолжает толстую линию (как в build_graph).

    Returns
    -------
    LineClasses
        Класс каждой линии, уровни и длины, порог, горки и плотность.

    Raises
    ------
    ValueError
        t_c или bandwidth не положительны, prominence вне (0, 1) или frame_span вне (0, 1].
    """
    if t_c <= 0 or bandwidth <= 0 or not 0 < prominence < 1 or not 0 < frame_span <= 1:
        raise ValueError(f"Некорректные параметры: {t_c=}, {frame_span=}, {prominence=}, {bandwidth=}")
    level, length = _levels(g, dist)
    frame = _frame(g, dist.shape, frame_span)
    body = ~frame
    if not body.any():
        e = np.empty(0)
        no = np.zeros(len(level), bool)
        return LineClasses(np.ones(len(level), bool), level, length, frame, None, e, e, e, e, None, e, e, no)

    w = length[body].astype(np.float64)
    grid, dens = _density(level[body], w, bandwidth)
    pk, prop = find_peaks(dens, prominence=0)
    sig = pk[prop["prominences"] >= prominence * dens.max()]
    hi = int(np.argmin(np.abs(grid[sig] - t_c)))
    vall = np.array([a + np.argmin(dens[a : b + 1]) for a, b in zip(sig[:-1], sig[1:], strict=True)], int)

    thr = float(grid[vall[hi - 1]]) if hi > 0 else None
    thick = level > thr if thr is not None else np.ones(len(level), bool)
    bridge = _bridges(g, thick, level, max_kink_deg)
    thick = thick | bridge
    hump = np.searchsorted(grid[vall], level[body], side="right")  # номер горки: между соседними долинами
    masses = np.bincount(hump, weights=w, minlength=len(sig)) / w.sum()
    return LineClasses(
        thick,
        level,
        length,
        frame,
        thr,
        grid[sig],
        grid[np.setdiff1d(pk, sig)],
        grid[vall],
        masses,
        hi,
        grid,
        dens,
        bridge,
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


def _bridges(g, thick, level, max_kink_deg):
    """Короткие тонкие пути между толстыми: с обоих концов толстые, хотя бы с одного - продолжение по прямой."""
    ends: dict[int, list[tuple[int, bool]]] = {}
    for i in range(g.n_paths):
        pix = g.path(i)
        ends.setdefault(int(pix[0]), []).append((i, False))
        ends.setdefault(int(pix[-1]), []).append((i, True))
    cos_lim = np.cos(np.radians(180.0 - max_kink_deg))  # продолжение: направления наружу почти противоположны
    out = np.zeros(g.n_paths, bool)
    for i in np.flatnonzero(~thick):
        pix = g.path(i)
        yx = g.coordinates[pix]
        nb = [[(j, e) for j, e in ends[int(pix[-1] if at_end else pix[0])] if j != i and thick[j]] for at_end in (0, 1)]
        if not (nb[0] and nb[1]) or len(pix) >= 3 * np.median([level[j] for side in nb for j, _ in side]):
            continue
        straight = (
            np.dot(_out_dir(yx, at_end), _out_dir(g.coordinates[g.path(j)], e)) <= cos_lim
            for at_end in (0, 1)
            for j, e in nb[at_end]
        )
        out[i] = any(straight)
    return out


def _out_dir(yx, at_end, k=4):
    """Единичное направление пути наружу через конец (at_end) или начало, по k точкам."""
    p = yx if at_end else yx[::-1]
    d = p[-1] - p[max(len(p) - 1 - k, 0)]
    n = np.hypot(*d)
    return d / n if n else d


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
