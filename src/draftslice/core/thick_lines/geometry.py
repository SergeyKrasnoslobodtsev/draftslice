from dataclasses import dataclass

import numpy as np

from draftslice.core.types import Array, Array2D


@dataclass(frozen=True, eq=False)
class Circle:
    """Окружность, подогнанная по точкам ветки.

    Attributes
    ----------
    center : Array[np.float32]
        (2,) центр (row, col); у прямой nan.
    r : float
        Радиус, px; у прямой inf.
    rel_err : float
        СКО расстояний точек до окружности, делённое на r; у прямой 0.
    """

    center: Array[np.float32]
    r: float
    rel_err: float


# публичные: их вызывает graph.py
def direction_from_node(p: Array2D[np.int64], k: int, tol_deg: float = 15.0, win: int = 3) -> Array[np.float32]:
    """Направление ветки от узла, без загиба скелета.

    Parameters
    ----------
    p : Array2D[np.int64]
        (N, 2) точки ветки (row, col) по порядку обхода, p[0] — узел.
    k : int
        Масштаб хорды, точек пути; ≈ полутолщина линии.
    tol_deg : float
        Загиб у узла — повороты выше медианы больше чем на tol_deg, градусы.
    win : int
        Длина окна направления в единицах k.

    Returns
    -------
    Array[np.float32]
        (2,) единичное направление от узла наружу. Ветка короче 3k — по всей длине.
        Меньше двух точек для направления — (0, 0): такой конец не соосен ни с одним.
    """
    if len(p) >= 3 * k:
        i0 = _hook_length(_turning_angles(_chord_directions(p, k), k), k, tol_deg)
        if i0 is not None:
            return _axis_direction(p, i0, win * k)
    return _axis_direction(p, 0, len(p))


def are_coaxial(
    dir_a: Array2D[np.float32],
    dir_b: Array2D[np.float32],
    max_angle_deg: float = 20.0,
) -> Array[np.bool_]:
    """Ветки продолжают друг друга в узле: направления от узла почти противоположны.

    Parameters
    ----------
    dir_a, dir_b : Array2D[np.float32]
        (N, 2) единичные направления концов от узла наружу; допустимо и (2,).
    max_angle_deg : float
        Наибольший излом между ветками, градусы.

    Returns
    -------
    Array[np.bool_]
        (N,) излом не больше max_angle_deg.
    """
    cos = np.einsum("...i,...i->...", dir_a, dir_b)
    return cos <= -np.cos(np.deg2rad(max_angle_deg))


def fit_circle(p: Array2D[np.int64], n_iter: int = 10, straight_px: float = 1.5) -> Circle:
    """Окружность по точкам ветки.

    Старт — метод Каса, затем уточнение по геометрическому расстоянию (Гаусс–Ньютон):
    Каса на дугах ~30° занижает радиус на 10–25 %, уточнение это убирает.

    Parameters
    ----------
    p : Array2D[np.int64]
        (N, 2) точки ветки (row, col).
    n_iter : int
        Наибольшее число итераций Гаусса–Ньютона.
    straight_px : float
        Прямая: размах точек поперёк главной оси не больше straight_px, px.

    Returns
    -------
    Circle
        Окружность; у прямой центр nan, радиус inf, невязка 0.
    """
    q = p.astype(np.float32)
    o = q.mean(axis=0)
    q = q - o
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    if np.ptp(q @ vt[1]) <= straight_px:  # отклонение от прямой — в пределах сетки
        return Circle(center=np.full(2, np.nan, np.float32), r=np.inf, rel_err=0.0)
    A = np.column_stack([2 * q, np.ones(len(q), np.float32)])
    (cx, cy, c), *_ = np.linalg.lstsq(A, (q * q).sum(axis=1), rcond=None)
    x = np.array([cx, cy, np.sqrt(c + cx * cx + cy * cy)], np.float32)  # cx, cy, r
    for _ in range(n_iter):
        d = q - x[:2]
        dist = np.hypot(d[:, 0], d[:, 1])
        res = dist - x[2]
        J = np.column_stack([-d / dist[:, None], -np.ones(len(q), np.float32)])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        x += step
        if np.abs(step).max() < 1e-3:
            break
    d = q - x[:2]
    r = float(x[2])
    rel_err = float(np.sqrt(np.mean((np.hypot(d[:, 0], d[:, 1]) - r) ** 2)) / r)

    return Circle(center=x[:2] + o, r=r, rel_err=rel_err)


def is_arc(p: Array2D[np.int64], c: Circle, max_err: float = 0.05, min_arc_deg: float = 30.0) -> bool:
    """Ветка — дуга: хорошо ложится на окружность и заметно изогнута.

    Parameters
    ----------
    p : Array2D[np.int64]
        (N, 2) точки ветки (row, col).
    c : Circle
        Окружность из fit_circle(p).
    max_err : float
        Наибольшая относительная невязка c.rel_err.
    min_arc_deg : float
        Наименьшая угловая длина дуги, градусы.

    Returns
    -------
    bool
        Ветка — дуга; прямая — никогда.
    """
    if not np.isfinite(c.r):
        return False
    return c.rel_err <= max_err and _arc_angle_deg(p, c.r) >= min_arc_deg


def circles_match(
    ca: Array2D[np.float32],
    ra: Array[np.float32],
    cb: Array2D[np.float32],
    rb: Array[np.float32],
    tol: float = 0.1,
) -> Array[np.bool_]:
    """Окружности попарно совпадают: центры и радиусы в пределах tol · R, R — средний радиус пары.

    Parameters
    ----------
    ca, cb : Array2D[np.float32]
        (N, 2) центры окружностей.
    ra, rb : Array[np.float32]
        (N,) радиусы.
    tol : float
        Допуск в долях среднего радиуса.

    Returns
    -------
    Array[np.bool_]
        (N,) пара совпадает; прямые (nan, inf) не совпадают ни с чем.
    """
    r = 0.5 * (ra + rb)
    with np.errstate(invalid="ignore"):
        return (np.linalg.norm(ca - cb, axis=-1) <= tol * r) & (np.abs(ra - rb) <= tol * r)


# приватные: их вызывают только соседи по файлу
def _chord_directions(p, k):
    """Единичные направления хорд [j, j+k]. -> (N-k, 2) float32"""
    v = (p[k:] - p[:-k]).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _turning_angles(dirs, k):
    """Угол поворота (град) между хордами j и j+k. -> (len(dirs)-k,) float32"""
    c = np.einsum("ij,ij->i", dirs[:-k], dirs[k:])
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _hook_length(turn, k, tol_deg=15.0):
    """
    Длина загиба у начала профиля поворота, в точках пути.
    Загиб — повороты выше фона (медианы) больше чем на tol_deg.
    0 — загиба нет; None — чистой части нет.
    """
    ok = turn <= np.median(turn) + tol_deg
    run = np.convolve(ok.view(np.int8), np.ones(k, np.int8), "valid") == k
    idx = np.flatnonzero(run[: max(1, len(turn) // 2)])
    if idx.size == 0:
        return None
    return 0 if idx[0] == 0 else int(idx[0]) + k


def _axis_direction(p, i0, m):
    """Главная ось точек p[i0 : i0+m], ориентированная от p[i0] вперёд; меньше двух точек — (0, 0)."""
    q = p[i0 : i0 + m].astype(np.float32)
    if len(q) < 2:
        return np.zeros(2, np.float32)
    q = q - q.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    return -d if d @ (q[-1] - q[0]) < 0 else d


def _arc_angle_deg(p, r):
    """Угловая длина пути на окружности радиуса r."""
    L = np.hypot(*np.diff(p.astype(np.float32), axis=0).T).sum()
    return float(np.degrees(L / r))
