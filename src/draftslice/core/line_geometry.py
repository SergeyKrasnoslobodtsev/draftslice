"""
Геометрия веток скелета.

Назначение 1 — соосность: продолжают ли две ветки друг друга в узле.
    direction_from_node(p, k)  -> направление ветки от узла (без загиба скелета)
    are_coaxial(d_a, d_b)      -> излом между ветками ≤ допуска

Назначение 2 — дуги и кольца.
    fit_circle(p)              -> центр, радиус, относительная невязка
    is_arc(p)                  -> путь — дуга (не прямая, не ломаная)
    circles_match(ca, ra, cb, rb) -> окружности совпадают (векторно)
    same_circle(p_a, p_b)      -> две дуги лежат на одной окружности
    is_ring(p)                 -> путь — замкнутое кольцо

Соглашения:
    p — точки пути (N, 2) в порядке обхода, int (как из skan).
    k — масштаб в точках пути, ≈ полутолщина линии.
"""

import numpy as np


# ------- Геометрия линий (хорды, углы, загибы, оси) -------
def chord_directions(p, k):
    """Единичные направления хорд [j, j+k]. -> (N-k, 2) float32"""
    v = (p[k:] - p[:-k]).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def turning_angles(dirs, k):
    """Угол поворота (град) между хордами j и j+k. -> (len(dirs)-k,) float32"""
    c = np.einsum("ij,ij->i", dirs[:-k], dirs[k:])
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def hook_length(turn, k, tol_deg=15.0):
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


def axis_direction(p, i0, m):
    """Главная ось точек p[i0 : i0+m], ориентированная от p[i0] вперёд. -> (2,) float32 или None"""
    q = p[i0 : i0 + m].astype(np.float32)
    if len(q) < 2:
        return None
    q = q - q.mean(axis=0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    return -d if d @ (q[-1] - q[0]) < 0 else d


def direction_from_node(p, k, tol_deg=15.0, win=3):
    """
    Направление ветки от узла, без загиба скелета. p[0] — узел.
    win — длина окна в единицах k. Короткая ветка (< 3k) — по всей длине.
    """
    if len(p) >= 3 * k:
        i0 = hook_length(turning_angles(chord_directions(p, k), k), k, tol_deg)
        if i0 is not None:
            return axis_direction(p, i0, win * k)
    return axis_direction(p, 0, len(p))


# ------- Соосность -------
def are_coaxial(dir_a, dir_b, max_angle_deg=20.0):
    """
    Ветки продолжают друг друга: направления от узла почти противоположны,
    излом ≤ max_angle_deg. dir_a, dir_b: (2,) или (N, 2).
    """
    cos = np.einsum("...i,...i->...", dir_a, dir_b)
    return cos <= -np.cos(np.deg2rad(max_angle_deg))


def dedup(p):
    """Убрать подряд идущие одинаковые точки после округления."""
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
    return p[keep]


def raster_line(start, angle_deg, length):
    """Пиксельная прямая (row, col): ось row вниз, угол от оси col против часовой."""
    t = np.deg2rad(angle_deg)
    s = np.arange(0, length, 0.5)
    pts = np.c_[start[0] - s * np.sin(t), start[1] + s * np.cos(t)]
    return dedup(np.rint(pts).astype(np.int32))


def unit(angle_deg):
    t = np.deg2rad(angle_deg)
    return np.array([-np.sin(t), np.cos(t)], np.float32)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))


# ------- Окружности и дуги -------


def fit_circle(p, n_iter=10, straight_px=1.5):
    """
    Окружность по точкам: старт методом Каса, затем уточнение
    по геометрическому расстоянию (Гаусс–Ньютон). Каса на дугах ~30°
    занижает радиус на 10–25%, уточнение это убирает.
    Прямая (все точки в пределах пикселя от своей оси) -> center = nan, radius = inf.
    -> center (2,) float32, radius, rel_err (СКО расстояний до окружности / radius)
    """
    q = p.astype(np.float32)
    o = q.mean(axis=0)
    q = q - o  # центрирование: float32 хватает
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    if np.ptp(q @ vt[1]) <= straight_px:  # отклонение от прямой — в пределах сетки
        return np.full(2, np.nan, np.float32), np.inf, 0.0
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
    return x[:2] + o, r, rel_err


def arc_angle_deg(p, radius):
    """Угловая длина пути на окружности радиуса radius."""
    L = np.hypot(*np.diff(p.astype(np.float32), axis=0).T).sum()
    return float(np.degrees(L / radius))


def is_arc(p, r, e, max_err=0.05, min_arc_deg=30.0):
    """Путь — дуга по готовой подгонке (r, e из fit_circle)."""
    return e <= max_err and arc_angle_deg(p, r) >= min_arc_deg


def same_circle(p_a, p_b, tol=0.1, max_err=0.05, min_arc_deg=30.0):
    """
    Две ветки — дуги одной окружности.
    tol — допуск на центр и радиус, в долях радиуса.
    """
    if not (is_arc(p_a, max_err, min_arc_deg) and is_arc(p_b, max_err, min_arc_deg)):
        return False
    ca, ra, _ = fit_circle(p_a)
    cb, rb, _ = fit_circle(p_b)
    r = 0.5 * (ra + rb)
    return bool(np.linalg.norm(ca - cb) <= tol * r and abs(ra - rb) <= tol * r)


def is_ring(p, r, e, max_err=0.05, min_arc_deg=330.0, close_ratio=0.15):
    """Путь — замкнутое кольцо по готовой подгонке (r, e из fit_circle)."""
    gap = np.linalg.norm((p[-1] - p[0]).astype(np.float32))
    return bool(e <= max_err and arc_angle_deg(p, r) >= min_arc_deg and gap <= close_ratio * r)


def circles_match(ca, ra, cb, rb, tol=0.1):
    """
    Две окружности совпадают: центры и радиусы в пределах tol·R.
    Работает и для массивов (N, 2)/(N,). Прямые (nan, inf) не совпадают ни с чем.
    """
    r = 0.5 * (ra + rb)
    with np.errstate(invalid="ignore"):
        return (np.linalg.norm(ca - cb, axis=-1) <= tol * r) & (np.abs(ra - rb) <= tol * r)
