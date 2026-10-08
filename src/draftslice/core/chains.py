"""
Склейка веток скелета в линии через узлы.

В каждом узле концы веток объединяются в пары:
    1) дуги одной окружности (приоритет — дуга не уйдёт на касательную прямую);
    2) соосные концы, от меньшего излома к большему.
Пары собираются в цепочки; цепочка, вернувшаяся в начало, — замкнутая.

    path_ends(g)                     -> концы всех веток (пиксель, ветка, сторона)
    group_by_node(pix, degrees)      -> концы, сгруппированные по узлам
    match_ends(dirs, is_arc, C, R)   -> пары концов в одном узле (чистая функция)
    build_chains(n_paths, links)     -> цепочки веток (чистая функция)
    chain_coordinates(paths, chain)  -> точки цепочки подряд
    merge_paths(g, k)                -> всё вместе для скелета skan
"""

import numpy as np

from draftslice.core import line_geometry as l


def path_ends(g):
    """
    Концы веток skan. Сторона 0 — начало пути, 1 — конец.
    -> pix (2P,) id пикселя, path (2P,), side (2P,)
    """
    ip, idx = g.paths.indptr, g.paths.indices
    P = g.n_paths
    pix = np.r_[idx[ip[:-1]], idx[ip[1:] - 1]]
    path = np.r_[np.arange(P), np.arange(P)]
    side = np.r_[np.zeros(P, np.int8), np.ones(P, np.int8)]
    return pix, path, side


def group_by_node(pix, degrees):
    """Индексы концов, сходящихся в одном узле (степень ≥ 3). -> list of int-массивов"""
    ids = np.flatnonzero(degrees[pix] >= 3)
    ids = ids[np.argsort(pix[ids], kind="stable")]
    _, starts = np.unique(pix[ids], return_index=True)
    return np.split(ids, starts[1:])


def match_ends(dirs, arc, centers, radii, max_kink_deg=20.0, tol=0.1):
    """
    Пары концов в одном узле. Все входы — по концам этого узла:
        dirs (n, 2)    направления от узла наружу
        arc (n,)       ветка — дуга
        centers (n, 2), radii (n,)  окружности веток (прямые: nan, inf)
    Дуги одной окружности — первыми; остальные — соосные, от меньшего излома.
    Каждый конец — не больше чем в одной паре.
    -> list of (i, j)
    """
    n = len(dirs)
    if n < 2:
        return []
    a, b = np.triu_indices(n, 1)
    circle = arc[a] & arc[b] & l.circles_match(centers[a], radii[a], centers[b], radii[b], tol)
    coax = l.are_coaxial(dirs[a], dirs[b], max_kink_deg)
    kink = 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", dirs[a], dirs[b]), -1, 1)))
    score = np.where(circle, -1.0, np.where(coax, kink, np.inf))
    used = np.zeros(n, bool)
    pairs = []
    for m in np.argsort(score, kind="stable"):
        if not np.isfinite(score[m]):
            break
        i, j = a[m], b[m]
        if not (used[i] or used[j]):
            used[i] = used[j] = True
            pairs.append((int(i), int(j)))
    return pairs


def build_chains(n_paths, links):
    """
    links — симметричный dict {(ветка, сторона): (ветка, сторона)}.
    -> list of (chain, closed); chain — list of (ветка, reversed)
    """
    visited = np.zeros(n_paths, bool)
    chains = []
    for p0 in range(n_paths):
        if visited[p0]:
            continue
        # назад к началу цепочки: входим в ветку p со стороны s
        p, s = p0, 0
        for _ in range(n_paths):
            if (p, s) not in links:
                break
            q, t = links[(p, s)]
            p, s = q, 1 - t
            if p == p0 and s == 0:
                break  # кольцо: начинаем с p0
        start = (p, s)
        chain, closed = [], False
        while True:
            visited[p] = True
            chain.append((p, bool(s)))
            nxt = links.get((p, 1 - s))
            if nxt is None:
                break
            if nxt == start:
                closed = True
                break
            p, s = nxt
            if visited[p]:  # защита от некорректных links
                break
        chains.append((chain, closed))
    return chains


def chain_coordinates(paths, chain):
    """Точки цепочки подряд; общая точка узла между ветками не дублируется."""
    parts = []
    for p, rev in chain:
        c = paths[p][::-1] if rev else paths[p]
        parts.append(c if not parts else c[1:])
    return np.vstack(parts)


def merge_paths(g, k, max_kink_deg=20.0, tol=0.1):
    """
    Склейка веток скелета skan.
    k — масштаб для направления у узла (≈ полутолщина линии, в точках).
    -> list of (chain, closed), paths (координаты веток), fits (подгонки окружностей)
    """
    paths = [g.path_coordinates(i) for i in range(g.n_paths)]
    pix, path, side = path_ends(g)

    fits = [l.fit_circle(p) for p in paths]
    centers = np.array([c for c, _, _ in fits], np.float32)
    radii = np.array([r for _, r, _ in fits], np.float32)
    arcs = np.array([l.is_arc(p, r, e) for p, (_, r, e) in zip(paths, fits, strict=False)])

    links = {}
    for ends in group_by_node(pix, g.degrees):
        dirs = np.stack([l.direction_from_node(paths[path[e]][::-1] if side[e] else paths[path[e]], k) for e in ends])
        pe = path[ends]
        for i, j in match_ends(dirs, arcs[pe], centers[pe], radii[pe], max_kink_deg, tol):
            u = (int(path[ends[i]]), int(side[ends[i]]))
            v = (int(path[ends[j]]), int(side[ends[j]]))
            links[u], links[v] = v, u

    chains = build_chains(g.n_paths, links)
    # ветка-петля без узла (начало и конец в одном пикселе) — тоже замкнута
    ip, idx = g.paths.indptr, g.paths.indices
    out = []
    for chain, closed in chains:
        if not closed and len(chain) == 1:
            p = chain[0][0]
            closed = idx[ip[p]] == idx[ip[p + 1] - 1]
        out.append((chain, bool(closed)))
    return out, paths, fits
