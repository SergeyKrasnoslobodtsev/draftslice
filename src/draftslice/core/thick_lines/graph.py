import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from skan import Skeleton
from skan.csr import PathGraph

import draftslice.core.thick_lines.geometry as geom
import draftslice.core.thick_lines.thickness as thick
from draftslice.core.types import Array, Array2D


def build_graph(
    mask: Array2D[np.uint8],
    dist: Array2D[np.float32],
    max_kink_deg: float = 20.0,
    tol: float = 0.1,
    thick_ratio: float = 1.3,
    prune: bool = True,
) -> Skeleton:
    """Граф по маске краски: скелет Zhang–Suen, ветки склеены по соосности и разрезаны по толщине.

    Сначала со скелета снимаются мелкие тупиковые отростки: ветка со свободным концом, не длиннее
    толщины в своём узле, — шпора пятна стыка, а не линия (prune). После снятия узел может стать
    проходным, поэтому снятие повторяется, пока что-то снимается.

    Масштаб направления у узла k — половина 25-го процентиля толщины на скелете (2·dist), не меньше 2.
    Сначала убираются рёбра-дубли — ветки из двух точек между узлами, параллельные другой ветке
    с теми же концами (ступенька скелета на диагонали). Затем стык, распавшийся на несколько узлов,
    стягивается в один узел. После каждой правки skan заново собирает ветки.
    Затем в каждом узле концы веток объединяются в пары: сначала дуги одной окружности, затем
    соосные концы от меньшего излома; толщина пары не учитывается. Пары собираются в цепочки.
    Каждая цепочка режется по профилю толщины (без пятен стыков): бинарным разбиением, где средние
    толщины двух частей различаются больше чем в thick_ratio раз. Куски становятся путями графа.

    Parameters
    ----------
    mask : Array2D[np.uint8]
        Маска краски без текста, 0/255.
    dist : Array2D[np.float32]
        Расстояние до фона по краске (cv2.distanceTransform), px: толщина линий и радиус пятна узла.
    max_kink_deg : float
        Наибольший излом между соосными ветками, градусы.
    tol : float
        Допуск совпадения окружностей двух дуг, доля радиуса.
    thick_ratio : float
        Единственный порог по толщине: куски линии, средние толщины которых различаются не больше чем в
        thick_ratio раз, остаются одной линией; больше — линия режется. np.inf — не резать.
    prune : bool
        Снимать мелкие тупиковые отростки скелета перед сборкой графа.

    Returns
    -------
    Skeleton
        Итоговый граф: пиксели скелета маски; связи без рёбер-дублей, стык из нескольких узлов — один
        узел; пути — цепочки соосных веток, разрезанные по толщине. Ветка, не склеенная ни с чем, —
        путь из одной ветки. Соседние куски одной линии делят крайний пиксель.
    """
    skel = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    if prune:
        skel = _prune_spurs(skel, dist)
    g = _merge_close_nodes(_without_duplicates(Skeleton(skel)), dist)
    k = _dir_scale(2 * dist[skel > 0])
    links = _link_ends(g, dist, k, max_kink_deg, tol)
    t_px = thick.skeleton_thickness(g, dist)
    pieces = []
    for chain, _ in _build_chains(g.n_paths, links):
        pix = _chain_pix(g, chain)
        t, _ = thick.junction_free(t_px[pix], g.degrees[pix])
        bounds = [0, *_level_cuts(t, thick_ratio), len(pix) - 1]
        pieces += [pix[a : b + 1] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    return replace_paths(g, pieces)


def coaxial_pairs(
    g: Skeleton,
    dist: Array2D[np.float32],
    max_kink_deg: float = 30.0,
    tol: float = 0.1,
) -> Array2D[np.int64]:
    """Пары путей, соосных в узле, — без учёта толщины.

    Те же пары концов в узлах, что у build_graph, но по путям графа g, а не по его веткам. Граф не меняется.

    Parameters
    ----------
    g : Skeleton
        Граф скелета skan, обычно результат build_graph.
    dist : Array2D[np.float32]
        Расстояние до фона по краске (cv2.distanceTransform), px.
    max_kink_deg : float
        Наибольший излом между соосными путями, градусы. Мягче, чем в build_graph: у пятна узла
        направление клина отклоняется от оси линии.
    tol : float
        Допуск совпадения окружностей двух дуг, доля радиуса.

    Returns
    -------
    Array2D[np.int64]
        (n, 4) строки [путь a, сторона a, путь b, сторона b]; сторона 0 — начало пути, 1 — конец.
    """
    k = _dir_scale(thick.skeleton_thickness(g, dist))
    links = _link_ends(g, dist, k, max_kink_deg, tol)
    return np.array([(*u, *v) for u, v in links.items() if u < v], np.int64).reshape(-1, 4)


def replace_paths(g: Skeleton, pix: list[Array[np.int32]]) -> Skeleton:
    """Тот же скелет, пути заменены.

    Parameters
    ----------
    g : Skeleton
        Граф скелета skan: пиксели, связи и координаты берутся из него.
    pix : list[Array[np.int32]]
        Новые пути: id пикселей g подряд, соседние точки связаны ребром g.

    Returns
    -------
    Skeleton
        Те же пиксели и связи, что у g; пути — pix.
    """
    ip = np.r_[0, np.cumsum([len(p) for p in pix])]
    ind = np.concatenate(pix)
    data = np.ones(len(ind)) if g.pixel_values is None else g.pixel_values[ind]
    paths = sparse.csr_matrix((data, ind, ip), shape=(len(pix), len(g.coordinates)))
    pg = PathGraph(
        adj=g.graph, node_coordinates=g.coordinates, node_values=g.pixel_values, paths=paths, spacing=tuple(g.spacing)
    )
    return Skeleton.from_path_graph(pg)


def _prune_spurs(skel, dist, max_iter=10):
    """Скелет без коротких тупиковых отростков: свободный конец, другой в узле, длина не больше толщины в узле."""
    for _ in range(max_iter):
        g = Skeleton(skel)
        ip, idx = g.paths.indptr, g.paths.indices
        first, last = idx[ip[:-1]], idx[ip[1:] - 1]
        deg, length = g.degrees, g.path_lengths()
        t = thick.skeleton_thickness(g, dist)
        spur_a = (deg[first] == 1) & (deg[last] >= 3) & (length <= t[last])  # свободный конец в начале пути
        spur_b = (deg[last] == 1) & (deg[first] >= 3) & (length <= t[first])  # свободный конец в конце пути
        if not (spur_a | spur_b).any():
            break
        for p in np.flatnonzero(spur_a | spur_b):
            pix = g.path(p)
            drop = pix[:-1] if spur_a[p] else pix[1:]  # узловой пиксель остаётся
            skel[g.coordinates[drop, 0], g.coordinates[drop, 1]] = 0
    return skel


def _dir_scale(t):
    """Масштаб направления у узла, точек: половина 25-го процентиля толщины, не меньше 2."""
    return max(2, round(np.percentile(t, 25) / 2))


def _link_ends(g, dist, k, max_kink_deg, tol):
    """Пары концов путей в узлах: {(путь, сторона): (путь, сторона)}, симметричный."""
    paths = [g.path_coordinates(i) for i in range(g.n_paths)]
    pix, path, side = _path_ends(g)

    circles = [geom.fit_circle(p) for p in paths]
    centers = np.array([c.center for c in circles], np.float32)
    radii = np.array([c.r for c in circles], np.float32)
    arcs = np.array([geom.is_arc(p, c) for p, c in zip(paths, circles, strict=True)])
    centers[~arcs] = np.nan  # не дуга с окружностью не совпадает
    stub = _stub_ends(g, thick.skeleton_thickness(g, dist), pix, path)

    links = {}
    for ends in _group_by_node(pix, g.degrees):
        ends = ends[~stub[ends]]  # короткий тупик направление не держит и не должен перехватывать пару
        if len(ends) < 2:
            continue
        pts = [paths[path[e]][::-1] if side[e] else paths[path[e]] for e in ends]  # от узла наружу
        dirs = np.stack([geom.direction_from_node(p, k) for p in pts])
        pe = path[ends]
        for i, j in _match_ends(dirs, centers[pe], radii[pe], max_kink_deg, tol):
            u = (int(path[ends[i]]), int(side[ends[i]]))
            v = (int(path[ends[j]]), int(side[ends[j]]))
            links[u], links[v] = v, u
    return links


def _stub_ends(g, t_px, pix, path):
    """Концы коротких тупиковых отростков: другой конец свободный, а длина не больше толщины у узла."""
    ip, idx = g.paths.indptr, g.paths.indices
    other = np.r_[idx[ip[1:] - 1], idx[ip[:-1]]]  # противоположный конец того же пути, в порядке _path_ends
    return (g.degrees[other] == 1) & (g.path_lengths()[path] <= t_px[pix])


def _find_duplicates(g):
    """Ветки из двух точек между узлами, у которых есть параллельная ветка с теми же концами."""
    ip, idx = g.paths.indptr, g.paths.indices
    key = np.sort(np.c_[idx[ip[:-1]], idx[ip[1:] - 1]], axis=1)  # пара концов без учёта направления
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return (np.diff(ip) == 2) & (cnt[inv.ravel()] > 1)


def _without_duplicates(g):
    """Скелет без рёбер-дублей; skan заново собирает ветки, узлы ступенек становятся точками веток."""
    dup = _find_duplicates(g)
    if not dup.any():
        return g
    ip, idx = g.paths.indptr, g.paths.indices
    a, b = idx[ip[:-1]][dup], idx[ip[1:] - 1][dup]
    adj = g.graph.copy()
    adj[a, b] = 0
    adj[b, a] = 0
    adj.eliminate_zeros()
    pg = PathGraph.from_graph(adj=adj, node_coordinates=g.coordinates, node_values=g.pixel_values, spacing=g.spacing)
    return Skeleton.from_path_graph(pg)


def _merge_close_nodes(g, dist):
    """
    Стык, распавшийся на несколько узлов, стягивается в один узел.
    Перемычка — ветка между двумя узлами короче суммы радиусов пятен (dist) на её концах: пятна
    перекрываются, это один стык. Узлы, связанные перемычками, вместе с пикселями перемычек становятся
    одним узлом — пикселем группы, ближайшим к её средней точке; skan заново собирает ветки.
    """
    ip, idx, deg, yx = g.paths.indptr, g.paths.indices, g.degrees, g.coordinates
    a, b = idx[ip[:-1]], idx[ip[1:] - 1]
    r = dist[yx[:, 0], yx[:, 1]]
    bridge = np.flatnonzero((deg[a] >= 3) & (deg[b] >= 3) & (a != b) & (g.path_lengths() <= r[a] + r[b]))
    if bridge.size == 0:
        return g

    # группы: пиксели перемычек, связанные подряд вдоль каждой перемычки
    n = len(yx)
    u = np.concatenate([idx[ip[i] : ip[i + 1] - 1] for i in bridge])
    v = np.concatenate([idx[ip[i] + 1 : ip[i + 1]] for i in bridge])
    _, comp = csgraph.connected_components(sparse.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)), directed=False)
    grp = np.unique(np.r_[u, v])
    grp = grp[np.argsort(comp[grp], kind="stable")]
    _, starts = np.unique(comp[grp], return_index=True)
    rep = np.arange(n)  # пиксель -> пиксель его узла
    for px in np.split(grp, starts[1:]):
        center = yx[px].mean(axis=0)
        rep[px] = px[np.argmin(((yx[px] - center) ** 2).sum(axis=1))]

    # рёбра группы перевешиваются на её пиксель; повторы после стягивания — одно ребро
    adj = g.graph.tocoo()
    p, q = rep[adj.row], rep[adj.col]
    pairs = np.unique(np.c_[p, q][p != q], axis=0)
    w = np.hypot(*((yx[pairs[:, 0]] - yx[pairs[:, 1]]) * g.spacing).T)
    adj = sparse.csr_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    pg = PathGraph.from_graph(adj=adj, node_coordinates=yx, node_values=g.pixel_values, spacing=g.spacing)
    return Skeleton.from_path_graph(pg)


def _path_ends(g):
    """Концы веток: пиксель, ветка, сторона (0 — начало, 1 — конец)."""
    ip, idx = g.paths.indptr, g.paths.indices
    P = g.n_paths
    pix = np.r_[idx[ip[:-1]], idx[ip[1:] - 1]]
    path = np.r_[np.arange(P), np.arange(P)]
    side = np.r_[np.zeros(P, np.int8), np.ones(P, np.int8)]
    return pix, path, side


def _group_by_node(pix, degrees):
    """Номера концов, сходящихся в одном узле степени ≥ 3."""
    ids = np.flatnonzero(degrees[pix] >= 3)
    ids = ids[np.argsort(pix[ids], kind="stable")]
    _, starts = np.unique(pix[ids], return_index=True)
    return np.split(ids, starts[1:])


def _match_ends(dirs, centers, radii, max_kink_deg, tol):
    """Пары концов в узле: сначала дуги одной окружности, затем соосные от меньшего излома. Центр не-дуги — NaN."""
    n = len(dirs)
    if n < 2:
        return []
    a, b = np.triu_indices(n, 1)
    circle = geom.circles_match(centers[a], radii[a], centers[b], radii[b], tol)
    coax = geom.are_coaxial(dirs[a], dirs[b], max_kink_deg)
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


def _level_cuts(t, thick_ratio, min_len=3.0):
    """Разрезы профиля толщины t, где средняя толщина частей различается больше чем в thick_ratio раз.

    Бинарное разбиение по логарифму толщины: лучший разрез (с весом по длинам частей) принимается, если
    отношение средних частей больше thick_ratio; обе части не короче min_len толщин линии.
    """
    lt = np.log(np.maximum(t, 1.0))
    lim = np.log(thick_ratio)

    def split(a, b):
        x = lt[a:b]
        n = len(x)
        m = int(np.ceil(min_len * np.exp(np.median(x))))
        if n < 2 * m:
            return []
        c = np.cumsum(x)
        i = np.arange(m, n - m + 1)  # разрез перед точкой i
        d = np.abs(c[i - 1] / i - (c[-1] - c[i - 1]) / (n - i))
        j = int(np.argmax(np.sqrt(i * (n - i) / n) * d))
        if d[j] <= lim:
            return []
        cut = a + int(i[j])
        return split(a, cut) + [cut] + split(cut, b)

    return split(0, len(t))


def _build_chains(n_paths, links):
    """Цепочки веток по парам концов и их замкнутость."""
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


def _chain_pix(g, chain):
    """id пикселей цепочки подряд; общий пиксель узла не дублируется."""
    ip, idx = g.paths.indptr, g.paths.indices
    parts = []
    for p, rev in chain:
        s = idx[ip[p] : ip[p + 1]]
        s = s[::-1] if rev else s
        parts.append(s if not parts else s[1:])
    return np.concatenate(parts)
