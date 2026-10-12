"""Сетка плиток листа и зоны плиток для склейки находок без дублей.

Сетка совпадает с `training/make_dataset.grid`: модель стрелок обучена на этих плитках.
"""

Box = tuple[int, int, int, int]
Zone = tuple[float, float, float, float]


def grid(h: int, w: int, tile: int, overlap: float) -> list[Box]:
    """Сетка плиток листа: шаг `tile - int(overlap * tile)`, последняя плитка в ряду прижата к краю.

    Parameters
    ----------
    h, w : int
        Высота и ширина листа, px.
    tile : int
        Сторона плитки, px. Если лист меньше плитки, плитка обрезается по листу.
    overlap : float
        Перекрытие соседних плиток, доля стороны, в [0, 1).

    Returns
    -------
    list[tuple[int, int, int, int]]
        Плитки (x0, y0, x1, y1) построчно, сверху вниз и слева направо.

    Raises
    ------
    ValueError
        `tile <= 0` или `overlap` вне [0, 1).
    """
    if tile <= 0 or not 0 <= overlap < 1:
        raise ValueError(f"Некорректная сетка: {tile=}, {overlap=}")
    ov = int(overlap * tile)
    boxes, y_min, y_max = [], 0, 0
    while y_max < h:
        x_min = x_max = 0
        y_max = y_min + tile
        while x_max < w:
            x_max = x_min + tile
            if y_max > h or x_max > w:
                xe, ye = min(w, x_max), min(h, y_max)
                boxes.append((max(0, xe - tile), max(0, ye - tile), xe, ye))
            else:
                boxes.append((x_min, y_min, x_max, y_max))
            x_min = x_max - ov
        y_min = y_max - ov
    return boxes


def zones(tiles: list[Box], w: int, h: int) -> dict[Box, Zone]:
    """Зоны плиток: граница соседних - середина их перекрытия, зоны делят лист без пересечений и пропусков.

    Находка засчитывается плитке, в зону которой попал центр её бокса: объект из перекрытия - один раз, а объект,
    обрезанный краем плитки, - той соседней, что видит его целиком (если он меньше половины перекрытия).

    Parameters
    ----------
    tiles : list[tuple[int, int, int, int]]
        Плитки из `grid()`.
    w, h : int
        Ширина и высота листа, px.

    Returns
    -------
    dict[tuple[int, int, int, int], tuple[float, float, float, float]]
        Плитка -> её зона (x0, y0, x1, y1), правая и нижняя границы не входят.
    """
    xs = _bounds(sorted({(b[0], b[2]) for b in tiles}), w)
    ys = _bounds(sorted({(b[1], b[3]) for b in tiles}), h)
    out = {}
    for b in tiles:
        (zx0, zx1), (zy0, zy1) = xs[(b[0], b[2])], ys[(b[1], b[3])]
        out[b] = (zx0, zy0, zx1, zy1)
    return out


def _bounds(spans, size):
    """Отрезки (a, b) одной оси по возрастанию -> (начало, конец) зоны; крайние зоны - до края листа."""
    cuts = [(prev[1] + cur[0]) / 2 for prev, cur in zip(spans, spans[1:], strict=False)]
    return dict(zip(spans, zip([0, *cuts], [*cuts, size], strict=True), strict=True))
