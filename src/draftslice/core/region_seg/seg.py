"""Сегментация чертежа на большие сегменты - виды детали.

`segment()` - единственная публичная функция файла, шаги - приватные функции:

1. `_edge_lines` - длинные прямые линии с краю листа (рамка, полосы скана). Удаляются по
   пикселам, а не по компоненте: вид, чьи выносные касаются рамки, остаётся целым.
2. `_find_seeds` - связные компоненты; seed - толстая (короткая сторона bbox) и крупная
   (площадь краски) компонента, не рамка.
3. `_fill_seeds` - заливка внешних контуров seed'ов; раскинутые (сноска с выносками)
   отсеиваются по доле бокса под заливкой.
4. `_merge_nested` - меньший сегмент, чей бокс в основном внутри бокса большего, сливается с ним.
"""

import cv2
import numpy as np

from draftslice.core.preprocess import preprocess
from draftslice.core.types import Array, Array2D, Mat
from draftslice.core.validators import validate_image


def _edge_hor(lines: Mat, content: Mat) -> Array2D[np.bool_]:
    """Горизонтальные длинные линии, между которыми и верхним или нижним краем нет содержимого.

    Проверка идёт только в колонках самой линии. Содержимое ближе собственной толщины линии
    не считается: это её же углы и зубцы, не прошедшие в открытие ядром высотой в 1 пиксел.
    """
    img_h = lines.shape[0]
    has = content > 0
    any_col = has.any(axis=0)
    col_top = np.where(any_col, has.argmax(axis=0), img_h)
    col_bottom = np.where(any_col, img_h - 1 - has[::-1].argmax(axis=0), -1)

    _, lbl, stats, _ = cv2.connectedComponentsWithStats(lines)
    edge = [
        i
        for i, (x, y, w, h, _) in enumerate(stats[1:], start=1)
        if col_top[x : x + w].min() >= y - h or col_bottom[x : x + w].max() <= y + 2 * h - 1
    ]
    return np.isin(lbl, edge)


def _edge_lines(binary: Mat, sheet_cov: float) -> Array2D[np.bool_]:
    """Маска длинных прямых линий, между которыми и краем листа нет содержимого."""
    img_h, img_w = binary.shape
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(sheet_cov * img_w), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(sheet_cov * img_h)))
    hor_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hor_kernel)
    ver_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ver_kernel)
    content = cv2.subtract(binary, cv2.bitwise_or(hor_lines, ver_lines))

    # Горизонтальные и вертикальные проверяются раздельно: вместе рамка стала бы одной
    # компонентой с боксом на весь лист. Вертикальные - та же проверка на транспонированном листе.
    ver_edge = _edge_hor(np.ascontiguousarray(ver_lines.T), np.ascontiguousarray(content.T)).T
    return _edge_hor(hor_lines, content) | ver_edge


def _find_seeds(
    binary: Mat, sheet_cov: float, thick_factor: float, area_factor: float
) -> tuple[Array2D[np.int32], Array[np.intp]]:
    """Карта меток компонент и метки seed'ов (1-based, 0 - фон)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    img_h, img_w = binary.shape
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]

    # Толщина - короткая сторона bbox: у слитного рукописного слова она равна высоте шрифта,
    # как бы длинно слово ни было, и не зависит от поворота надписи.
    thickness = np.minimum(w, h)
    thick_thresh = thick_factor * thickness.mean()
    # Площадь краски отсекает крупные одиночные буквы, прошедшие по толщине.
    area_thresh = area_factor * area.mean()

    is_frame = (w >= sheet_cov * img_w) & (h >= sheet_cov * img_h)
    is_seed = (thickness > thick_thresh) & (area > area_thresh) & ~is_frame
    return labels, np.flatnonzero(is_seed) + 1


def _fill_seeds(labels: Array2D[np.int32], seeds: Array[np.intp], min_extent: float) -> Mat:
    """Залитые внешние контуры сплошных seed'ов: 1 - сегмент, 0 - фон."""
    seed_mask = np.isin(labels, seeds).astype(np.uint8)
    cnts, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Доля бокса под залитым контуром: вид после заливки сплошной, сноска с выносками - нет.
    solid = []
    for c in cnts:
        _, _, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > min_extent * w * h:
            solid.append(c)

    filled = np.zeros_like(seed_mask)
    cv2.drawContours(filled, solid, -1, 1, cv2.FILLED)
    return filled


def _box_inside(x: Array[np.int32], y: Array[np.int32], w: Array[np.int32], h: Array[np.int32]) -> Array2D[np.float64]:
    """Доля бокса i внутри бокса j: матрица (n, n), 1.0 - i целиком внутри j."""
    inter_w = np.maximum(0, np.minimum(x[:, None] + w[:, None], x + w) - np.maximum(x[:, None], x))
    inter_h = np.maximum(0, np.minimum(y[:, None] + h[:, None], y + h) - np.maximum(y[:, None], y))
    return inter_w * inter_h / (w * h)[:, None]


def _merge_nested(filled: Mat, inside: float) -> Array2D[np.uint16]:
    """Карта больших сегментов: меньший, чей бокс в основном внутри большего, сливается с ним."""
    num, seg, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8, ltype=cv2.CV_16U)
    x = stats[1:, cv2.CC_STAT_LEFT]
    y = stats[1:, cv2.CC_STAT_TOP]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    ins = _box_inside(x, y, w, h)

    # Сливаем только меньший в больший - иначе пара выберет друг друга и поменяется метками.
    area = w * h
    ins[area[:, None] >= area] = 0
    partner = ins.argmax(axis=1)
    merge = ins.max(axis=1) > inside

    lut = np.arange(num, dtype=np.uint16)
    lut[1:][merge] = partner[merge] + 1
    return lut[seg]


@validate_image(channels=3)
def segment(
    img: Mat,
    sheet_cov: float = 0.85,
    thick_factor: float = 5.0,
    area_factor: float = 2.0,
    min_extent: float = 0.1,
    inside: float = 0.5,
) -> Array2D[np.uint16]:
    """Размечает большие сегменты (виды детали) на чертеже.

    Parameters
    ----------
    img : Mat
        Изображение чертежа, RGB.
    sheet_cov : float
        Доля стороны листа, начиная с которой прямая линия считается линией листа (рамка,
        полоса скана); она удаляется, если лежит с краю листа.
    thick_factor : float
        Seed толще `thick_factor` * средняя толщина компонент листа.
    area_factor : float
        Seed крупнее `area_factor` * средняя площадь краски компонент листа.
    min_extent : float
        Seed остаётся, если его залитый контур занимает больше такой доли своего бокса.
    inside : float
        Меньший сегмент сливается с большим, если больше такой доли его бокса лежит внутри
        бокса большего.

    Returns
    -------
    Array2D[np.uint16]
        Карта сегментов размера `img`: номер большого сегмента на пикселах краски, лежащих
        внутри его залитого контура, 0 - всё остальное. Номера 1-based и могут идти с пропусками
        после слияния вложенных.

    Raises
    ------
    DraftsliceTypeError
        Если `img` не трёхканальное.
    """
    binary = preprocess(img, target=None).binary
    binary[_edge_lines(binary, sheet_cov)] = 0
    empty = np.zeros(binary.shape, dtype=np.uint16)
    if not binary.any():
        return empty

    labels, seeds = _find_seeds(binary, sheet_cov, thick_factor, area_factor)
    filled = _fill_seeds(labels, seeds, min_extent)
    if not filled.any():
        return empty

    return _merge_nested(filled, inside) * (binary > 0)
