import cv2
import numpy as np
import pytest

from draftslice.core.denoise.sheet import Sheet
from draftslice.core.denoise.views import Views
from draftslice.core.thick_lines.graph import build_graph

ORIG = (250, 350)  # исходный лист, (h, w)
SCALE = 2.0  # рабочий масштаб preprocess
T = 6  # толщина контура в рабочем масштабе, px
INK = (40, 70, 110)  # цвет линий исходного листа, RGB


@pytest.fixture
def sheet_views():
    """Исходный RGB-лист: большой вид с отверстием, малый вид, открытая скобка; рабочий - он же x2."""
    paint = np.zeros(ORIG, np.uint8)
    cv2.rectangle(paint, (30, 30), (150, 130), 255, T // 2)  # большой вид
    cv2.circle(paint, (90, 80), 20, 255, T // 2)  # отверстие большого вида
    cv2.rectangle(paint, (200, 40), (260, 90), 255, T // 2)  # малый вид
    cv2.polylines(paint, [np.array([[300, 120], [280, 120], [280, 220], [300, 220]])], False, 255, T // 2)  # скобка
    axes = np.zeros(ORIG, np.uint8)  # осевые 1 px - тонкие: у вида всегда есть осевые или размеры
    cv2.line(axes, (90, 20), (90, 140), 255, 1)
    cv2.line(axes, (230, 30), (230, 100), 255, 1)
    orig = np.full((*ORIG, 3), 255, np.uint8)
    orig[paint > 0] = INK
    ink = cv2.resize(paint | axes, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist)
    med = [np.median(dist[tuple(np.rint(g.path_coordinates(i)).astype(int).T)]) for i in range(g.n_paths)]
    sheet = Sheet(ink, g, dist, np.array(med) >= T / 4, T)  # осевая 2 px в рабочем масштабе - тонкая
    return Views(sheet, orig, SCALE), orig, paint


@pytest.mark.req("VIEW-01")
def test_views_group_view_with_its_hole_and_keep_open_piece(sheet_views):
    views, _, _ = sheet_views

    assert len(views) == 3  # большой вид вместе с отверстием, малый вид, скобка - отдельный вид
    areas = [(v.box[2] - v.box[0]) * (v.box[3] - v.box[1]) for v in views]
    assert areas == sorted(areas, reverse=True)  # от крупного к мелкому


@pytest.mark.req("VIEW-01")
def test_view_box_in_original_coords_with_margin(sheet_views):
    views, _, paint = sheet_views
    big = views[0]

    margin = 3 * T / SCALE  # отступ по умолчанию: 3 толщины контура, в исходных px
    x0, y0, x1, y1 = big.box
    assert x0 == pytest.approx(30 - 1 - margin, abs=2) and y0 == pytest.approx(30 - 1 - margin, abs=2)
    assert x1 == pytest.approx(150 + 2 + margin, abs=2) and y1 == pytest.approx(130 + 2 + margin, abs=2)
    assert big.mask.shape == (y1 - y0, x1 - x0)


@pytest.mark.req("VIEW-01")
def test_view_mask_and_image_from_original(sheet_views):
    views, orig, paint = sheet_views
    big = views[0]
    x0, y0, x1, y1 = big.box

    own = paint[y0:y1, x0:x1] > 0
    m = big.mask > 0
    near = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert not (own & ~m).any()  # и контур, и отверстие целиком
    assert not (m & ~near).any()  # лишнее - выступ до 1 px у стыка с осевой; других видов в рамке нет
    assert (big.img[own] == INK).all()  # пиксели линий - из исходного листа
    assert (big.img[~m] == 255).all()  # фон белый


@pytest.mark.req("DUP-05")
def test_crops_share_sheet_scale_and_are_square(sheet_views):
    views, _, _ = sheet_views

    crops = views.crops(size=100)

    assert len(crops) == 3 and all(c.shape == (100, 100, 3) for c in crops)
    sides = [max(v.box[2] - v.box[0], v.box[3] - v.box[1]) for v in views]
    k = 100 / max(sides)  # общий множитель листа: самый крупный вид - во весь квадрат
    for c, side in zip(crops, sides, strict=True):
        ys, xs = np.nonzero((c < 250).any(-1))
        assert max(np.ptp(ys), np.ptp(xs)) + 1 <= side * k + 1
    big = np.nonzero((crops[0] < 250).any(-1))
    assert max(np.ptp(big[0]), np.ptp(big[1])) + 1 >= 0.8 * 100


@pytest.mark.req("DUP-06")
def test_rotations_are_quarter_turns_of_crops(sheet_views):
    views, _, _ = sheet_views

    crops, rots = views.crops(size=64), views.rotations(size=64)

    assert len(rots) == len(crops)
    for c, r in zip(crops, rots, strict=True):
        assert len(r) == 4
        for k in range(4):
            assert np.array_equal(r[k], np.rot90(c, k))


def _split_frame(gap):
    """Рамка вида, разорванная в верхней и нижней стороне зазором gap px (как контур, прерванный в краске листа)."""
    paint = np.zeros((200, 300), np.uint8)
    cv2.rectangle(paint, (50, 50), (250, 150), 255, T)
    paint[40:60, 150 - gap // 2 : 150 + gap - gap // 2] = 0
    paint[140:160, 150 - gap // 2 : 150 + gap - gap // 2] = 0
    orig = np.full((*paint.shape, 3), 255, np.uint8)
    orig[paint > 0] = INK
    dist = cv2.distanceTransform(paint, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(paint, dist)
    return Views(Sheet(paint, g, dist, np.ones(g.n_paths, bool), T), orig, 1.0)


@pytest.mark.req("VIEW-01")
def test_pieces_closer_than_two_max_thicknesses_are_one_view():
    # максимальная толщина листа 2 * max(dist) = T: куски ближе 2T - один вид, дальше - разные
    assert len(_split_frame(T)) == 1
    assert len(_split_frame(4 * T)) == 2
