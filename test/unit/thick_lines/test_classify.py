import cv2
import numpy as np
import pytest
from skan import Skeleton

from draftslice.core.thick_lines.classify import classify_lines
from draftslice.core.thick_lines.graph import build_graph

THIN, MID, CONTOUR = 2, 4, 6  # толщины линий синтетического листа, px
SIZE = (600, 800)


def _box(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int, t: int) -> None:
    """Прямоугольный контур шириной t px, целиком внутри [y0, y1) x [x0, x1)."""
    ink[y0 : y0 + t, x0:x1] = 255
    ink[y1 - t : y1, x0:x1] = 255
    ink[y0:y1, x0 : x0 + t] = 255
    ink[y0:y1, x1 - t : x1] = 255


@pytest.fixture
def sheet():
    """Фабрика графа синтетического листа 600x800: рамка 4 px по краю, контур 6 px, четыре тонкие по 2 px.

    Рамка и контур не касаются тонких линий. Рамка охватывает лист целиком, контур и тонкие - нет.
    """

    def make(
        *, frame: bool = True, contour: bool = True, thin: bool = True, bump: bool = False
    ) -> tuple[Skeleton, np.ndarray]:
        h, w = SIZE
        ink = np.zeros(SIZE, np.uint8)
        if frame:
            _box(ink, 10, h - 10, 10, w - 10, MID)
        if contour:
            _box(ink, 100, 300, 150, 650, CONTOUR)
        if thin:
            for y in (350, 400, 450, 500):
                ink[y : y + THIN, 150:650] = 255
        if bump:  # короткий обрывок уровня MID: слабый максимум плотности, не горка
            ink[550 : 550 + MID, 700:740] = 255
        dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return build_graph(ink, dist), dist

    return make


def _near_border(g, i: int, margin: int = 15) -> bool:
    """Путь i целиком лежит у края листа."""
    yx = g.path_coordinates(i)
    d = np.minimum(np.minimum(yx[:, 0], SIZE[0] - 1 - yx[:, 0]), np.minimum(yx[:, 1], SIZE[1] - 1 - yx[:, 1]))
    return bool((d <= margin).all())


def _mean_y(g) -> np.ndarray:
    return np.array([g.path_coordinates(i)[:, 0].mean() for i in range(g.n_paths)])


@pytest.mark.req("KEEP-01", "DEL-01")
def test_contour_thick_thin_lines_thin(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR)

    y = _mean_y(g)
    assert cls.thick[~cls.frame & (y < 320)].all()
    assert not cls.thick[~cls.frame & (y > 340)].any()


@pytest.mark.req("KEEP-01")
def test_threshold_lies_between_thin_and_contour_humps(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert cls.thr is not None
    assert THIN + 0.5 < cls.thr < CONTOUR - 0.5


def test_frame_is_marked_and_gives_no_hump(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR)

    on_border = np.array([_near_border(g, i) for i in range(g.n_paths)])
    assert cls.frame.any()
    assert (cls.frame == on_border).all()
    assert np.allclose(sorted(cls.peaks), [THIN, CONTOUR], atol=0.7)  # горки рамки (4 px) нет


def test_masses_sum_to_one_and_contour_points_to_its_hump(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert sum(cls.masses) == pytest.approx(1.0)
    assert cls.peaks[cls.contour] == pytest.approx(CONTOUR, abs=0.7)
    assert cls.masses[cls.contour] == pytest.approx(1380 / (1380 + 4 * 500), abs=0.05)  # периметр контура / вся длина


def test_small_bump_is_weak_not_hump(sheet):
    g, dist = sheet(bump=True)

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert len(cls.peaks) == 2
    assert len(cls.weak) >= 1


@pytest.mark.req("KEEP-01")
def test_no_hump_below_contour_keeps_everything_thick(sheet):
    g, dist = sheet(thin=False)

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert cls.thr is None
    assert cls.thick.all()


def test_frame_span_is_a_parameter(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR, frame_span=1.0)  # рамка охватывает 96 % листа: при 1.0 не рамка

    assert not cls.frame.any()
    assert len(cls.peaks) == 3  # без исключения рамки её уровень 4 px даёт свою горку


def test_level_and_length_are_float32(sheet):
    g, dist = sheet()

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert cls.level.dtype == np.float32
    assert cls.length.dtype == np.float32


@pytest.mark.req("KEEP-01")
def test_only_frame_gives_no_humps_and_keeps_it_thick(sheet):
    g, dist = sheet(contour=False, thin=False)

    cls = classify_lines(g, dist, t_c=CONTOUR)

    assert cls.frame.all()
    assert cls.thr is None
    assert cls.contour is None
    assert len(cls.peaks) == 0
    assert cls.thick.all()


@pytest.mark.parametrize(
    "kw",
    [
        {"t_c": 0.0},
        {"t_c": CONTOUR, "prominence": 0.0},
        {"t_c": CONTOUR, "prominence": 1.0},
        {"t_c": CONTOUR, "bandwidth": 0.0},
        {"t_c": CONTOUR, "frame_span": 0.0},
        {"t_c": CONTOUR, "frame_span": 1.5},
    ],
    ids=["t_c", "prominence_zero", "prominence_one", "bandwidth", "frame_span_zero", "frame_span_over_one"],
)
def test_invalid_params_raise(sheet, kw):
    g, dist = sheet()

    with pytest.raises(ValueError):
        classify_lines(g, dist, **kw)


def _path_at(g, y, x):
    """Номер пути, проходящего ближе всего к пикселю (y, x)."""
    d = [np.hypot(*(np.rint(g.path_coordinates(i)) - (y, x)).T).min() for i in range(g.n_paths)]
    return int(np.argmin(d))


@pytest.mark.req("KEEP-01", "OUT-03")
def test_short_thin_piece_between_thick_pieces_of_same_line_is_thick():
    # перетяжка контура: кусок 10 px тонкой толщины между толстыми кусками той же линии (на скане - короткий
    # кусок между узлами штриховки, где скелет намерил тонкую). Штрих штриховки под 45° между двумя толстыми
    # параллельными линиями той же длины - тонкий: он не продолжает толстые линии.
    h, w = SIZE
    ink = np.zeros(SIZE, np.uint8)
    _box(ink, 10, h - 10, 10, w - 10, MID)  # рамка
    ink[150 : 150 + CONTOUR, 100:300] = 255  # контур слева от перетяжки
    ink[152 : 152 + THIN, 300:312] = 255  # перетяжка
    ink[150 : 150 + CONTOUR, 312:600] = 255  # контур справа
    for x in (300, 312):
        ink[100:200, x : x + THIN] = 255  # тонкие поперёк контура на краях перетяжки: узлы режут линию
    ink[300 : 300 + CONTOUR, 100:600] = 255  # две толстые параллельные с зазором 6 px
    ink[312 : 312 + CONTOUR, 100:600] = 255
    cv2.line(ink, (350, 312), (362, 305), 255, 1)  # штрих штриховки: 16 px - короче трёх толщин, под углом
    for y in (400, 450, 500):
        ink[y : y + THIN, 150:650] = 255  # тонкие: горка тонких
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist)

    cls = classify_lines(g, dist, CONTOUR)

    neck = _path_at(g, 153, 306)
    hatch = _path_at(g, 309, 356)
    assert cls.level[neck] < cls.thr  # своя толщина - тонкая
    assert cls.thick[neck] and cls.bridge[neck]  # но это разрыв толстой линии
    assert not cls.thick[hatch] and not cls.bridge[hatch]
    assert not cls.thick[_path_at(g, 401, 400)]
