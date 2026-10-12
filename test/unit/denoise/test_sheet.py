import cv2
import numpy as np
import pytest

from draftslice.core.denoise.sheet import Sheet
from draftslice.core.thick_lines.graph import build_graph

SIZE = (500, 700)
T = 6  # толщина контура, px


def _sheet(ink, thick=None, polys=None, t_c=T):
    """Sheet по краске; thick по умолчанию - медиана dist пути не меньше T / 2 (линия cv2 толщины 2 - dist 2)."""
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist)
    if thick is None:
        med = [np.median(dist[tuple(np.rint(g.path_coordinates(i)).astype(int).T)]) for i in range(g.n_paths)]
        thick = np.array(med) >= T / 2
    if callable(thick):
        thick = thick(g)
    return Sheet(ink, g, dist, thick, t_c, polys)


def _in(mask):
    """thick-фабрика: путь толстый, если больше 90 % его пикселей в маске."""
    return lambda g: np.array(
        [(mask[tuple(np.rint(g.path_coordinates(i)).astype(int).T)] > 0).mean() > 0.9 for i in range(g.n_paths)]
    )


def _text(x, y, w=30, h=14):
    """Полигон OCR (4, 2) x, y с центром в (x, y), как OcrResult.poly."""
    return np.array([[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]])


# --- clean: сборка краски ---------------------------------------------------------------------------------------


def _contour_sheet(t):
    """Толстый замкнутый контур, толстая линия со свободными концами, тонкая сквозь контур, наконечник в контур."""
    ink = np.zeros(SIZE, np.uint8)
    contour = np.zeros(SIZE, np.uint8)
    cv2.rectangle(contour, (100, 100), (400, 300), 255, t)
    contour[200 - t // 2 : 200 + t - t // 2, 20:80] = 255  # свободные концы: квадратные торцы
    ink |= contour
    ink[150:152, 50:450] = 255  # тонкая 2 px сквозь контур
    cv2.fillPoly(ink, [np.array([[400, 200], [400 + 3 * t, 200 - t], [400 + 3 * t, 200 + t]])], 255)  # наконечник
    return ink, contour


def _lost(contour, out, t):
    """Пиксели контура, которых нет в out, кроме углов квадратных торцов свободной линии x 20..80.

    Скелет кончается в полутолщине от торца, ломаная - с круглым концом: углы торца теряются (известная проблема).
    """
    lost = (contour > 0) & (out == 0)
    lost[:, 20 : 20 + t // 2 + 1] = False
    lost[:, 80 - t // 2 - 1 : 80] = False
    return lost


@pytest.mark.req("KEEP-01", "OUT-03", "DEL-01", "DEL-03")
@pytest.mark.parametrize("t", [5, 6, 9, 14])
def test_contour_kept_whole_and_thin_line_with_arrowhead_removed(t):
    ink, contour = _contour_sheet(t)

    out = _sheet(ink, _in(contour), t_c=t).clean

    assert not _lost(contour, out, t).any()  # контур целиком, и на чётной толщине; углы прямоугольника тоже
    dt = cv2.distanceTransform((contour == 0).astype(np.uint8), cv2.DIST_L2, 5)
    # выступ у места стыка - не больше 2 px: ломаная шириной ceil(2 * median dist) + 1; тонкая линия и наконечник
    # дальше от контура не остаются
    assert dt[(out > 0) & (contour == 0)].max(initial=0) <= 2.0


@pytest.mark.req("OUT-03")
def test_free_line_end_loses_only_corners():
    ink, contour = _contour_sheet(14)

    out = _sheet(ink, _in(contour), t_c=14).clean

    lost = (contour > 0) & (out == 0)
    assert lost.any() and not _lost(contour, out, 14).any()  # теряются только углы торцов
    assert out[200, 20] and out[200, 79]  # середина торца на месте: линия не укорочена


@pytest.mark.req("DEL-04")
def test_pathless_dots_do_not_stick_to_kept_line():
    # у мелкой точки нет пути в графе: её пиксели ничем не покрыты и не должны уйти к ближайшей линии через фон
    ink = np.zeros((300, 400), np.uint8)
    ink[100:107, 50:350] = 255
    cv2.circle(ink, (200, 130), 4, 255, -1)  # рядом с линией
    cv2.circle(ink, (60, 250), 5, 255, -1)  # вдали
    sheet = _sheet(ink, lambda g: np.array([len(g.path_coordinates(i)) > 100 for i in range(g.n_paths)]))

    out = sheet.clean

    assert out[103, 100] and not out[130, 200] and not out[250, 60]


def test_all_thick_keeps_contour_and_none_gives_empty():
    ink, contour = _contour_sheet(9)

    out = _sheet(ink, lambda g: np.ones(g.n_paths, bool)).clean

    assert not ((out > 0) & (ink == 0)).any() and not _lost(contour, out, 9).any()
    assert not _sheet(ink, lambda g: np.zeros(g.n_paths, bool)).clean.any()


def test_clean_is_uint8_mask_inside_ink():
    ink, contour = _contour_sheet(6)

    out = _sheet(ink, _in(contour)).clean

    assert out.dtype == np.uint8 and out.shape == ink.shape
    assert set(np.unique(out)) <= {0, 255}
    assert not ((out > 0) & (ink == 0)).any()


def test_remove_rebuilds_clean_and_chains():
    ink, contour = _contour_sheet(6)
    sheet = _sheet(ink, _in(contour))
    before = sheet.clean

    same = sheet.remove(sheet.keep)  # снять всё, что оставалось

    assert same is sheet
    assert before.any() and not sheet.clean.any()
    assert not sheet.keep.any()


def test_bad_inputs_raise():
    ink, _ = _contour_sheet(6)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist)

    with pytest.raises(ValueError):
        Sheet(ink, g, dist, np.ones(g.n_paths - 1, bool), T)
    with pytest.raises(ValueError):
        Sheet(ink, g, dist[:-1], np.ones(g.n_paths, bool), T)


# --- arrows ------------------------------------------------------------------------------------------------------


@pytest.mark.req("DEL-03")
def test_trim_mask_in_original_scale_removes_path_wholly_inside():
    # лист в рабочем масштабе 2x; маска стрелок - в исходном (вдвое меньше)
    ink = np.zeros((300, 400), np.uint8)
    ink[100:106, 50:350] = 255  # линия целиком под маской
    ink[294:300, 50:350] = 255  # линия в последнем ряду листа: при округлении индекс вышел бы за маску
    mask = np.zeros((150, 200), np.uint8)
    mask[45:60, 20:180] = 255
    sheet = _sheet(ink, lambda g: np.ones(g.n_paths, bool))

    sheet.trim(mask)

    assert sheet.keep.sum() == len(sheet) - 1
    assert not sheet.clean[100:106].any() and sheet.clean[294:300].any()


@pytest.mark.req("DEL-03", "OUT-03")
def test_trim_cuts_arrowhead_off_line_end_and_keeps_line():
    # ось с наконечником - один путь: стрелка занимает больше половины пути, но линию снимать нельзя
    ink = np.zeros((200, 400), np.uint8)
    ink[97:103, 100:220] = 255  # линия 120 px
    cv2.fillPoly(ink, [np.array([[220, 88], [380, 100], [220, 112]])], 255)  # наконечник 160 px
    mask = np.zeros_like(ink)
    cv2.fillPoly(mask, [np.array([[215, 86], [384, 100], [215, 114]])], 255)
    sheet = _sheet(ink, lambda g: np.ones(g.n_paths, bool))

    out = sheet.trim(mask).clean

    assert out[97:103, 101:200].all()  # линия, кроме угла свободного торца
    assert not out[:, 260:].any()  # наконечника нет


@pytest.mark.req("OUT-03")
def test_trim_keeps_contour_under_arrow_tip_in_its_middle():
    ink, contour = _contour_sheet(6)
    mask = np.zeros_like(ink)
    cv2.circle(mask, (400, 200), 8, 255, -1)  # маска острия наконечника заходит на середину стороны контура
    sheet = _sheet(ink, _in(contour))
    span = sheet.span.copy()

    out = sheet.trim(mask).clean

    assert np.array_equal(sheet.span[sheet.keep], span[sheet.keep])  # середина пути не режется
    assert not _lost(contour, out, 6).any()


# --- tables ------------------------------------------------------------------------------------------------------


def _table(ink, x0, y0, cols, rows, cw=80, rh=40):
    """Таблица: рамка толщины T, внутренние линии 2 px. Возвращает центры ячеек."""
    x1, y1 = x0 + cols * cw, y0 + rows * rh
    cv2.rectangle(ink, (x0, y0), (x1, y1), 255, T)
    for c in range(1, cols):
        cv2.line(ink, (x0 + c * cw, y0), (x0 + c * cw, y1), 255, 2)
    for r in range(1, rows):
        cv2.line(ink, (x0, y0 + r * rh), (x1, y0 + r * rh), 255, 2)
    return [(x0 + (c + 0.5) * cw, y0 + (r + 0.5) * rh) for r in range(rows) for c in range(cols)]


def _view(ink):
    """Прямоугольный вид с круглым отверстием; число размера - внутри контура."""
    cv2.rectangle(ink, (420, 300), (650, 460), 255, T)
    cv2.circle(ink, (590, 380), 30, 255, T)
    return [_text(480, 380)]


@pytest.mark.req("DEL-07")
def test_table_with_text_in_cells_is_found_whole():
    ink = np.zeros(SIZE, np.uint8)
    cells = _table(ink, 40, 40, cols=3, rows=2)
    table = ink.copy()
    polys = [_text(x, y) for x, y in cells] + _view(ink)
    sheet = _sheet(ink, polys=polys)

    found = sheet.tables()

    assert found.any()
    assert np.array_equal(found, _in(table)(sheet.g))  # все пути таблицы, и только они


@pytest.mark.req("DEL-07")
def test_table_with_some_empty_cells_is_found():
    ink = np.zeros(SIZE, np.uint8)
    cells = _table(ink, 40, 40, cols=3, rows=2)

    assert _sheet(ink, polys=[_text(x, y) for x, y in cells[:4]]).tables().all()  # 2 из 6 ячеек пустые


@pytest.mark.req("KEEP-01")
def test_view_with_dimension_inside_is_not_table():
    ink = np.zeros(SIZE, np.uint8)
    polys = _view(ink)

    assert not _sheet(ink, polys=polys).tables().any()


@pytest.mark.req("KEEP-01")
def test_grid_of_pockets_without_text_is_not_table():
    ink = np.zeros(SIZE, np.uint8)
    _table(ink, 40, 40, cols=3, rows=2)

    assert not _sheet(ink, polys=[_text(600, 450)]).tables().any()  # текст есть, но не в ячейках


def test_single_cell_with_text_is_not_table():
    # кружок позиции и рамка вокруг одного текста - не таблица (DEL-04, правило marks)
    ink = np.zeros(SIZE, np.uint8)
    cv2.circle(ink, (150, 300), 30, 255, 2)
    cv2.rectangle(ink, (300, 280), (420, 320), 255, 2)

    assert not _sheet(ink, polys=[_text(150, 300, 20, 20), _text(360, 300)]).tables().any()


def test_no_text_no_tables():
    ink = np.zeros(SIZE, np.uint8)
    ink[100:110, 100:400] = 255
    sheet = _sheet(ink)

    assert sheet.tables().shape == (len(sheet),) and not sheet.tables().any()


# --- marks -------------------------------------------------------------------------------------------------------


def _view_with_lines(ink):
    """Вид: прямоугольник с отверстием, две выносные и осевая сквозь вид. Возвращает маску контура вида."""
    view = np.zeros(SIZE, np.uint8)
    cv2.rectangle(view, (200, 150), (450, 330), 255, T)
    cv2.circle(view, (325, 240), 40, 255, T)
    ink |= view
    cv2.line(ink, (200, 330), (200, 420), 255, 2)  # выносные
    cv2.line(ink, (450, 330), (450, 420), 255, 2)
    cv2.line(ink, (180, 240), (470, 240), 255, 2)  # осевая сквозь вид и отверстие
    return view


def _bracket(ink):
    """Открытая скобка бокового вида (3 изгиба) с тонкой выносной. Возвращает маску скобки."""
    bracket = np.zeros(SIZE, np.uint8)
    cv2.polylines(bracket, [np.array([[620, 300], [660, 300], [660, 470], [620, 470], [620, 450]])], False, 255, T)
    ink |= bracket
    cv2.line(ink, (660, 385), (695, 385), 255, 2)  # выносная: у вида всегда есть тонкие линии
    return bracket


@pytest.mark.req("DEL-04", "DEL-05", "KEEP-01", "KEEP-02", "OUT-03")
def test_simple_open_figures_outside_views_removed_views_and_holes_kept():
    # вне вида: знак √ (2 изгиба), полка выноски (1 изгиб), штрих разреза (прямой) - мусор;
    # открытая скобка с 3 изгибами - сложнее, остаётся; внутри вида - отверстие остаётся, мелкая чёрточка уходит
    ink = np.zeros(SIZE, np.uint8)
    view = _view_with_lines(ink)
    marks = np.zeros(SIZE, np.uint8)
    cv2.polylines(marks, [np.array([[60, 80], [75, 110], [100, 40], [170, 40]])], False, 255, T)  # знак √
    cv2.polylines(marks, [np.array([[500, 60], [600, 60], [630, 30]])], False, 255, T)  # полка выноски
    marks[400:460, 560 : 560 + T] = 255  # штрих линии разреза
    marks[180 : 180 + T, 225:247] = 255  # чёрточка внутри вида, 22 px
    ink |= marks
    bracket = _bracket(ink)
    sheet = _sheet(ink, polys=[_text(100, 450, 120, 50)])  # высота текста 50 px

    found = sheet.marks()
    out = sheet.remove(found).clean

    assert not (found & ~_sheet(ink).keep).any()  # удаляются только оставленные пути
    assert not ((view > 0) & (out == 0)).any()  # вид и отверстие целиком
    assert not ((bracket > 0) & (out == 0)).any()  # скобка: 3 изгиба - фигура сложнее обозначения
    assert not ((marks > 0) & (out > 0)).any()


def test_segments_describe_figures():
    ink = np.zeros(SIZE, np.uint8)
    _view_with_lines(ink)
    cv2.polylines(ink, [np.array([[60, 80], [75, 110], [100, 40], [170, 40]])], False, 255, T)  # √
    sheet = _sheet(ink, polys=[_text(100, 450, 120, 50)])

    seg = sheet.segments
    k_view = seg.labels[200, 200]  # левая сторона прямоугольника
    k_hole = seg.labels[240, 285]  # левый край отверстия - отдельный сегмент
    k_sign = seg.labels[40, 140]  # полка знака

    assert seg.holes[k_view] == 1 and seg.view[k_view] and not seg.circle[k_view]
    assert seg.circle[k_hole] and not seg.view[k_hole] and seg.inside[k_hole]  # отверстие: круг внутри вида
    assert seg.holes[k_sign] == 0 and seg.bends[k_sign] == 2 and not seg.view[k_sign] and not seg.inside[k_sign]


@pytest.mark.req("KEEP-01")
def test_view_with_two_thin_lines_is_not_mark():
    ink = np.zeros(SIZE, np.uint8)
    cv2.rectangle(ink, (200, 150), (450, 330), 255, T)
    cv2.line(ink, (200, 330), (200, 420), 255, 2)
    cv2.line(ink, (450, 330), (450, 420), 255, 2)

    assert not _sheet(ink).marks().any()


@pytest.mark.req("DEL-04", "KEEP-02")
def test_small_fragment_with_many_thin_bits_removed_small_hole_in_view_kept():
    # обрывок кружка позиции, слипшийся с буквой: толстый кусок, к нему - дуги кружка и штрихи буквы (3+ тонких);
    # меньше высоты текста и вне вида - мусор. Отверстие меньше высоты текста, но внутри вида - часть вида.
    ink = np.zeros(SIZE, np.uint8)
    view = _view_with_lines(ink)  # отверстие радиуса 40: 86 px
    frag = np.zeros(SIZE, np.uint8)
    frag[100:124, 600 : 600 + T] = 255  # толстый обрывок 24 px
    ink |= frag
    for x1, y1 in ((640, 80), (640, 140), (580, 90)):  # три тонкие «дуги» вокруг обрывка
        cv2.line(ink, (603, 112), (x1, y1), 255, 2)
    polys = [_text(100, 450, 120, 100)]  # высота текста 100 px: больше обрывка и отверстия
    sheet = _sheet(ink, polys=polys)

    out = sheet.remove(sheet.marks()).clean

    assert sheet.text_h == pytest.approx(100)
    assert not ((frag > 0) & (out > 0)).any()
    assert not ((view > 0) & (out == 0)).any()


def test_no_text_no_size_rule():
    ink = np.zeros(SIZE, np.uint8)
    _view_with_lines(ink)

    assert _sheet(ink).text_h == 0


@pytest.mark.req("DEL-04", "KEEP-02")
def test_circle_and_text_frame_outside_view_removed_hole_inside_kept():
    ink = np.zeros(SIZE, np.uint8)
    view = _view_with_lines(ink)  # отверстие - круг внутри вида
    marks = np.zeros(SIZE, np.uint8)
    cv2.circle(marks, (580, 90), 30, 255, T)  # кружок позиции вне вида
    cv2.polylines(marks, [np.array([[80, 140], [130, 60], [180, 140]])], True, 255, T)  # треугольник вокруг текста
    ink |= marks
    sheet = _sheet(ink, polys=[_text(130, 115, 30, 14)])  # текст внутри треугольника; высота текста 14 px

    out = sheet.remove(sheet.marks()).clean

    assert not ((view > 0) & (out == 0)).any()
    assert not ((marks > 0) & (out > 0)).any()


@pytest.mark.req("DEL-05", "KEEP-01")
def test_complex_open_figure_without_thin_lines_removed():
    # буква обозначения вида крупнее текста размеров, OCR её не нашёл (лист 20: «Ж», «И»): открытая, больше 2
    # изгибов, но к ней не подходит ни одна тонкая линия - не скобка вида
    ink = np.zeros(SIZE, np.uint8)
    view = _view_with_lines(ink)
    letter = np.zeros(SIZE, np.uint8)
    cv2.polylines(letter, [np.array([[560, 40], [580, 120], [600, 40], [620, 120], [640, 40]])], False, 255, T)
    ink |= letter
    sheet = _sheet(ink, polys=[_text(100, 450, 40, 30)])  # высота текста 30 px - буква крупнее

    sg = sheet.segments
    k = sg.labels[80, 570]  # середина первого штриха
    out = sheet.remove(sheet.marks()).clean

    assert sg.holes[k] == 0 and sg.bends[k] > sheet.max_bends and sg.n_thin[k] == 0 and not sg.small[k]
    assert not ((letter > 0) & (out > 0)).any()
    assert not ((view > 0) & (out == 0)).any()


@pytest.mark.req("DEL-05", "KEEP-01")
def test_letter_with_loops_outside_view_removed():
    # крупная буква с петлями (обозначение разреза «В»), которую OCR не нашёл: замкнутые контуры без текста и не
    # мелкая - но к ней не подходит ни одна тонкая линия, это не вид
    ink = np.zeros(SIZE, np.uint8)
    view = _view_with_lines(ink)
    letter = np.zeros(SIZE, np.uint8)
    cv2.rectangle(letter, (560, 60), (600, 95), 255, T)  # верхняя петля
    cv2.rectangle(letter, (560, 95), (605, 135), 255, T)  # нижняя петля
    ink |= letter
    sheet = _sheet(ink, polys=[_text(100, 450, 40, 30)])  # высота текста 30 px - буква крупнее

    sg = sheet.segments
    k = sg.labels[100, 562]
    out = sheet.remove(sheet.marks()).clean

    assert sg.free[k] > 0 and not sg.small[k] and sg.n_thin[k] == 0 and not sg.view[k]
    assert not ((letter > 0) & (out > 0)).any()
    assert not ((view > 0) & (out == 0)).any()


def test_rule_thresholds_are_sheet_parameters():
    ink = np.zeros(SIZE, np.uint8)
    _view_with_lines(ink)
    _bracket(ink)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist)
    med = [np.median(dist[tuple(np.rint(g.path_coordinates(i)).astype(int).T)]) for i in range(g.n_paths)]
    thick = np.array(med) >= T / 2

    def out(**kw):
        sheet = Sheet(ink, g, dist, thick, T, [_text(100, 450, 120, 50)], **kw)
        return sheet.remove(sheet.marks()).clean

    assert out()[300:470, 620:666].any()  # скобка: 3 изгиба > max_bends=2 - остаётся
    assert not out(max_bends=3)[300:470, 620:666].any()  # с max_bends=3 - обозначение


@pytest.mark.req("DEL-03")
def test_trim_removes_arrowhead_poking_out_of_mask():
    # маска модели короче наконечника: остриё и основание высовываются на несколько пикселей
    ink = np.zeros((200, 400), np.uint8)
    ink[97:103, 40:150] = 255  # линия
    cv2.fillPoly(ink, [np.array([[150, 90], [260, 100], [150, 110]])], 255)  # наконечник
    mask = np.zeros_like(ink)
    cv2.fillPoly(mask, [np.array([[156, 90], [252, 100], [156, 110]])], 255)  # без кончика и основания
    sheet = _sheet(ink, lambda g: np.ones(g.n_paths, bool))

    out = sheet.trim(mask).clean

    assert out[97:103, 41:140].all()  # линия, кроме угла свободного торца
    assert not out[:, 170:].any()  # наконечника нет вместе с высунутым кончиком


def test_path_belongs_to_segment_of_most_of_its_pixels_not_of_its_head():
    # первый пиксель пути может попасть в крошку у узла (отдельный мелкий сегмент): по нему путь с длинной дугой
    # снимался бы вместе с крошкой
    ink = np.zeros(SIZE, np.uint8)
    _view_with_lines(ink)
    sheet = _sheet(ink, polys=[_text(100, 450, 120, 50)])
    sg = sheet.segments
    i = int(np.argmax([len(c) if sheet.keep[j] else 0 for j, c in enumerate(sheet.coords)]))  # самый длинный путь вида
    y, x = sheet.coords[i][0]
    lab = sg.labels.copy()
    crumb = lab.max() + 1
    lab[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = crumb  # первый пиксель пути - в отдельной крошке
    sheet.__dict__["segments"] = type(sg)(
        lab,
        np.vstack([sg.stats, [x - 1, y - 1, 3, 3, 9]]),
        *[np.append(f, v) for f, v in ((sg.holes, 0), (sg.free, 0), (sg.bends, 0))],
        *[np.append(f, v) for f, v in ((sg.small, True), (sg.circle, False))],
        np.append(sg.n_thin, 0),
        np.append(sg.view, False),
        np.append(
            sg.owner,
            sg.owner[
                sg.labels[sheet.coords[i][len(sheet.coords[i]) // 2][0], sheet.coords[i][len(sheet.coords[i]) // 2][1]]
            ],
        ),
        np.append(sg.inside, True),
    )

    assert not sheet.marks()[i]
