import cv2
import numpy as np
import pytest

from draftslice.core.exceptions import DraftsliceTypeError
from draftslice.core.preprocess import preprocess
from draftslice.core.region_seg.seg import _edge_lines, segment

BLACK = (0, 0, 0)


def _n_segments(seg: np.ndarray) -> int:
    return len(np.unique(seg)) - 1


@pytest.mark.req("KEEP-01", "DEL-06")
def test_view_is_one_segment_without_text(sheet):
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)

    seg = segment(sheet)

    assert _n_segments(seg) == 1
    assert seg[100, 200] > 0  # контур вида
    assert not seg[520:, :].any()  # «текст» внизу листа в сегмент не попал


@pytest.mark.req("KEEP-01")
@pytest.mark.parametrize("n_views", [1, 2, 3])
def test_separate_views_stay_separate(sheet, n_views):
    for i in range(n_views):
        x = 60 + i * 240
        cv2.rectangle(sheet, (x, 60), (x + 180, 200), BLACK, 3)

    assert _n_segments(segment(sheet)) == n_views


@pytest.mark.req("VIEW-02")
def test_text_only_sheet_has_no_segments(sheet):
    assert not segment(sheet).any()


def test_blank_sheet_has_no_segments():
    blank = np.full((200, 300, 3), 255, np.uint8)

    seg = segment(blank)

    assert seg.shape == (200, 300)
    assert not seg.any()


@pytest.mark.req("DEL-07", "OUT-03")
def test_view_touching_frame_through_leader_survives(sheet):
    cv2.rectangle(sheet, (5, 5), (794, 594), BLACK, 2)
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)
    cv2.line(sheet, (300, 175), (794, 175), BLACK, 1)  # выноска упирается в рамку

    seg = segment(sheet)

    assert _n_segments(seg) == 1
    assert seg[100, 200] > 0
    assert not seg[5, :].any()  # верхняя линия рамки удалена


@pytest.mark.req("DEL-07")
def test_edge_lines_take_frame_but_keep_inner_long_line(sheet):
    cv2.rectangle(sheet, (5, 5), (794, 594), BLACK, 2)
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)
    cv2.line(sheet, (10, 300), (790, 300), BLACK, 1)  # длинная размерная между видом и текстом

    edge = _edge_lines(preprocess(sheet, target=None).binary, sheet_cov=0.95)

    assert edge[5, 400]
    assert not edge[300, 400]


def test_callout_with_long_leaders_is_not_segment(sheet):
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)
    # Сноска: кружок и две длинные выноски - бокс большой, а заливка почти пустая.
    cv2.circle(sheet, (420, 80), 15, BLACK, 2)
    cv2.line(sheet, (435, 80), (750, 80), BLACK, 2)
    cv2.line(sheet, (750, 80), (750, 400), BLACK, 2)

    assert _n_segments(segment(sheet)) == 1


def test_small_view_inside_box_of_bigger_merges(sheet):
    # U-образный вид: маленький вид лежит в его вырезе - внутри бокса, но не внутри заливки.
    u = np.array([(400, 60), (430, 60), (430, 230), (570, 230), (570, 60), (600, 60), (600, 260), (400, 260)])
    cv2.polylines(sheet, [u], isClosed=True, color=BLACK, thickness=2)
    cv2.rectangle(sheet, (460, 100), (540, 180), BLACK, 2)

    assert _n_segments(segment(sheet)) == 1


@pytest.mark.req("OUT-03")
def test_light_axis_crossing_contour_does_not_break_it(sheet):
    cv2.rectangle(sheet, (200, 100), (400, 250), (0, 0, 255), 3)
    # Светлая оранжевая осевая поверх контура - штрихи ровно на пересечениях.
    for y0, y1 in [(70, 90), (95, 115), (235, 255), (260, 280)]:
        cv2.line(sheet, (300, y0), (300, y1), (255, 165, 0), 2)

    assert _n_segments(segment(sheet)) == 1


def test_grayscale_input_raises():
    with pytest.raises(DraftsliceTypeError):
        segment(np.zeros((10, 10), np.uint8))
