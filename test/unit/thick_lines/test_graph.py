import cv2
import numpy as np
import pytest

from draftslice.core.thick_lines.graph import build_graph

T = 6  # толщина контура, px: dist на скелете 3


def _staggered(offset: int):
    """Толстая горизонтальная линия, тонкая сверху в x=200 и тонкая снизу в x=200+offset."""
    ink = np.zeros((300, 400), np.uint8)
    ink[147 : 147 + T, 20:380] = 255
    ink[40:147, 199:201] = 255
    ink[153:260, 199 + offset : 201 + offset] = 255
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return build_graph(ink, dist), dist


def _thin_ends_on_line(g):
    """Концы вертикальных путей, лежащие на толстой линии (y 147..152)."""
    out = []
    for i in range(g.n_paths):
        yx = np.rint(g.path_coordinates(i)).astype(int)
        if np.ptp(yx[:, 0]) > 50:  # вертикальный
            out += [tuple(p) for p in (yx[0], yx[-1]) if 145 <= p[0] <= 154]
    return out


@pytest.mark.req("KEEP-01")
def test_nodes_whose_bridge_lies_in_their_spots_merge_into_one():
    # узлы в 7 px: перемычка длиннее суммы dist (3 + 3), но целиком в пятнах узлов (по ceil(dist) = 3 точки) -
    # это один стык: обе тонкие приходят в один узел, толстая проходит его насквозь одним путём
    g, _ = _staggered(7)

    hor = [i for i in range(g.n_paths) if np.ptp(g.path_coordinates(i)[:, 1]) > 300]
    ver = [i for i in range(g.n_paths) if np.ptp(g.path_coordinates(i)[:, 0]) > 200]
    assert len(hor) == 1  # толстая - один путь насквозь
    assert len(ver) == 1  # тонкие сверху и снизу склеены через один узел
    assert not _thin_ends_on_line(g)


def test_far_nodes_stay_separate():
    g, _ = _staggered(14)

    ends = _thin_ends_on_line(g)
    assert len(ends) == 2 and ends[0] != ends[1]
