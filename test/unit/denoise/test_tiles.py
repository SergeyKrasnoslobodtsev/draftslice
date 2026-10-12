import numpy as np
import pytest

from draftslice.core.denoise.tiles import grid, zones


@pytest.mark.parametrize(("h", "w"), [(150, 200), (64, 110), (40, 30), (640, 640)])
def test_grid_covers_sheet_and_last_tile_is_pushed_to_edge(h, w):
    tiles = grid(h, w, 64, 0.2)

    hits = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in tiles:
        assert (x1 - x0, y1 - y0) == (min(64, w), min(64, h))  # плитка не выходит за лист и не меньше нужного
        hits[y0:y1, x0:x1] = True
    assert hits.all()
    assert max(t[2] for t in tiles) == w and max(t[3] for t in tiles) == h


def test_grid_overlap_is_share_of_tile():
    xs = sorted({(x0, x1) for x0, _, x1, _ in grid(64, 300, 64, 0.2)})

    assert xs[:3] == [(0, 64), (52, 116), (104, 168)]  # шаг 64 - int(0.2 * 64) = 52


@pytest.mark.parametrize(("h", "w"), [(64, 110), (150, 200), (40, 30), (700, 1300)])
def test_zones_split_sheet_without_overlap(h, w):
    tiles = grid(h, w, 64, 0.2)

    hits = np.zeros((h, w), int)
    for x0, y0, x1, y1 in zones(tiles, w, h).values():
        hits[int(np.ceil(y0)) : int(np.ceil(y1)), int(np.ceil(x0)) : int(np.ceil(x1))] += 1
    assert (hits == 1).all()


def test_zone_border_is_middle_of_overlap():
    tiles = grid(64, 110, 64, 0.2)  # по x: (0, 64) и (46, 110), перекрытие 46..64

    z = zones(tiles, 110, 64)

    assert z[(0, 0, 64, 64)] == (0, 0, 55.0, 64)
    assert z[(46, 0, 110, 64)] == (55.0, 0, 110, 64)


@pytest.mark.parametrize(("tile", "overlap"), [(64, 1.0), (64, -0.1), (0, 0.2)])
def test_grid_rejects_bad_params(tile, overlap):
    with pytest.raises(ValueError):
        grid(100, 100, tile, overlap)
