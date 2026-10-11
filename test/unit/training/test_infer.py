import types

import cv2
import numpy as np
import pytest

from training.infer import _blob, _bounds, _masks, _zones, predict
from training.make_dataset import grid

SIZE = 64
P = SIZE // 4  # прототипы масок в 1/4 входа, как у YOLO-seg


class FakeSession:
    """Подмена onnxruntime: «модель» находит тёмные пятна на плитке, выход как у экспорта (1, 300, 6 + 32) + прототипы.

    Пятно, обрезанное краем плитки, тоже находится - как у настоящей модели на обрезанной стрелке. Прототип 0 -
    логит «тёмное» (+10 / -10) в 1/4 разрешения, коэффициенты маски выбирают только его.
    """

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={"imgsz": f"[{SIZE}, {SIZE}]"})

    def get_inputs(self):
        return [types.SimpleNamespace(name="images")]

    def run(self, _, feed):
        dark = (feed["images"][0, 0] < 0.3).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        det = np.zeros((300, 38), np.float32)
        for i, (x, y, w, h, _) in enumerate(stats[1:]):
            det[i, :6] = (x, y, x + w, y + h, 0.9, 0)
            det[i, 6] = 1.0
        proto = np.zeros((32, P, P), np.float32)
        cover = cv2.resize(dark.astype(np.float32), (P, P), interpolation=cv2.INTER_AREA)  # доля тёмного в клетке
        proto[0] = np.where(cover > 0, 10, -10)
        return [det[None], proto[None]]


def _sheet(w, h, squares):
    img = np.full((h, w, 3), 255, np.uint8)
    for x, y, s in squares:
        img[y : y + s, x : x + s] = 0
    return img


def test_bounds_split_axis_at_overlap_middles():
    assert _bounds([(0, 64), (46, 110)], 110) == {(0, 64): (0, 55.0), (46, 110): (55.0, 110)}
    assert _bounds([(0, 50)], 50) == {(0, 50): (0, 50)}


@pytest.mark.parametrize(("h", "w"), [(64, 110), (150, 200), (40, 30)])
def test_zones_cover_sheet_without_overlap(h, w):
    tiles = grid(h, w, SIZE, 0.2)
    hits = np.zeros((h, w), int)
    for x0, y0, x1, y1 in _zones(tiles, w, h).values():
        hits[int(np.ceil(y0)) : int(np.ceil(y1)), int(np.ceil(x0)) : int(np.ceil(x1))] += 1
    assert (hits == 1).all()


def test_blob_pads_small_tile_right_and_bottom():
    blob = _blob(np.zeros((20, 30, 3), np.uint8), SIZE)

    assert blob.shape == (1, 3, SIZE, SIZE)
    assert (blob[0, :, :20, :30] == 0).all()
    assert np.allclose(blob[0, :, 20:, :], 114 / 255) and np.allclose(blob[0, :, :, 30:], 114 / 255)


def test_masks_are_cropped_by_box_at_full_resolution():
    proto = np.full((1, P, P), 5.0, np.float32)  # логит > 0 везде

    (m,) = _masks(np.ones((1, 1), np.float32), proto, np.array([[10.5, 20, 30, 41]], np.float32), SIZE)

    ys, xs = np.nonzero(m)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (11, 29, 20, 40)  # граница бокса в px входа, не в 1/4


def test_arrow_seen_by_two_tiles_counted_once_and_cut_one_dropped():
    # плитки по x: (0, 64) и (46, 110), граница зон x = 55
    # квадрат 48..56 целиком в обеих плитках, центр 52 - зона первой;
    # квадрат 60..68 обрезан краем первой плитки (её находка с центром 62 вне зоны), целиком - во второй
    img = _sheet(110, 64, [(48, 20, 8), (60, 40, 8)])

    res = predict(FakeSession(), img, conf=0.5)

    assert len(res.score) == 2
    assert sorted(map(tuple, res.boxes[:, :2].round().astype(int))) == [(48, 20), (60, 40)]  # координаты листа
    assert res.mask.shape == (64, 110)
    for x, y in ((48, 20), (60, 40)):  # квадрат без 4 угловых px: их срезает билинейное увеличение прототипа
        assert res.mask[y : y + 8, x + 1 : x + 7].all() and res.mask[y + 1 : y + 7, x : x + 8].all()
    assert not res.mask[:, :40].any() and not res.mask[:15].any()


def test_conf_filters_detections():
    res = predict(FakeSession(), _sheet(110, 64, [(10, 10, 8)]), conf=0.95)

    assert len(res.score) == 0 and not res.mask.any()
