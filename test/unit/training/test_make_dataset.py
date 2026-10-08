import json

import numpy as np
import pytest
from conftest import H, W, _write_sheet, triangle

from training.make_dataset import _area, _clip, _grid, make_dataset


def _names(folder):
    return sorted(p.stem for p in folder.iterdir())


def test_split_is_by_sheet_and_close_to_share(export, tmp_path):
    res = make_dataset(export, tmp_path / "d", tile=None, val_share=0.2)

    assert set(res.train).isdisjoint(res.val)
    assert set(res.train) | set(res.val) == {"aa-1", "bb-2", "cc-3", "dd-4"}
    assert sum(res.objects[s] for s in res.val) == 2  # 2 из 10 объектов: ровно 20 %


def test_same_split_with_and_without_tiles(export, tmp_path):
    full = make_dataset(export, tmp_path / "full", tile=None, seed=3)
    tiles = make_dataset(export, tmp_path / "tiles", tile=64, seed=3)

    assert (full.train, full.val) == (tiles.train, tiles.val)


def test_manual_val_by_sheet_number(export, tmp_path):
    res = make_dataset(export, tmp_path / "d", tile=None, val=["3"])

    assert res.val == ("cc-3",)


@pytest.mark.parametrize("val", [["9"], ["aa-1", "bb-2", "cc-3", "dd-4"]], ids=["нет листа", "весь датасет"])
def test_bad_manual_val_raises(export, tmp_path, val):
    with pytest.raises(ValueError):
        make_dataset(export, tmp_path / "d", tile=None, val=val)


def test_full_mode_copies_sheets_and_yaml_has_no_path(export, tmp_path):
    dst = tmp_path / "d"
    res = make_dataset(export, dst, tile=None)

    assert _names(dst / "images" / "val") == sorted(res.val)
    assert _names(dst / "labels" / "train") == sorted(res.train)
    yaml = (dst / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in yaml
    assert "train: images/train" in yaml
    assert '0: "arrow"' in yaml
    assert json.loads((dst / "split.json").read_text(encoding="utf-8"))["val"] == list(res.val)


def test_tiles_have_valid_labels_and_size(export, tmp_path):
    dst = tmp_path / "d"
    make_dataset(export, dst, tile=64, bg_ratio=1.0)

    for part in ("train", "val"):
        for img in (dst / "images" / part).iterdir():
            label = dst / "labels" / part / f"{img.stem}.txt"
            assert label.is_file()
            for line in label.read_text(encoding="utf-8").splitlines():
                xy = np.array(line.split()[1:], float)
                assert len(xy) >= 6
                assert 0 <= xy.min() and xy.max() <= 1


def test_object_cut_below_min_area_is_dropped_and_tile_not_background(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    (root / "classes.txt").write_text("arrow\n", encoding="utf-8")
    # квадрат x 54..74 на границе плиток 0..64 и 64..128: слева 10/20 = 50 %, при min_area=0.6 слева меньше порога
    square = [(54, 20), (74, 20), (74, 40), (54, 40)]
    _write_sheet(root, "aa-1", [square])
    _write_sheet(root, "bb-2", [triangle(10, 10)])
    dst = tmp_path / "d"

    make_dataset(root, dst, tile=64, overlap=0.0, min_area=0.6, val=["1"])

    tiles = _names(dst / "images" / "val")
    assert "aa-1_0_0" not in tiles  # видимый, но не размеченный объект: плитка не годится даже как фон
    assert "aa-1_64_0" not in tiles  # справа 50 %, тоже ниже порога 0.6
    keep = make_dataset(root, tmp_path / "d2", tile=64, overlap=0.0, min_area=0.5, val=["1"])
    assert keep.images["val"] >= 1


def test_val_keeps_all_background_tiles_and_train_is_limited(export, tmp_path):
    none = make_dataset(export, tmp_path / "a", tile=64, bg_ratio=0.0, val=["3"])
    all_bg = make_dataset(export, tmp_path / "b", tile=64, bg_ratio=100.0, val=["3"])

    grid = len(_grid(H, W, 64, 0.2))
    assert all_bg.images["val"] == none.images["val"] == grid  # в val фон не прореживается
    pos_train = sum(1 for p in (tmp_path / "a" / "labels" / "train").iterdir() if p.read_text(encoding="utf-8").strip())
    assert none.images["train"] == pos_train
    assert all_bg.images["train"] > none.images["train"]


def test_dst_not_empty_requires_overwrite(export, tmp_path):
    dst = tmp_path / "d"
    make_dataset(export, dst, tile=None)

    with pytest.raises(ValueError):
        make_dataset(export, dst, tile=None)
    make_dataset(export, dst, tile=None, overwrite=True)


@pytest.mark.parametrize(
    "line",
    ["5 0.1 0.1 0.2 0.1 0.2 0.2", "0 0.1 0.1 0.2 0.1", "0 0.1 0.1 0.2 0.1 1.5 0.2", "0 a b c d e f"],
    ids=["класс вне списка", "две точки", "координата больше 1", "не число"],
)
def test_bad_label_line_raises(export, tmp_path, line):
    (export / "labels" / "aa-1.txt").write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        make_dataset(export, tmp_path / "d", tile=None)


def test_missing_label_raises(export, tmp_path):
    (export / "labels" / "aa-1.txt").unlink()

    with pytest.raises(FileNotFoundError):
        make_dataset(export, tmp_path / "d", tile=None)


@pytest.mark.parametrize(("h", "w"), [(150, 200), (518, 1085), (919, 1475), (30, 40), (640, 640), (641, 1280)], ids=str)
def test_grid_covers_image_and_aligns_to_edge(h, w):
    boxes = _grid(h, w, 64, 0.2)

    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        cover[y0:y1, x0:x1] = True
        assert x1 - x0 == min(64, w) and y1 - y0 == min(64, h)
    assert cover.all()


@pytest.mark.parametrize(("h", "w"), [(150, 200), (518, 1085), (919, 1475), (640, 640), (700, 1300)], ids=str)
def test_grid_matches_sahi(h, w):
    get = pytest.importorskip("sahi.slicing").get_slice_bboxes

    ref = get(h, w, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)

    assert sorted(map(tuple, ref)) == sorted(_grid(h, w, 640, 0.2))


def test_clip_inside_outside_and_straddling():
    tri = np.array([[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]])

    assert _area(_clip(tri, (0, 0, 64, 64))) == pytest.approx(_area(tri))
    assert _clip(tri, (30, 30, 64, 64)) is None
    half = _clip(tri, (0, 0, 15, 64))
    assert _area(half) == pytest.approx(_area(tri) / 2)


def test_clip_matches_shapely_on_random_triangles():
    geometry = pytest.importorskip("shapely.geometry")
    rng = np.random.default_rng(0)
    box = (20, 20, 80, 70)
    for _ in range(200):
        tri = rng.uniform(0, 100, (3, 2))
        want = geometry.Polygon(tri).intersection(geometry.box(*box)).area
        got = _area(_clip(tri, box))
        assert got == pytest.approx(want, abs=1e-6)
