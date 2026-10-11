import sys
import types
from pathlib import Path

import pytest

from training.train import Augment, main, train


class FakeYOLO:
    """Подмена ultralytics.YOLO: запоминает вызовы."""

    last = None

    def __init__(self, model):
        self.model = model
        self.trained = None
        FakeYOLO.last = self

    def train(self, **kwargs):
        self.trained = kwargs
        self.trainer = types.SimpleNamespace(save_dir="runs/out")


WEIGHTS_DIR = Path("/w")


def _fake_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules, "ultralytics.utils", types.SimpleNamespace(SETTINGS={"weights_dir": WEIGHTS_DIR}))


class FakeBlur:
    def __init__(self, blur_limit, p):
        self.blur_limit, self.p = blur_limit, p


@pytest.fixture
def fake_libs(monkeypatch, tmp_path):
    FakeYOLO.last = None
    monkeypatch.chdir(tmp_path)  # веса по имени ищутся и в текущей папке: в корне репозитория они могут лежать
    _fake_ultralytics(monkeypatch)
    monkeypatch.setitem(sys.modules, "albumentations", types.SimpleNamespace(Blur=FakeBlur))


def test_train_passes_data_size_and_augmentations(fake_libs):
    out = train(Path("d/tiles/data.yaml"), imgsz=640, aug=Augment(degrees=90.0, scale=0.7))

    kw = FakeYOLO.last.trained
    assert FakeYOLO.last.model == str(WEIGHTS_DIR / "yolo26n-seg.pt")  # предобученные - в weights_dir
    assert (kw["data"], kw["imgsz"], kw["degrees"], kw["scale"]) == ("d/tiles/data.yaml", 640, 90.0, 0.7)
    assert (kw["hsv_h"], kw["hsv_s"]) == (0.0, 0.0)  # чертежи чёрно-белые
    assert "blur" not in kw
    assert kw["name"] == "tiles_yolo26n-seg_640"
    assert "project" not in kw  # папка запуска - runs_dir Ultralytics
    assert out == Path("runs/out")


def test_blur_becomes_albumentations_transform(fake_libs):
    train(Path("d/data.yaml"), aug=Augment(blur=0.4))

    (blur,) = FakeYOLO.last.trained["augmentations"]
    assert (blur.blur_limit, blur.p) == ((3, 5), 0.4)


def test_no_blur_does_not_need_albumentations(monkeypatch):
    _fake_ultralytics(monkeypatch)
    monkeypatch.setitem(sys.modules, "albumentations", None)  # импорт даёт ImportError

    train(Path("d/data.yaml"), aug=Augment(blur=0.0))

    assert "augmentations" not in FakeYOLO.last.trained


def test_own_weights_path_is_used_as_is(fake_libs, tmp_path):
    best = tmp_path / "best.pt"
    best.touch()

    train(Path("d/data.yaml"), model=str(best))

    assert FakeYOLO.last.model == str(best)


def test_device_is_passed_only_when_given(fake_libs):
    train(Path("d/data.yaml"))
    assert "device" not in FakeYOLO.last.trained

    train(Path("d/data.yaml"), device="0")
    assert FakeYOLO.last.trained["device"] == "0"


def test_cli_overrides_augmentation(fake_libs, monkeypatch, capsys):
    args = ["--data", "d/data.yaml", "--imgsz", "1024", "--degrees", "45", "--blur", "0"]
    monkeypatch.setattr(sys, "argv", ["train.py", *args])

    main()

    kw = FakeYOLO.last.trained
    assert (kw["imgsz"], kw["degrees"]) == (1024, 45.0)
    assert "augmentations" not in kw
    assert "запуск:" in capsys.readouterr().out
