import sys
import types

import pytest

from training.export import export


class FakeYOLO:
    """Подмена ultralytics.YOLO: export пишет пустой .onnx рядом с весами, как Ultralytics."""

    kwargs = None

    def __init__(self, weights):
        self.weights = weights

    def export(self, **kwargs):
        FakeYOLO.kwargs = kwargs
        out = self.weights.replace(".pt", ".onnx")
        open(out, "wb").close()
        return out


@pytest.fixture
def fake_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))


def test_export_embeds_nms_and_moves_onnx_to_dst(fake_ultralytics, tmp_path):
    weights = tmp_path / "run" / "best.pt"
    weights.parent.mkdir()
    weights.touch()
    dst = tmp_path / "models" / "arrows.onnx"

    out = export(weights, dst, imgsz=640)

    assert out == dst and dst.is_file()
    assert not (weights.parent / "best.onnx").exists()  # перенесён, а не скопирован
    # conf встроенного NMS - нижняя граница порога infer.py: зашиваем минимальный, как у val
    assert FakeYOLO.kwargs == {"format": "onnx", "imgsz": 640, "nms": True, "conf": 0.001, "simplify": True}


def test_export_without_weights_raises(fake_ultralytics, tmp_path):
    with pytest.raises(FileNotFoundError):
        export(tmp_path / "nope.pt", tmp_path / "a.onnx")
