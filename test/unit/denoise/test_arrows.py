import sys
import types

import cv2
import numpy as np
import pytest

from draftslice.core.denoise.arrows import ArrowModel, ArrowResult, _masks

SIZE = 64
P = SIZE // 4  # прототипы масок в 1/4 входа, как у YOLO-seg


class FakeSession:
    """Подмена onnxruntime.InferenceSession: выход как у экспорта YOLO-seg с NMS - (1, 300, 6 + 32) и прототипы.

    «Модель» находит тёмные пятна на плитке, в том числе обрезанные краем плитки - как настоящая модель находит
    обрезанную стрелку. Прототип 0 - логит «тёмное» (+10 / -10), коэффициенты маски выбирают только его.
    """

    feeds: list = []

    def __init__(self, path, providers):
        self.path, self.providers = path, providers

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={"imgsz": f"[{SIZE}, {SIZE}]", "names": "{0: 'arrow'}"})

    def get_inputs(self):
        return [types.SimpleNamespace(name="images")]

    def get_providers(self):
        return self.providers

    def run(self, _, feed):
        FakeSession.feeds.append(feed["images"])
        dark = (feed["images"][0].mean(0) < 0.3).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        det = np.zeros((300, 38), np.float32)
        for i, (x, y, w, h, _) in enumerate(stats[1:]):
            det[i, :6] = (x, y, x + w, y + h, 0.9, 0)
            det[i, 6] = 1.0
        cover = cv2.resize(dark.astype(np.float32), (P, P), interpolation=cv2.INTER_AREA)  # доля тёмного в клетке
        proto = np.zeros((32, P, P), np.float32)
        proto[0] = np.where(cover > 0, 10, -10)
        return [det[None], proto[None]]


@pytest.fixture
def model(monkeypatch, tmp_path):
    """ArrowModel на поддельном onnxruntime; CUDA «есть»."""
    FakeSession.feeds = []
    ort = types.SimpleNamespace(
        InferenceSession=FakeSession,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    path = tmp_path / "arrows.onnx"
    path.touch()
    return ArrowModel(path)


def _sheet(w: int, h: int, squares: list[tuple[int, int, int]]) -> np.ndarray:
    """Белый RGB-лист с чёрными квадратами (x, y, сторона)."""
    img = np.full((h, w, 3), 255, np.uint8)
    for x, y, s in squares:
        img[y : y + s, x : x + s] = 0
    return img


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ArrowModel(tmp_path / "nope.onnx")


def test_model_reads_input_size_and_names_from_onnx_metadata(model):
    assert model.imgsz == SIZE
    assert model.names == {0: "arrow"}
    assert model.device == "cuda"


def test_cpu_when_gpu_disabled(monkeypatch, tmp_path, model):
    path = tmp_path / "cpu.onnx"
    path.touch()

    cpu = ArrowModel(path, device="cpu")

    assert cpu.device == "cpu"
    assert cpu.sess.providers == ["CPUExecutionProvider"]


@pytest.mark.req("DEL-03", "DEL-05")
def test_arrow_in_overlap_counted_once_and_cut_one_dropped(model):
    # плитки по x: (0, 64) и (46, 110), граница зон x = 55
    # квадрат 48..56 целиком в обеих плитках, центр 52 - зона первой;
    # квадрат 60..68 обрезан краем первой плитки (её находка с центром 62 вне зоны), целиком - во второй
    img = _sheet(110, 64, [(48, 20, 8), (60, 40, 8)])

    res = model(img, conf=0.5)

    assert isinstance(res, ArrowResult)
    assert len(res) == 2
    assert sorted(map(tuple, res.xyxy[:, :2].round().astype(int))) == [(48, 20), (60, 40)]  # координаты листа
    assert res.orig_shape == (64, 110)
    assert res.mask.shape == (64, 110) and res.mask.dtype == np.uint8
    for x, y in ((48, 20), (60, 40)):  # квадрат без 4 угловых px: их срезает билинейное увеличение прототипа
        assert res.mask[y : y + 8, x + 1 : x + 7].all() and res.mask[y + 1 : y + 7, x : x + 8].all()
    assert not res.mask[:, :40].any() and not res.mask[:15].any()


def test_conf_filters_detections(model):
    res = model.predict(_sheet(110, 64, [(10, 10, 8)]), conf=0.95)

    assert len(res) == 0 and res.xyxy.shape == (0, 4) and not res.mask.any()


def test_rgb_goes_to_network_as_is_and_small_tile_is_padded_gray(model):
    img = np.zeros((20, 30, 3), np.uint8)
    img[..., 0] = 200  # R

    model(img)

    (blob,) = FakeSession.feeds
    assert blob.shape == (1, 3, SIZE, SIZE)
    # RGB без перестановки каналов
    assert np.allclose(blob[0, 0, :20, :30], 200 / 255) and np.allclose(blob[0, 1:, :20, :30], 0)
    assert np.allclose(blob[0, :, 20:, :], 114 / 255) and np.allclose(blob[0, :, :, 30:], 114 / 255)


def test_gray_image_is_rejected(model):
    with pytest.raises(TypeError):
        model(np.zeros((64, 64), np.uint8))


def test_masks_are_cropped_by_box_at_full_resolution():
    proto = np.full((1, P, P), 5.0, np.float32)  # логит > 0 везде

    (m,) = _masks(np.ones((1, 1), np.float32), proto, np.array([[10.5, 20, 30, 41]], np.float32), SIZE)

    ys, xs = np.nonzero(m)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (11, 29, 20, 40)  # граница бокса в px входа, не в 1/4
