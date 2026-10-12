"""Стрелки на листе: ONNX-модель YOLO-seg через onnxruntime, лист режется на плитки, как при обучении.

    model = ArrowModel("models/arrows.onnx")
    res = model(img, conf=0.3)
    res.mask, res.xyxy, res.conf, len(res)

ONNX - `training/export.py`: NMS встроен в граф, выход `(1, 300, 6 + 32)` со строками `x1 y1 x2 y2 conf cls` и
коэффициентами маски, плюс прототипы масок `(1, 32, 160, 160)`. Размер входа и имена классов - в метаданных ONNX.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from draftslice.core.denoise.tiles import grid, zones
from draftslice.core.utils.checks import check_image

PAD = 114  # серый letterbox


@dataclass(frozen=True)
class ArrowResult:
    """Стрелки одного листа, в координатах исходного изображения.

    Attributes
    ----------
    mask : np.ndarray[uint8]
        (h, w) маска стрелок листа, `0/255`.
    xyxy : np.ndarray[float32]
        (n, 4) боксы стрелок `x1 y1 x2 y2`, px.
    conf : np.ndarray[float32]
        (n,) уверенность модели.
    """

    mask: np.ndarray
    xyxy: np.ndarray
    conf: np.ndarray

    @property
    def orig_shape(self) -> tuple[int, int]:
        """(h, w) листа."""
        return self.mask.shape[0], self.mask.shape[1]

    def __len__(self) -> int:
        return len(self.conf)


class ArrowModel:
    """Модель стрелок: сессия onnxruntime, размер входа и имена классов из метаданных ONNX.

    Attributes
    ----------
    sess : onnxruntime.InferenceSession
        Сессия модели.
    imgsz : int
        Сторона входа сети, px; она же сторона плитки - модель учили на плитках этого размера.
    names : dict[int, str]
        Имена классов.
    device : str
        `cuda` или `cpu` - где сессия считает на самом деле.
    """

    def __init__(self, model: str | Path, device: str = "cuda") -> None:
        """Загружает ONNX-модель.

        Parameters
        ----------
        model : str | Path
            ONNX из `training/export.py`.
        device : str
            `cuda` - GPU, если onnxruntime-gpu видит CUDA, иначе CPU; `cpu` - всегда CPU.

        Raises
        ------
        FileNotFoundError
            Нет файла модели.
        """
        import onnxruntime as ort

        if not Path(model).is_file():
            raise FileNotFoundError(f"Нет модели стрелок: {model}")
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            ort.preload_dlls()  # CUDA и cuDNN из pip-пакетов nvidia-*
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(str(model), providers=providers)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.imgsz = int(ast.literal_eval(meta["imgsz"])[0])
        self.names = ast.literal_eval(meta["names"])
        self.device = "cuda" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        self._input = self.sess.get_inputs()[0].name

    def __call__(self, img: np.ndarray, conf: float = 0.25, overlap: float = 0.2) -> ArrowResult:
        """То же, что `predict`."""
        return self.predict(img, conf, overlap)

    def predict(self, img: np.ndarray, conf: float = 0.25, overlap: float = 0.2) -> ArrowResult:
        """Стрелки на листе: плитки, модель, зоны плиток, маска листа.

        Parameters
        ----------
        img : np.ndarray
            Исходный лист RGB `(h, w, 3)`, `uint8`, в исходном масштабе: модель учили на нём, а не на
            масштабированном `preprocess`.
        conf : float
            Порог уверенности, в [0, 1].
        overlap : float
            Перекрытие плиток, доля стороны; как при обучении.

        Returns
        -------
        ArrowResult
            Маска, боксы и уверенность стрелок в координатах `img`.

        Raises
        ------
        TypeError
            `img` не трёхканальное.
        ValueError
            `overlap` вне [0, 1).
        """
        check_image(img, channels=3)
        h, w = img.shape[:2]
        tiles = grid(h, w, self.imgsz, overlap)
        zone = zones(tiles, w, h)
        mask = np.zeros((h, w), np.uint8)
        xyxy, scores = [], []
        for box in tiles:
            x0, y0, x1, y1 = box
            det, proto = self.sess.run(None, {self._input: _blob(img[y0:y1, x0:x1], self.imgsz)})
            det = det[0][det[0][:, 4] >= conf]
            cx, cy = (det[:, 0] + det[:, 2]) / 2 + x0, (det[:, 1] + det[:, 3]) / 2 + y0
            zx0, zy0, zx1, zy1 = zone[box]
            det = det[(cx >= zx0) & (cx < zx1) & (cy >= zy0) & (cy < zy1)]
            if len(det):
                m = np.any(_masks(det[:, 6:], proto[0], det[:, :4], self.imgsz), axis=0)
                mask[y0:y1, x0:x1][m[: y1 - y0, : x1 - x0]] = 255
            xyxy.append(det[:, :4] + (x0, y0, x0, y0))
            scores.append(det[:, 4])
        return ArrowResult(mask, np.concatenate(xyxy).astype(np.float32), np.concatenate(scores).astype(np.float32))


def _blob(tile, size):
    """Плитка RGB -> вход сети (1, 3, size, size) float32 в [0, 1]; дополнение серым справа и снизу.

    Каналы не переставляются: сеть обучена на RGB.
    """
    h, w = tile.shape[:2]
    tile = cv2.copyMakeBorder(tile, 0, size - h, 0, size - w, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return cv2.dnn.blobFromImage(tile, 1 / 255)


def _masks(coef, proto, boxes, size):
    """Маски (n, size, size) в разрешении входа.

    Коэффициенты x прототипы, билинейное увеличение до входа, обрезка по боксу в полном разрешении, логит > 0.
    Обрезка до увеличения даёт маску шире стрелки.
    """
    c, ph, pw = proto.shape
    logits = (coef @ proto.reshape(c, -1)).reshape(-1, ph, pw)
    xs, ys = np.arange(size), np.arange(size)[:, None]
    out = np.empty((len(coef), size, size), bool)
    for i, (lg, (x1, y1, x2, y2)) in enumerate(zip(logits, boxes, strict=True)):
        up = cv2.resize(lg, (size, size), interpolation=cv2.INTER_LINEAR) > 0
        out[i] = up & (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return out
