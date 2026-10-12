"""Инференс стрелок на листе: ONNX-модель YOLO-seg (`export.py`) через onnxruntime, GPU при наличии.

`load()` и `predict()` - публичные функции файла. onnxruntime импортируется внутри `load()`, поэтому файл
импортируется и без него.

Лист режется той же сеткой, что при обучении (`make_dataset.grid`: плитка = вход сети, перекрытие 0.2), плитка
меньше входа дополняется серым (114, как letterbox Ultralytics) справа и снизу - координаты плитки не меняются.
NMS встроен в модель (`export.py`): в одной плитке дублей нет; выход `(1, 300, 6 + 32)` - `x1 y1 x2 y2 score cls`
и коэффициенты маски, плюс прототипы `(1, 32, 160, 160)`.

Маска объекта - как `ultralytics.utils.ops.process_mask_native` (`retina_masks`): коэффициенты x прототипы,
билинейное увеличение до входа, обрезка по боксу в полном разрешении, логит > 0. Обрезка до увеличения
(`process_mask`) даёт маску шире стрелки - для удаления стрелок нужна точная.

Дубли между плитками убирает правило зон, а не NMS: граница зон соседних плиток проходит по середине их перекрытия,
зоны делят лист без пересечений, и стрелка засчитывается только плитке, в зону которой попал центр её бокса.
Стрелку, обрезанную краем плитки, модель тоже находит (в обучении оставлены объекты с половиной площади), но её центр
лежит у края плитки, за границей зоны; целиком её видит соседняя плитка. Условие: стрелка меньше половины
перекрытия (64 px при плитке 640 и перекрытии 0.2).

Прогон по папке (рисунки: стрелки красным поверх листа):
    python training/infer.py data/images --out data/output/arrows
"""

import argparse
import ast
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from make_dataset import IMG_EXT, grid

ROOT = Path(__file__).resolve().parent.parent  # корень репозитория
PAD = 114


@dataclass(frozen=True)
class Arrows:
    """Стрелки листа.

    Attributes
    ----------
    mask : np.ndarray
        (h, w) uint8, 255 - пиксели стрелок, размер листа.
    boxes : np.ndarray
        (n, 4) float32, `x1 y1 x2 y2` в px листа.
    score : np.ndarray
        (n,) float32, уверенность модели.
    """

    mask: np.ndarray
    boxes: np.ndarray
    score: np.ndarray


def load(path: Path = ROOT / "models" / "arrows.onnx", gpu: bool = True):
    """Сессия onnxruntime для ONNX из `export.py`.

    Parameters
    ----------
    path : Path
        ONNX-модель.
    gpu : bool
        Считать на CUDA, если onnxruntime-gpu её видит; иначе и при `False` - на CPU.

    Returns
    -------
    onnxruntime.InferenceSession
        Сессия; размер входа - в её метаданных.

    Raises
    ------
    FileNotFoundError
        Нет файла модели.
    """
    import onnxruntime as ort

    if not Path(path).is_file():
        raise FileNotFoundError(f"Нет модели: {path}")
    providers = ["CPUExecutionProvider"]
    if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()  # CUDA и cuDNN из pip-пакетов nvidia-* (их ставит torch)
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), providers=providers)


def predict(sess, img: np.ndarray, conf: float = 0.25, overlap: float = 0.2) -> Arrows:
    """Стрелки на листе: плитки, модель, правило зон, маска листа.

    Parameters
    ----------
    sess : onnxruntime.InferenceSession
        Сессия из `load()`.
    img : np.ndarray
        Лист, uint8 BGR (h, w, 3) или серый (h, w).
    conf : float
        Порог уверенности, в [0, 1].
    overlap : float
        Перекрытие плиток, доля стороны, как при обучении.

    Returns
    -------
    Arrows
        Маска стрелок листа, боксы и уверенность в px листа.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    size = int(ast.literal_eval(sess.get_modelmeta().custom_metadata_map["imgsz"])[0])
    name = sess.get_inputs()[0].name
    h, w = img.shape[:2]
    tiles = grid(h, w, size, overlap)
    zones = _zones(tiles, w, h)
    mask = np.zeros((h, w), np.uint8)
    boxes, score = [], []
    for box in tiles:
        x0, y0, x1, y1 = box
        det, proto = sess.run(None, {name: _blob(img[y0:y1, x0:x1], size)})
        det = det[0][det[0][:, 4] >= conf]
        cx, cy = (det[:, 0] + det[:, 2]) / 2 + x0, (det[:, 1] + det[:, 3]) / 2 + y0
        zx0, zy0, zx1, zy1 = zones[box]
        det = det[(cx >= zx0) & (cx < zx1) & (cy >= zy0) & (cy < zy1)]
        if len(det):
            m = np.any(_masks(det[:, 6:], proto[0], det[:, :4], size), axis=0)
            mask[y0:y1, x0:x1][m[: y1 - y0, : x1 - x0]] = 255
        boxes.append(det[:, :4] + (x0, y0, x0, y0))
        score.append(det[:, 4])
    return Arrows(mask, np.concatenate(boxes).astype(np.float32), np.concatenate(score).astype(np.float32))


def _blob(tile: np.ndarray, size: int) -> np.ndarray:
    """Плитка -> вход сети (1, 3, size, size) float32 RGB в [0, 1]; дополнение серым справа и снизу."""
    h, w = tile.shape[:2]
    tile = cv2.copyMakeBorder(tile, 0, size - h, 0, size - w, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return cv2.dnn.blobFromImage(tile, 1 / 255, swapRB=True)


def _masks(coef: np.ndarray, proto: np.ndarray, boxes: np.ndarray, size: int) -> np.ndarray:
    """Маски объектов (n, size, size) bool: коэффициенты x прототипы, увеличение, обрезка по боксу, логит > 0."""
    c, ph, pw = proto.shape
    logits = (coef @ proto.reshape(c, -1)).reshape(-1, ph, pw)
    xs, ys = np.arange(size), np.arange(size)[:, None]
    out = np.empty((len(coef), size, size), bool)
    for i, (lg, (x1, y1, x2, y2)) in enumerate(zip(logits, boxes, strict=True)):
        up = cv2.resize(lg, (size, size), interpolation=cv2.INTER_LINEAR) > 0
        out[i] = up & (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return out


def _zones(tiles: list, w: int, h: int) -> dict:
    """Плитка -> её зона (x0, y0, x1, y1): границы по серединам перекрытий соседних плиток, зоны делят лист."""
    xs = _bounds(sorted({(b[0], b[2]) for b in tiles}), w)
    ys = _bounds(sorted({(b[1], b[3]) for b in tiles}), h)
    zones = {}
    for b in tiles:
        (zx0, zx1), (zy0, zy1) = xs[(b[0], b[2])], ys[(b[1], b[3])]
        zones[b] = (zx0, zy0, zx1, zy1)
    return zones


def _bounds(spans: list[tuple[int, int]], size: int) -> dict:
    """Отрезки (a, b) одной оси по возрастанию -> (начало, конец) зоны; крайние зоны до края листа."""
    cuts = [(prev[1] + cur[0]) / 2 for prev, cur in zip(spans, spans[1:], strict=False)]
    return dict(zip(spans, zip([0, *cuts], [*cuts, size], strict=True), strict=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="папка с листами")
    ap.add_argument("--out", type=Path, required=True, help="куда писать рисунки: стрелки красным поверх листа")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "arrows.onnx")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    sess = load(a.model, gpu=not a.cpu)
    print(f"модель {a.model}, провайдер {sess.get_providers()[0]}, conf {a.conf}")
    a.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in a.src.iterdir() if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))
    print(f"{'лист':24s} {'размер':>11s} {'стрелок':>7s} {'мс':>6s}")
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{p.name:24s} не читается, пропущен")
            continue
        t = time.perf_counter()
        res = predict(sess, img, a.conf)
        ms = (time.perf_counter() - t) * 1000
        h, w = img.shape[:2]
        print(f"{p.name:24s} {w:>5d}x{h:<5d} {len(res.score):7d} {ms:6.0f}")
        img[res.mask > 0] = (0, 0, 255)
        cv2.imwrite(str(a.out / f"{p.stem}.png"), img)


if __name__ == "__main__":
    main()
