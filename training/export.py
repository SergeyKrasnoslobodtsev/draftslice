"""Экспорт обученной модели YOLO-seg в ONNX для инференса (`infer.py`).

Ultralytics импортируется внутри `export()`: файл импортируется и без него.

ONNX кладётся в `models/` репозитория: туда попадает только выбранная модель, эксперименты остаются в `runs/`.
Экспорт с `nms=True` - NMS встроен в граф: выход `(1, 300, 6 + 32)` со строками `x1 y1 x2 y2 score cls` и
коэффициентами маски, плюс прототипы масок `(1, 32, 160, 160)`. Это та же голова и тот же NMS, что у `predict` и
`val` на `.pt` (обученные веса - `end2end=False`): проверено, совпадают число объектов, уверенность (до 5e-4) и маски.
`nms=False` включил бы другую голову YOLO26 (без NMS), на которой метрики не считались. Размер входа и имена классов
Ultralytics пишет в метаданные ONNX - инференс берёт их оттуда.

Порог уверенности встроенного NMS зашивается в граф (по умолчанию Ultralytics 0.25) и становится нижней границей:
ниже него `infer.py --conf` ничего не найдёт. Поэтому экспорт с `conf=0.001`, а рабочий порог - в
`infer.py`.

Ultralytics при нехватке пакетов ставит их сам; здесь это выключено (`YOLO_AUTOINSTALL=false`): окружение не меняется,
экспорт падает с сообщением, чего не хватает.

Запуск:
    python training/export.py --weights runs/segment/<имя>/weights/best.pt
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # корень репозитория


def export(weights: Path, dst: Path = ROOT / "models" / "arrows.onnx", imgsz: int = 640) -> Path:
    """Экспортирует веса Ultralytics в ONNX со встроенным NMS и кладёт файл в `dst`.

    Parameters
    ----------
    weights : Path
        Обученные веса `.pt`, обычно `runs/segment/<имя>/weights/best.pt`.
    dst : Path
        Куда положить ONNX; существующий файл заменяется.
    imgsz : int
        Размер входа сети, px: сторона плитки, на которой учили модель.

    Returns
    -------
    Path
        Путь к ONNX, равный `dst`.

    Raises
    ------
    FileNotFoundError
        Нет файла весов.
    """
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    if not Path(weights).is_file():
        raise FileNotFoundError(f"Нет весов: {weights}")
    out = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, nms=True, conf=0.001, simplify=True))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(out, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True, help="runs/segment/<имя>/weights/best.pt")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "arrows.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    print(f"ONNX: {export(a.weights, a.dst, a.imgsz)}")


if __name__ == "__main__":
    main()
