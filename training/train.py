"""Обучение YOLO-seg (Ultralytics) на датасете, подготовленном `make_dataset.py`.

`train()` - единственная публичная функция файла. Ultralytics и albumentations импортируются внутри, поэтому файл
импортируется и без них (тесты подменяют оба модуля).

Параметры аугментации - поля `Augment`, значения по умолчанию рассчитаны на чёрно-белые чертежи: цвет отключён,
повороты на любой угол (стрелки бывают под любым углом), масштаб, отражения, небольшое размытие.

Запуск (с плитками, imgsz равен плитке):
    python training/train.py --data data/dataset_yolo/tiles/data.yaml --imgsz 640
Запуск (целые листы):
    python training/train.py --data data/dataset_yolo/full/data.yaml --imgsz 1280
"""

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Augment:
    """Аугментации обучения. Все поля, кроме `blur`, - одноимённые параметры `YOLO.train`.

    Attributes
    ----------
    degrees : float
        Поворот на случайный угол из [-degrees, degrees], 0..180.
    scale : float
        Масштаб случайный из (1 - scale, 1 + scale); больше 2 раз увеличить этим параметром нельзя.
    translate : float
        Сдвиг, доля размера изображения.
    shear, perspective : float
        Сдвиг-скос и перспектива; для чертежей отключены.
    flipud, fliplr : float
        Вероятности отражения по вертикали и по горизонтали.
    hsv_h, hsv_s, hsv_v : float
        Цвет: оттенок и насыщенность отключены, яркость - небольшая (качество скана).
    mosaic, mixup, copy_paste : float
        Вероятности мозаики, смешивания и копирования объектов между изображениями.
    close_mosaic : float
        Мозаика выключается на последних close_mosaic эпохах.
    blur : float
        Вероятность размытия albumentations (`Blur`, ядро 3-5 px); 0 - не добавлять.
    """

    degrees: float = 180.0
    scale: float = 0.5
    translate: float = 0.1
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.5
    fliplr: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.2
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    close_mosaic: float = 10
    blur: float = 0.3


def train(
    data: Path,
    model: str = "yolo26n-seg.pt",
    imgsz: int = 640,
    epochs: int = 100,
    batch: float = -1,
    patience: int = 50,
    device: str | None = None,
    workers: int = 8,
    mask_ratio: int = 4,
    seed: int = 0,
    project: Path = Path("data/runs"),
    name: str | None = None,
    aug: Augment = Augment(),  # noqa: B008 - frozen dataclass, делить состояние нечему
) -> Path:
    """Обучает YOLO-seg на `data.yaml` и возвращает папку запуска.

    Parameters
    ----------
    data : Path
        `data.yaml` датасета (результат `make_dataset.py`).
    model : str
        Веса или конфигурация Ultralytics, например `yolo26n-seg.pt`.
    imgsz : int
        Размер входа. Для плиток 640 - сторона плитки: сеть получает их без ресайза. Для целых листов 1280.
    epochs, patience : int
        Число эпох и терпение ранней остановки.
    batch : float
        Размер батча; -1 - автоподбор по 60 % памяти GPU.
    device : str | None
        Устройство Ultralytics (`0`, `cpu`, `mps`); `None` - выбирает сам.
    workers : int
        Число процессов загрузки данных.
    mask_ratio : int
        Во сколько раз уменьшается маска при обучении. По умолчанию Ultralytics 4: у стрелки 17 px маска
        получается ~4 px, для мелких объектов можно попробовать 1.
    seed : int
        Зерно обучения.
    project : Path
        Папка запусков.
    name : str | None
        Имя запуска; по умолчанию `<папка датасета>_<веса>_<imgsz>`.
    aug : Augment
        Аугментации.

    Returns
    -------
    Path
        Папка запуска Ultralytics с весами и графиками.
    """
    from ultralytics import YOLO

    params = {k: v for k, v in asdict(aug).items() if k != "blur"}
    if aug.blur > 0:
        import albumentations as A

        params["augmentations"] = [A.Blur(blur_limit=(3, 5), p=aug.blur)]
    if device is not None:
        params["device"] = device
    yolo = YOLO(model)
    yolo.train(
        data=str(data),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        mask_ratio=mask_ratio,
        seed=seed,
        project=str(project),
        name=name or f"{Path(data).parent.name}_{Path(model).stem}_{imgsz}",
        **params,
    )
    return Path(yolo.trainer.save_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=float, default=-1)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mask-ratio", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("data/runs"))
    ap.add_argument("--name")
    for f in fields(Augment):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=f.default, help="аугментация, см. Augment")
    a = ap.parse_args()
    aug = Augment(**{f.name: getattr(a, f.name) for f in fields(Augment)})
    save_dir = train(
        a.data, a.model, a.imgsz, a.epochs, a.batch, a.patience, a.device, a.workers, a.mask_ratio, a.seed,
        a.project, a.name, aug,
    )  # fmt: skip
    print(f"запуск: {save_dir}")


if __name__ == "__main__":
    main()
