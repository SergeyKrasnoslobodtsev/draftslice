# Обучение YOLO-seg

Скрипты не зависят от пакета `draftslice`: нужны только `ultralytics`, `sahi` и opencv.

## Установка

```bash
uv sync --group training
```

## Датасет

Исходный экспорт Label Studio (формат YOLO-seg) лежит в `data/dataset`: `images/`, `labels/`, `classes.txt`.
Подготовленный датасет пишется в `data/dataset_yolo/` (папка `data/` в git не попадает).

## Запуск

Подготовка датасета, плитки 640 (результат в `data/dataset_yolo/tiles`):

```bash
python training/make_dataset.py --overwrite
```

Подготовка датасета, целые листы (результат в `data/dataset_yolo/full`):

```bash
python training/make_dataset.py --no-tile --overwrite
```

Обучение на плитках:

```bash
python training/train.py --data data/dataset_yolo/tiles/data.yaml --imgsz 640
```

Обучение на целых листах:

```bash
python training/train.py --data data/dataset_yolo/full/data.yaml --imgsz 1280
```

Результаты обучения пишутся в `data/runs`.

## Что важно знать

- Листы делятся на train и val целиком, не по стрелкам. С одними и теми же `--val-share`, `--seed`, `--val`
  режимы с плитками и без них дают одно и то же разбиение, поэтому модели сравнимы на одних val-листах.
  Val можно задать вручную номерами листов: `--val 1 5`.
- Плитки режутся в исходном масштабе листа по сетке SAHI. Параметры: `--tile 640`, `--overlap 0.2`,
  `--min-area 0.5` (доля площади объекта, которая должна остаться в плитке), `--bg-ratio 0.25` (плиток без стрелок
  в train, доля от числа плиток со стрелками; в val остаются все).
- При обучении на плитках `--imgsz` равен стороне плитки, на целых листах - 1280. На инференсе SAHI должен резать
  оригинал теми же плитками (640, перекрытие 0.2), иначе выигрыш от нарезки пропадает.
- Аугментации - поля `Augment` в `train.py`, переопределяются флагами (`--degrees 90`, `--scale 0.7`, `--blur 0`).
- В `data.yaml` нет ключа `path`, поэтому папку `data/dataset_yolo/<режим>` можно скопировать на другую машину.
- Метрики Ultralytics на плитках и на целых листах несравнимы. Для сравнения режимов нужен прогон на целых
  val-листах в координатах листа.
