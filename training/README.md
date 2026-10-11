# Обучение YOLO-seg: стрелки

Скрипты не зависят от пакета `draftslice`. Обучение - `ultralytics` и opencv, экспорт - `onnx` и `onnxslim`
(группа `training`), инференс - `onnxruntime-gpu` (основные зависимости).

## Установка

```bash
uv sync --group training
```

## Где что лежит

| что | где | в git |
|---|---|---|
| код прогона | `training/` | да |
| экспорт Label Studio (исходник, руками не трогаем) | `data/dataset/` | нет |
| датасеты YOLO | `data/dataset_yolo/<режим>/` | нет |
| предобученные веса Ultralytics | `weights/` (`weights_dir` в `yolo settings`) | нет |
| запуски обучения | `runs/segment/<имя>/` (`runs_dir` в `yolo settings`) | нет |
| выбранная модель для инференса | `models/arrows.onnx` | да, как модели OCR |
| рисунки прогона инференса | `data/output/arrows/` | нет |

## Прогон

### 1. Разметка

Стрелки размечаются в Label Studio полигонами, экспорт в формате YOLO-seg распаковывается в `data/dataset`:
`images/`, `labels/`, `classes.txt`. Служебные файлы macOS (`._*`) скрипты пропускают.

### 2. Датасет

Плитки 640 (результат в `data/dataset_yolo/tiles`):

```bash
python training/make_dataset.py --overwrite
```

Целые листы (результат в `data/dataset_yolo/full`):

```bash
python training/make_dataset.py --no-tile --overwrite
```

### 3. Обучение

На плитках (запуск в `runs/segment/tiles_yolo26n-seg_640`):

```bash
python training/train.py --data data/dataset_yolo/tiles/data.yaml --imgsz 640 --epochs 300
```

На целых листах:

```bash
python training/train.py --data data/dataset_yolo/full/data.yaml --imgsz 1280
```

Повторный запуск с тем же именем Ultralytics пишет в `<имя>2`, `<имя>3`; своё имя - `--name`.

### 4. Конвертация в ONNX

Выбранный запуск -> `models/arrows.onnx`:

```bash
python training/export.py --weights runs/segment/tiles_yolo26n-seg_640/weights/best.pt
```

### 5. Инференс: прогон по всем листам

Рисунки со стрелками красным поверх листа - в `data/output/arrows/`, в консоль - число стрелок и время на лист:

```bash
python training/infer.py data/images --out data/output/arrows
python training/infer.py data/drawing --out data/output/arrows_drawing
```

Флаги: `--conf 0.25` - порог уверенности, `--model` - другой ONNX, `--cpu` - без GPU.

## Что важно знать

- Листы делятся на train и val целиком, не по стрелкам. С одними и теми же `--val-share`, `--seed`, `--val`
  режимы с плитками и без них дают одно и то же разбиение, поэтому модели сравнимы на одних val-листах.
  Val можно задать вручную номерами листов: `--val 1 5`.
- Плитки режутся в исходном масштабе листа по сетке SAHI. Параметры: `--tile 640`, `--overlap 0.2`,
  `--min-area 0.5` (доля площади объекта, которая должна остаться в плитке), `--bg-ratio 0.25` (плиток без стрелок
  в train, доля от числа плиток со стрелками; в val остаются все).
- При обучении на плитках `--imgsz` равен стороне плитки, на целых листах - 1280. `infer.py` режет лист той же
  сеткой (`make_dataset.grid`, 640, перекрытие 0.2), иначе выигрыш от нарезки пропадает.
- Аугментации - поля `Augment` в `train.py`, переопределяются флагами (`--degrees 90`, `--scale 0.7`, `--blur 0`).
- В `data.yaml` нет ключа `path`, поэтому папку `data/dataset_yolo/<режим>` можно скопировать на другую машину.
- Метрики Ultralytics на плитках и на целых листах несравнимы. Для сравнения режимов нужен прогон на целых
  val-листах в координатах листа.
- ONNX экспортируется со встроенным NMS (`nms=True`): та же голова и тот же NMS, что у `predict`/`val` на `.pt`,
  результат совпадает с Ultralytics (`retina_masks=True`). Порог встроенного NMS зашивается в граф и ограничивает
  `infer.py --conf` снизу, поэтому экспорт идёт с `conf=0.001`, как у `val`. `export.py` не даёт Ultralytics
  доустанавливать пакеты.
- Дубли стрелок между плитками `infer.py` убирает правилом зон: стрелка засчитывается плитке, в чью зону (граница -
  середина перекрытия) попал центр её бокса.
- Листы крупнее учебных (4102-9239 px против 1085-1475 px) - вне масштаба обучения, на них модель ошибается.
