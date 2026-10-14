# core.ocr

Распознаёт текст на растровом чертеже (размерные надписи, позиции, технические требования) через PaddleOCR и
возвращает строки с полигонами в координатах исходного листа.

## Использование

```python
from draftslice.core.ocr.engine import OCREngine, PaddleOCRParameters
from draftslice.core.ocr.pipeline import run

engine = OCREngine(PaddleOCRParameters())  # модели грузятся один раз на пачку листов
results = run(img, engine)                 # list[OcrResult]
for r in results:
    r.text, r.confidence, r.poly           # строка, уверенность, полигон (4, 2) float32, точки (x, y)
```

## Интерфейс

```mermaid
classDiagram
    direction LR
    class PaddleOCRParameters {
        <<dataclass>>
        +float text_rec_score_thresh
        +str text_detection_model_dir
        +str text_recognition_model_dir
        +tuple angles
    }
    class OCREngine {
        +recognize(image) list~OcrResult~
    }
    class OcrResult {
        <<dataclass>>
        +str text
        +float confidence
        +ndarray poly
    }
    class pipeline {
        <<module>>
        +run(image, engine, iou_threshold) list~OcrResult~
    }
    class result_filtering {
        <<module>>
        +filter_results(results, iou_threshold) list~OcrResult~
    }
    OCREngine o-- PaddleOCRParameters
    OCREngine ..> OcrResult
    pipeline ..> OCREngine
    pipeline ..> result_filtering
```

| Объект | Файл | Описание |
|---|---|---|
| `run(image, engine, iou_threshold)` | `pipeline.py` | Распознавание и фильтрация: единственная точка входа |
| `OCREngine.recognize(image)` | `engine.py` | Прогон детекции и распознавания на каждом угле поворота; сырой список |
| `filter_results(results, iou_threshold)` | `result_filtering.py` | Убирает нетекстовый мусор и дубли между поворотами |
| `OcrResult` | `engine.py` | Строка, уверенность и полигон в координатах исходного листа |

## Аргументы

`PaddleOCRParameters`:

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `text_rec_score_thresh` | `float` | `0.3` | Порог уверенности распознавания |
| `text_detection_model_dir` | `str` | `../models/PP-OCRv6_medium_det_infer` | Папка модели детекции |
| `text_recognition_model_dir` | `str` | `../models/cyrillic_PP-OCRv5_mobile_rec_infer` | Папка модели распознавания |
| `text_detection_model_name` | `str` | `PP-OCRv6_medium_det` | Имя модели детекции |
| `text_recognition_model_name` | `str` | `cyrillic_PP-OCRv5_mobile_rec` | Имя модели распознавания |
| `angles` | `tuple[float, ...]` | `(0, 90, 270)` | Углы поворота листа, градусы, на которых ищется текст |

`run`:

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `image` | `np.ndarray` | — | Лист RGB `(H, W, 3)` |
| `engine` | `OCREngine` | — | Заранее созданный движок |
| `iou_threshold` | `float` | `0.3` | Порог IoU, выше которого результаты разных поворотов — одна надпись |

## Пайплайн

```mermaid
sequenceDiagram
    autonumber
    actor C as Вызывающий код
    participant R as pipeline.run
    participant E as OCREngine
    participant O as utils.ops
    participant PO as PaddleOCR
    participant F as filter_results

    C->>+R: run(image, engine, iou_threshold)
    R->>+E: recognize(image)
    loop angle из params.angles
        E->>+O: rotate_image(image, angle)
        O-->>-E: Rotated(img, mat)
        E->>+PO: predict(img)
        PO-->>-E: rec_polys, rec_texts, rec_scores
        E->>E: _polys_to_original(polys, mat) через M⁻¹
    end
    E-->>-R: list[OcrResult], сырые: дубли и мусор
    R->>+F: filter_results(raw, iou_threshold)
    F->>F: отбросить результаты без букв и цифр
    F->>F: NMS по IoU осевых прямоугольников
    F-->>-R: list[OcrResult]
    R-->>-C: list[OcrResult]
```

`OCREngine` держит загруженные модели, поэтому создаётся вызывающим кодом один раз и передаётся в `run`, а не
создаётся внутри на каждый лист.

## Шаг 1. Распознавание на нескольких поворотах

Детектор PaddleOCR надёжно находит только текст, близкий к горизонтальному, а размерные надписи на чертеже часто идут
вертикально вдоль размерных линий. `OCREngine.recognize` прогоняет полный цикл детекция + распознавание на каждом
угле из `angles`: лист поворачивается целиком (`utils.ops.rotate_image`, холст расширяется под новый размер), а
найденные `rec_polys` переводятся обратной матрицей поворота (`cv2.invertAffineTransform`) в систему координат
исходного листа. Другие модули всегда получают `OcrResult.poly` в одной системе координат, независимо от того, на
каком повороте найден текст.

$$
\text{poly}_{\text{original}} = M^{-1} \cdot \text{poly}_{\text{rotated}}
$$

$M$ — аффинная матрица поворота для угла и размера холста.

```text
угол 0°: найдено только горизонтальное        угол 90°: вертикальная надпись стала горизонтальной

┌──────────────────────────┐                  ┌───────────┐
│  [ 40 ]                  │                  │  [ 25 ]   │   polys → M⁻¹ → в координаты
│                      2   │                  │           │   исходного листа
│                      5   │                  └───────────┘
└──────────────────────────┘
```

!!! warning "Встроенный `use_textline_orientation` не решает задачу"
    Классификатор ориентации строки в PaddleOCR/PaddleX корректирует только 0° и 180° (`rotate_angle_list` принимает
    `0` или `1`). Для вертикального текста (90°/270°) он бесполезен.

## Шаг 2. Фильтрация и слияние дублей

Один и тот же текст может быть найден на нескольких поворотах. `filter_results`:

1. Отбрасывает результаты без единой буквы или цифры (`str.isalnum()`, корректно для кириллицы) — шум детектора на
   графике чертежа.
2. Убирает дубли между поворотами через NMS по IoU осевых прямоугольников полигонов: результаты сортируются по
   `confidence`, запись остаётся, если она не перекрывается больше `iou_threshold` с уже оставленной, более
   уверенной.

$$
\text{IoU}(a, b) = \frac{\text{area}(a \cap b)}{\text{area}(a \cup b)}
$$

!!! note "IoU по осевому прямоугольнику"
    Перекрытие считается по осевому прямоугольнику полигона, а не по самому четырёхугольнику: различить «та же
    надпись» и «соседняя надпись» этого достаточно. У полигона, повёрнутого далеко от осей, прямоугольник
    заметно больше самого полигона; на чертежах текст идёт под 0°, 90°, 180° и 270°, где они совпадают.

## Граница с другими модулями

`OcrResult` содержит только `text`, `confidence` и `poly` и ничего не знает о линиях чертежа. Потребители полигонов
(`denoise.text.clear_text`, правила `Sheet`) берут `OcrResult.poly` как вход.
