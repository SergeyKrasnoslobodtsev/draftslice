# Архитектура draftslice

Библиотека обработки растровых чертежей: из скана получает чистый лист без шума (стрелки, таблицы,
обозначения, текст) и виды детали для ViT. Работает как библиотека (`draftslice.core`) и как CLI
(`draftslice clean`, `draftslice views`).

## Пакеты

Стрелка — зависимость по импорту: «использует». Данные между модулями идут через вызывающий код
(`denoise.run`), см. [последовательность вызовов](#последовательность-вызовов).

```mermaid
flowchart TB
    cli["«cli»<br/>draftslice.cli"]

    subgraph core["draftslice.core"]
        direction TB
        denoise["«package»<br/>denoise"]
        ocr["«package»<br/>ocr"]
        thick_lines["«package»<br/>thick_lines"]
        preprocess["«module»<br/>preprocess"]
        io["«module»<br/>io"]
        subgraph utils["utils"]
            checks["«module»<br/>checks"]
            ops["«module»<br/>ops"]
        end
    end

    subgraph ext["Внешние зависимости"]
        direction TB
        paddle["«external»<br/>PaddleOCR"]
        ort["«external»<br/>onnxruntime"]
        skan["«external»<br/>skan"]
        cv["«external»<br/>OpenCV contrib"]
    end

    train["«offline»<br/>training/"]
    onnx[("models/arrows.onnx")]

    cli -.-> denoise
    cli -.-> ocr
    cli -.-> io
    denoise -.-> ocr
    denoise -.-> thick_lines
    denoise -.-> preprocess
    denoise -.-> io
    denoise -.-> ops
    denoise -.-> checks
    ocr -.-> ops
    preprocess -.-> checks
    ocr -.-> paddle
    denoise -.-> ort
    thick_lines -.-> skan
    preprocess -.-> cv
    train --> onnx
    onnx --> denoise
```

| Пакет | Роль | Входит в публичный интерфейс |
|---|---|---|
| `core.io` | Загрузка и сохранение изображения | `load_image`, `save_image` |
| [`core.preprocess`](modules/preprocess.md) | Серое, маска краски, нормализация масштаба по толщине контура | `preprocess`, `Preprocessed` |
| [`core.ocr`](modules/ocr.md) | Текст на чертеже, в том числе повёрнутый: полигоны и строки | `run`, `OCREngine`, `OcrResult` |
| [`core.thick_lines`](modules/thick_lines.md) | Граф линий по маске краски, деление линий на толстые и тонкие | `run`, `Lines`, `build_graph`, `classify_lines` |
| [`core.denoise`](modules/denoise.md) | Стрелки, правила чистки, чистый лист и виды | `run`, `DenoiseResult`, `ArrowModel`, `Sheet`, `Views` |
| `core.utils` | Проверки входа и операции над изображением | `check_image`, `rotate_image`, `scale_mask` |
| `cli` | Командная строка поверх `denoise.run` | `draftslice clean`, `draftslice views` |
| `training/` | Разметка → датасет → обучение YOLO-seg → экспорт ONNX. Не импортируется пакетом | `models/arrows.onnx` |

## Типы данных и объекты

Публичные типы модулей и связи между ними. Функции без состояния не показаны (они в таблице выше и на страницах
модулей). `Skeleton` — класс skan.

```mermaid
classDiagram
    direction LR

    namespace preprocess {
        class Preprocessed {
            <<frozen dataclass>>
            +ndarray img
            +ndarray gray
            +ndarray binary
            +float scale
            +float t_c
        }
    }

    namespace ocr {
        class PaddleOCRParameters {
            <<dataclass>>
            +float text_rec_score_thresh
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
    }

    namespace thick_lines {
        class Lines {
            <<frozen dataclass>>
            +ndarray ink
            +ndarray dist
            +Skeleton g
            +LineClasses cls
            +plot() ndarray
        }
        class LineClasses {
            <<frozen dataclass>>
            +ndarray thick
            +ndarray level
            +float thr
        }
    }

    namespace denoise {
        class ArrowModel {
            +predict(img, conf, overlap) ArrowResult
        }
        class ArrowResult {
            <<frozen dataclass>>
            +ndarray mask
            +ndarray xyxy
            +ndarray conf
        }
        class Sheet {
            +trim(mask) Sheet
            +remove(paths) Sheet
            +tables() ndarray
            +marks() ndarray
            +clean ndarray
            +segments Segments
        }
        class Segments {
            <<frozen dataclass>>
        }
        class Views {
            +crops(size) list
            +rotations(size) list
        }
        class View {
            <<frozen dataclass>>
            +tuple box
            +ndarray mask
            +ndarray img
        }
        class DenoiseResult {
            <<frozen dataclass>>
            +ndarray orig
            +float scale
            +ndarray mask
            +ndarray clean
            +plot() ndarray
            +save(path) Path
            +save_views(save_dir, stem, size) list
        }
    }

    OCREngine o-- PaddleOCRParameters
    OCREngine ..> OcrResult : создаёт
    Lines *-- LineClasses
    ArrowModel ..> ArrowResult : создаёт
    Sheet ..> Segments : создаёт
    Views o-- Sheet
    Views *-- "1..*" View
    DenoiseResult *-- Lines
    DenoiseResult *-- ArrowResult
    DenoiseResult *-- Sheet
    DenoiseResult *-- Views
```

## Последовательность вызовов

`denoise.run` — единственное место, где модули соединяются: остальные пакеты друг о друге не знают.

```mermaid
sequenceDiagram
    autonumber
    actor U as Вызывающий код
    participant D as denoise.run
    participant O as ocr.run
    participant P as preprocess
    participant T as thick_lines.run
    participant A as ArrowModel
    participant S as Sheet
    participant V as Views

    U->>+D: run(img, arrow_model, ocr_engine)
    D->>+O: run(img, ocr_engine)
    O-->>-D: list[OcrResult]
    D->>+P: preprocess(img, target, max_scale)
    P-->>-D: Preprocessed
    D->>D: полигоны × scale, clear_text(binary, polys)
    D->>+T: run(ink, t_c × scale)
    T-->>-D: Lines
    D->>+A: arrow_model(img, conf)
    A-->>-D: ArrowResult
    D->>+S: Sheet(ink, g, dist, thick, t_c, polys)
    S-->>-D: sheet
    D->>S: trim(arrows.mask)
    D->>S: remove(tables())
    D->>S: remove(marks())
    D->>+V: Views(sheet, img, scale, margin)
    V-->>-D: views
    D-->>-U: DenoiseResult
```

Масштабы разные: текст и стрелки ищутся на исходном листе, линии и правила чистки — в
рабочем масштабе `preprocess`. Результат возвращается в исходный масштаб делением на `Preprocessed.scale`
(`utils.ops.scale_mask`).

## Общие примитивы

Не самодостаточные модули пайплайна, используются несколькими пакетами.

| Файл | Функции | Для чего |
|---|---|---|
| `core/io.py` | `load_image`, `save_image` | Лист RGB из пути, байтов или массива; запись с кириллицей в пути |
| `core/utils/checks.py` | `check_image` | Проверка входа, как `ultralytics.utils.checks`: явный вызов в начале функции |
| `core/utils/ops.py` | `rotate_image`, `scale_mask` | Поворот листа с расширением холста; перенос маски между масштабами |

## Соглашения

- Аннотации — `np.ndarray`; форма и dtype указаны в докстринге. Своих alias типов нет.
- Ошибки — встроенные исключения: `TypeError`, `ValueError`, `FileNotFoundError`.
- Составной результат — `frozen dataclass`; правила чистки — методы объекта, общее считается один раз
  (`cached_property`).
- Изображение — RGB, `uint8`. Координаты пикселей — `(row, col)` в графе линий и `(x, y)` в боксах и полигонах.
- Модели (`ArrowModel`, `OCREngine`) создаются один раз и передаются в `run`.

Подробности каждого пакета — в [`modules/`](modules/), проблемы и отброшенные варианты — в [`decisions/`](decisions/).
