"""Пайплайн листа: текст, рабочий масштаб, линии, стрелки, правила чистки, виды.

```python
arrow_model = ArrowModel("models/arrows.onnx")
ocr_engine = OCREngine(PaddleOCRParameters())
res = run(img, arrow_model, ocr_engine)
res.clean               # чистый лист RGB в исходном масштабе
res.views               # виды для ViT
res.plot()              # лист: чистое - как в оригинале, снятое - бледным, рамки видов
res.save("out/1.png")
res.save_views("out/views", "1", size=224)
```

Пачка листов, по процессу на группу листов (`run_batch`): модели грузятся в каждом процессе, GPU общий.

```python
def save(path, res):  # считается в процессе-воркере, функция уровня модуля
    return res.save(f"out/{path.stem}.png")

for r in run_batch(paths, "models", save, workers=6):
    print(r.path.name, r.error or r.value)
```
"""

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from draftslice.core import preprocess as pp
from draftslice.core.denoise import pipeline as denoise
from draftslice.core.denoise.arrows import ArrowModel, ArrowResult
from draftslice.core.denoise.sheet import Sheet
from draftslice.core.denoise.text import clear_text
from draftslice.core.denoise.views import Views
from draftslice.core.io import load_image, save_image
from draftslice.core.ocr import pipeline as ocr
from draftslice.core.ocr.engine import OCREngine, PaddleOCRParameters
from draftslice.core.thick_lines import pipeline as thick_lines
from draftslice.core.thick_lines.pipeline import Lines
from draftslice.core.utils.checks import check_image
from draftslice.core.utils.ops import scale_mask


@dataclass(frozen=True)
class Result:
    """Лист после чистки и его виды.

    Attributes
    ----------
    orig : np.ndarray[uint8]
        Исходный лист RGB `(H, W, 3)`.
    scale : float
        Во сколько раз рабочий масштаб больше исходного (`Preprocessed.scale`).
    lines : Lines
        Краска без текста, граф и классы линий в рабочем масштабе.
    arrows : ArrowResult
        Стрелки в исходном масштабе.
    sheet : Sheet
        Лист после правил чистки.
    views : Views
        Виды листа.
    """

    orig: np.ndarray
    scale: float
    lines: Lines
    arrows: ArrowResult
    sheet: Sheet
    views: Views

    @cached_property
    def mask(self) -> np.ndarray:
        """Чистая маска `0/255` `(H, W)` в исходном масштабе."""
        return scale_mask(self.sheet.clean, self.orig.shape[:2], self.scale)

    @cached_property
    def clean(self) -> np.ndarray:
        """Чистый лист RGB `(H, W, 3)`: пиксели исходного листа под чистой маской, фон белый (OQ-02)."""
        out = np.full_like(self.orig, 255)
        on = self.mask > 0
        out[on] = self.orig[on]
        return out

    def plot(self) -> np.ndarray:
        """Лист для проверки: чистое - как в оригинале, снятое - бледным, рамки видов зелёным с номером.

        Returns
        -------
        np.ndarray[uint8]
            RGB `(H, W, 3)`.
        """
        out = (255 - (255 - self.orig.astype(np.int32)) // 4).astype(np.uint8)  # снятое - бледным
        on = self.mask > 0
        out[on] = self.orig[on]
        lw = max(1, round(max(self.orig.shape[:2]) / 600))
        for k, v in enumerate(self.views):
            x0, y0, x1, y1 = v.box
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 170, 0), lw)
            cv2.putText(out, str(k), (x0 + lw, y0 + 12 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * lw, (0, 170, 0), lw)
        return out

    def save(self, path: str | Path) -> Path:
        """Сохраняет чистый лист `clean`.

        Parameters
        ----------
        path : str | Path
            Путь файла, формат - по расширению.

        Returns
        -------
        Path
            Путь сохранённого файла.
        """
        return save_image(path, self.clean)

    def save_views(self, save_dir: str | Path, stem: str, size: int | None = None) -> list[Path]:
        """Сохраняет виды: `{stem}_{k}.png`, k - номер вида от крупного к мелкому.

        Parameters
        ----------
        save_dir : str | Path
            Папка; создаётся.
        stem : str
            Начало имени файлов, обычно имя листа.
        size : int | None
            Сторона квадрата для ViT (`Views.crops`); `None` - вид в исходном масштабе (`View.img`).

        Returns
        -------
        list[Path]
            Пути сохранённых файлов.
        """
        imgs = self.views.crops(size) if size else [v.img for v in self.views]
        return [save_image(Path(save_dir) / f"{stem}_{k}.png", img) for k, img in enumerate(imgs)]


def run(
    img: np.ndarray,
    arrow_model: ArrowModel,
    ocr_engine: OCREngine,
    target: float = 9.0,
    max_scale: float = 2.0,
    conf: float = 0.3,
    margin: float = 3.0,
) -> Result:
    """Чистит лист и режет его на виды.

    Порядок: текст (`ocr`), рабочий масштаб (`preprocess`), краска без текста (`denoise.text`), линии
    (`thick_lines`), стрелки на исходном листе, правила чистки (`denoise`), виды.

    Parameters
    ----------
    img : np.ndarray[uint8]
        Исходный лист RGB `(H, W, 3)`.
    arrow_model : ArrowModel
        Модель стрелок, создаётся один раз на пачку листов.
    ocr_engine : OCREngine
        Движок OCR, создаётся один раз на пачку листов.
    target : float
        Толщина контура в рабочем масштабе, px (`preprocess`).
    max_scale : float
        Наибольшее увеличение листа (`preprocess`).
    conf : float
        Порог уверенности стрелок.
    margin : float
        Отступ вокруг вида в толщинах контура (`Views`).

    Returns
    -------
    Result
        Чистый лист, виды и промежуточные результаты.

    Raises
    ------
    TypeError
        `img` не `(H, W, 3)` uint8.
    """
    check_image(img, channels=3, dtype=np.uint8)
    texts = ocr.run(img, engine=ocr_engine)
    prep = pp.preprocess(img, target=target, max_scale=max_scale)
    polys = [r.poly * prep.scale for r in texts]
    t_c = prep.t_c * prep.scale
    lines = thick_lines.run(clear_text(prep.binary, polys), t_c)
    arrows = arrow_model(img, conf=conf)
    sheet = denoise.run(lines, arrows.mask, t_c, polys)
    return Result(img, prep.scale, lines, arrows, sheet, Views(sheet, img, prep.scale, margin))


@dataclass(frozen=True)
class SheetResult:
    """Итог одного листа пачки `run_batch`.

    Attributes
    ----------
    path : Path
        Лист.
    value : Any
        Что вернула `fn` на этом листе; `None`, если лист упал.
    error : str | None
        `ТипОшибки: сообщение`, если лист упал; иначе `None`.
    """

    path: Path
    value: Any
    error: str | None


def run_batch(
    paths: Iterable[Path],
    models: str | Path,
    fn: Callable[[Path, Result], Any],
    workers: int = 1,
    conf: float = 0.3,
) -> Iterator[SheetResult]:
    """Пачка листов: `run` и `fn` по каждому листу, в одном процессе или в workers процессах.

    Каждый процесс один раз грузит модели из `models` и считает листы по одному, пока они есть. `fn` вызывается
    в том же процессе, что считал лист: тяжёлый `Result` между процессами не передаётся, только то, что вернула
    `fn`. Процессы стартуют через spawn (fork после инициализации CUDA и Paddle небезопасен), поэтому `fn` - функция
    уровня модуля или `functools.partial` от неё. Потоки OpenCV делятся между процессами поровну.

    Каждый процесс держит на GPU около 3.5 ГБ (OCR и стрелки); если памяти GPU не хватает, всё замедляется в разы.

    Parameters
    ----------
    paths : Iterable[Path]
        Листы.
    models : str | Path
        Папка моделей: `arrows.onnx` и модели PaddleOCR.
    fn : Callable[[Path, Result], Any]
        Что сделать с результатом листа: сохранить, вырезать виды. Возвращаемое значение уходит в `SheetResult.value`.
    workers : int
        Число процессов; 1 - в текущем процессе.
    conf : float
        Порог уверенности стрелок.

    Returns
    -------
    Iterator[SheetResult]
        По листу в порядке `paths`, по мере готовности. Ошибка листа (битый файл, сбой конвейера) не останавливает
        остальные: она в `SheetResult.error`.

    Raises
    ------
    ValueError
        `workers` меньше 1.
    """
    if workers < 1:
        raise ValueError(f"Некорректное число процессов: {workers=}")
    init = (Path(models), fn, conf, workers)
    if workers == 1:
        _init_worker(*init)
        try:
            yield from map(_run_one, paths)
        finally:
            _ctx.clear()  # модели держат ~4 ГБ RAM и ~3.5 ГБ GPU: после пачки не нужны
        return
    pool = mp.get_context("spawn").Pool(workers, _init_worker, init)
    try:
        yield from pool.imap(_run_one, paths)
    except BaseException:  # Ctrl+C, сбой, вызывающий бросил итерацию: процессы снимаются сразу
        pool.terminate()
        raise
    else:
        pool.close()  # штатно процессы выходят сами; terminate снял бы их SIGTERM
    finally:
        pool.join()


_ctx = {}  # модели и параметры процесса: _init_worker заполняет, _run_one читает


def _init_worker(models, fn, conf, workers):
    """Модели - один раз на процесс; потоки OpenCV делятся между процессами."""
    if workers > 1:
        cv2.setNumThreads(max(1, os.cpu_count() // workers))
    defaults = PaddleOCRParameters()
    ocr_params = PaddleOCRParameters(
        text_detection_model_dir=str(models / Path(defaults.text_detection_model_dir).name),
        text_recognition_model_dir=str(models / Path(defaults.text_recognition_model_dir).name),
    )
    _ctx.update(arrow=ArrowModel(models / "arrows.onnx"), ocr=OCREngine(ocr_params), fn=fn, conf=conf)


def _run_one(path):
    try:
        res = run(load_image(path), _ctx["arrow"], _ctx["ocr"], conf=_ctx["conf"])
        return SheetResult(path, _ctx["fn"](path, res), None)
    except Exception as e:  # один битый лист не должен ронять пачку
        return SheetResult(path, None, f"{type(e).__name__}: {e}")
