"""Точка входа `denoise`: лист целиком - текст, масштаб, линии, стрелки, правила чистки, виды.

Интерфейс как у Ultralytics: модели создаются один раз и передаются в `run`, результат - объект с `plot`/`save`.

    arrow_model = ArrowModel("models/arrows.onnx")
    ocr_engine = OCREngine(PaddleOCRParameters())
    res = run(img, arrow_model, ocr_engine)
    res.clean               # чистый лист RGB в исходном масштабе
    res.views               # виды для ViT
    res.plot()              # лист: чистое - чёрным, снятое - бледным, рамки видов
    res.save("out/1.png")
    res.save_views("out/views", "1", size=224)
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from draftslice.core import preprocess as pp
from draftslice.core.denoise.arrows import ArrowModel, ArrowResult
from draftslice.core.denoise.sheet import Sheet
from draftslice.core.denoise.text import clear_text
from draftslice.core.denoise.views import Views
from draftslice.core.io import save_image
from draftslice.core.ocr import pipeline as ocr
from draftslice.core.ocr.engine import OCREngine
from draftslice.core.thick_lines import pipeline as thick_lines
from draftslice.core.thick_lines.pipeline import Lines
from draftslice.core.utils.ops import scale_mask


@dataclass(frozen=True)
class DenoiseResult:
    """Лист после чистки.

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
) -> DenoiseResult:
    """Чистит лист и режет его на виды.

    Порядок: текст (OCR), рабочий масштаб (`preprocess`), краска без текста, линии (`thick_lines`), стрелки на
    исходном листе, правила `Sheet` (стрелки, таблицы, обозначения), виды.

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
    DenoiseResult
        Чистый лист, виды и промежуточные результаты.
    """
    texts = ocr.run(img, engine=ocr_engine)
    prep = pp.preprocess(img, target=target, max_scale=max_scale)
    polys = [r.poly * prep.scale for r in texts]
    t_c = prep.t_c * prep.scale
    lines = thick_lines.run(clear_text(prep.binary, polys), t_c)
    arrows = arrow_model(img, conf=conf)
    sheet = Sheet(lines.ink, lines.g, lines.dist, lines.cls.thick, t_c, polys)
    sheet.trim(arrows.mask).remove(sheet.tables())
    sheet.remove(sheet.marks())
    return DenoiseResult(img, prep.scale, lines, arrows, sheet, Views(sheet, img, prep.scale, margin))
