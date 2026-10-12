"""CLI draftslice: чистый лист и виды для ViT.

draftslice clean data/images -o runs/clean --plot
draftslice views data/images/1.jpg -o runs/views --size 224
"""

from pathlib import Path
from typing import Annotated

import typer

from draftslice.core.denoise.arrows import ArrowModel
from draftslice.core.io import load_image, save_image
from draftslice.core.ocr.engine import OCREngine, PaddleOCRParameters
from draftslice.pipeline import run

app = typer.Typer(help="Чистка чертежей: чистый лист без шума и виды для ViT.", no_args_is_help=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

Sources = Annotated[list[Path], typer.Argument(help="Листы или папки с листами.", exists=True)]
Models = Annotated[Path, typer.Option("--models", help="Папка моделей: arrows.onnx и модели PaddleOCR.")]
Conf = Annotated[float, typer.Option("--conf", help="Порог уверенности стрелок.")]


@app.command()
def clean(
    sources: Sources,
    out: Annotated[Path, typer.Option("--out", "-o", help="Папка результата.")] = Path("runs/clean"),
    models: Models = Path("models"),
    conf: Conf = 0.3,
    plot: Annotated[bool, typer.Option("--plot", help="Сохранить и картинку проверки {имя}_plot.png.")] = False,
) -> None:
    """Чистый лист в исходном масштабе: {имя}.png, фон белый."""
    for path, res in _run(sources, models, conf):
        typer.echo(f"{path.name}: видов {len(res.views)} -> {res.save(out / f'{path.stem}.png')}")
        if plot:
            save_image(out / f"{path.stem}_plot.png", res.plot())


@app.command()
def views(
    sources: Sources,
    out: Annotated[Path, typer.Option("--out", "-o", help="Папка результата.")] = Path("runs/views"),
    size: Annotated[int | None, typer.Option("--size", help="Квадрат для ViT; без него - вид как есть.")] = None,
    models: Models = Path("models"),
    conf: Conf = 0.3,
) -> None:
    """Виды листа: {имя}_{k}.png, k - от крупного к мелкому."""
    for path, res in _run(sources, models, conf):
        typer.echo(f"{path.name}: видов {len(res.save_views(out, path.stem, size))} -> {out}")


def _run(sources, models, conf):
    """Модели - один раз на все листы; по листу - (путь, результат)."""
    arrow_model = ArrowModel(models / "arrows.onnx")
    defaults = PaddleOCRParameters()
    ocr_engine = OCREngine(
        PaddleOCRParameters(
            text_detection_model_dir=str(models / Path(defaults.text_detection_model_dir).name),
            text_recognition_model_dir=str(models / Path(defaults.text_recognition_model_dir).name),
        )
    )
    for path in _images(sources):
        yield path, run(load_image(path), arrow_model, ocr_engine, conf=conf)


def _images(sources):
    """Файлы листов: файлы как есть, из папок - изображения по имени."""
    for src in sources:
        if src.is_dir():
            yield from sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        else:
            yield src


if __name__ == "__main__":
    app()
