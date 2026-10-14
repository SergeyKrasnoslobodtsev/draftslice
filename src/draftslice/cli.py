"""CLI draftslice: чистый лист и виды для ViT.

draftslice clean data/images -o runs/clean --plot
draftslice views data/images/1.jpg -o runs/views --size 224
draftslice clean data/images -o runs/clean --workers 8
"""

from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from draftslice.core.io import save_image
from draftslice.pipeline import run_batch

app = typer.Typer(help="Чистка чертежей: чистый лист без шума и виды для ViT.", no_args_is_help=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

Sources = Annotated[list[Path], typer.Argument(help="Листы или папки с листами.", exists=True)]
Models = Annotated[Path, typer.Option("--models", help="Папка моделей: arrows.onnx и модели PaddleOCR.")]
Conf = Annotated[float, typer.Option("--conf", help="Порог уверенности стрелок.")]
Workers = Annotated[
    int, typer.Option("--workers", "-j", min=1, help="Процессов: листы параллельно, модели в каждом, GPU общий.")
]


@app.command()
def clean(
    sources: Sources,
    out: Annotated[Path, typer.Option("--out", "-o", help="Папка результата.")] = Path("runs/clean"),
    models: Models = Path("models"),
    conf: Conf = 0.3,
    plot: Annotated[bool, typer.Option("--plot", help="Сохранить и картинку проверки {имя}_plot.png.")] = False,
    workers: Workers = 1,
) -> None:
    """Чистый лист в исходном масштабе: {имя}.png, фон белый."""
    _run(sources, models, conf, workers, partial(_save_clean, out=out, plot=plot))


@app.command()
def views(
    sources: Sources,
    out: Annotated[Path, typer.Option("--out", "-o", help="Папка результата.")] = Path("runs/views"),
    size: Annotated[int | None, typer.Option("--size", help="Квадрат для ViT; без него - вид как есть.")] = None,
    models: Models = Path("models"),
    conf: Conf = 0.3,
    workers: Workers = 1,
) -> None:
    """Виды листа: {имя}_{k}.png, k - от крупного к мелкому."""
    _run(sources, models, conf, workers, partial(_save_views, out=out, size=size))


def _save_clean(path, res, out, plot):
    msg = f"{path.name}: видов {len(res.views)} -> {res.save(out / f'{path.stem}.png')}"
    if plot:
        save_image(out / f"{path.stem}_plot.png", res.plot())
    return msg


def _save_views(path, res, out, size):
    return f"{path.name}: видов {len(res.save_views(out, path.stem, size))} -> {out}"


def _run(sources, models, conf, workers, save):
    """Листы пачкой (run_batch); ошибка листа печатается и не останавливает остальные, код выхода тогда 1."""
    failed = 0
    for r in run_batch(_images(sources), models, save, workers, conf):
        typer.echo(r.value if r.error is None else f"{r.path.name}: ошибка {r.error}", err=r.error is not None)
        failed += r.error is not None
    if failed:
        typer.echo(f"Листов с ошибкой: {failed}", err=True)
        raise typer.Exit(1)


def _images(sources):
    """Файлы листов: файлы как есть, из папок - изображения по имени."""
    for src in sources:
        if src.is_dir():
            yield from sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
        else:
            yield src


if __name__ == "__main__":
    app()
