"""Датасет эмбеддингов видов: имя эскиза и эмбеддинги его видов, хранение в одном npz.

Число видов у разных эскизов разное — один лист может дать один вид, другой пять, поэтому
эмбеддинги нельзя сложить в один прямоугольный массив. Каждый эскиз хранится под собственным ключом
в npz, а не общим тензором фиксированной формы.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from embeddings import DinoEmbedder, embed_views
from tqdm import tqdm

from draftslice.pipeline import Result, run_batch


@dataclass(frozen=True)
class SketchEmbeddings:
    """Эмбеддинги всех видов одного эскиза.

    Attributes
    ----------
    name : str
        Имя файла эскиза, с расширением, без каталога.
    embeddings : np.ndarray
        Эмбеддинги видов `(n_views, dim)`, по строке на вид.
    """

    name: str
    embeddings: np.ndarray


@dataclass(frozen=True)
class EmbeddingDataset:
    """Эмбеддинги видов всех эскизов датасета.

    Attributes
    ----------
    sketches : list[SketchEmbeddings]
        Эмбеддинги по каждому эскизу.
    """

    sketches: list[SketchEmbeddings]


def build_dataset(
    source_files: list[Path],
    models: Path,
    embedder: DinoEmbedder,
    target_size: int,
    workers: int = 1,
) -> EmbeddingDataset:
    """Построить датасет эмбеддингов по списку файлов эскизов.

    Эскизы чистятся `draftslice.pipeline.run_batch` в workers процессах, каждый со своими моделями стрелок и OCR;
    эмбеддинги видов считаются в текущем процессе, пока процессы чистят следующие эскизы.

    Parameters
    ----------
    source_files : list[Path]
        Пути к файлам эскизов.
    models : Path
        Папка моделей `draftslice`: `arrows.onnx` и модели PaddleOCR.
    embedder : DinoEmbedder
        Загруженная модель эмбеддингов.
    target_size : int
        Сторона квадрата вида под вход модели эмбеддинга.
    workers : int
        Число процессов чистки; каждый держит на GPU около 3.5 ГБ.

    Returns
    -------
    EmbeddingDataset
        Эмбеддинги видов каждого эскиза в порядке `source_files`; эскиз, на котором не нашлось ни одного вида
        или чистка упала, пропускается (ошибка печатается).
    """
    sketches = []
    batch = run_batch(source_files, models, partial(_crops, size=target_size), workers)
    for r in tqdm(batch, total=len(source_files), desc="эскизы"):
        if r.error is not None:
            tqdm.write(f"{r.path.name}: ошибка {r.error}")
        elif r.value:
            sketches.append(SketchEmbeddings(name=r.path.name, embeddings=embed_views(embedder, r.value)))
    return EmbeddingDataset(sketches=sketches)


def _crops(path: Path, res: Result, size: int) -> list[np.ndarray]:
    """Виды эскиза квадратами под вход эмбеддера; считается в процессе чистки."""
    return res.views.crops(size)


def save_dataset(dataset: EmbeddingDataset, path: Path) -> None:
    """Сохранить датасет эмбеддингов в один npz-файл.

    Parameters
    ----------
    dataset : EmbeddingDataset
        Эмбеддинги видов всех эскизов.
    path : Path
        Путь к npz-файлу.
    """
    arrays = {"names": np.array([sketch.name for sketch in dataset.sketches])}
    for sketch in dataset.sketches:
        arrays[f"embeddings::{sketch.name}"] = sketch.embeddings

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_dataset(path: Path) -> EmbeddingDataset:
    """Прочитать датасет эмбеддингов из npz-файла.

    Parameters
    ----------
    path : Path
        Путь к npz-файлу, сохраненному `save_dataset`.

    Returns
    -------
    EmbeddingDataset
        Эмбеддинги видов всех эскизов.
    """
    with np.load(path) as stored:
        names = [str(name) for name in stored["names"]]
        sketches = [SketchEmbeddings(name=name, embeddings=stored[f"embeddings::{name}"]) for name in names]
    return EmbeddingDataset(sketches=sketches)
