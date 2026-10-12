"""Датасет эмбеддингов видов: имя эскиза и эмбеддинги его видов, хранение в одном npz.

Число видов у разных эскизов разное — один лист может дать один вид, другой пять, поэтому
эмбеддинги нельзя сложить в один прямоугольный массив. Каждый эскиз хранится под собственным ключом
в npz, а не общим тензором фиксированной формы.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from embeddings import DinoEmbedder, embed_views
from tqdm import tqdm

from draftslice.core.denoise.arrows import ArrowModel
from draftslice.core.io import load_image
from draftslice.core.ocr.engine import OCREngine
from draftslice.pipeline import run


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
    arrow_model: ArrowModel,
    ocr_engine: OCREngine,
    embedder: DinoEmbedder,
    target_size: int,
) -> EmbeddingDataset:
    """Построить датасет эмбеддингов по списку файлов эскизов.

    Parameters
    ----------
    source_files : list[Path]
        Пути к файлам эскизов.
    arrow_model : ArrowModel
        Модель стрелок для `draftslice.pipeline.run`.
    ocr_engine : OCREngine
        Движок OCR для `draftslice.pipeline.run`.
    embedder : DinoEmbedder
        Загруженная модель эмбеддингов.
    target_size : int
        Сторона квадрата вида под вход модели эмбеддинга.

    Returns
    -------
    EmbeddingDataset
        Эмбеддинги видов каждого эскиза в порядке `source_files`; эскиз, на котором не нашлось ни одного вида,
        пропускается.
    """
    sketches = []
    for path in tqdm(source_files, desc="эскизы"):
        views = run(load_image(path), arrow_model, ocr_engine).views.crops(target_size)
        if views:
            sketches.append(SketchEmbeddings(name=path.name, embeddings=embed_views(embedder, views)))
    return EmbeddingDataset(sketches=sketches)


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
