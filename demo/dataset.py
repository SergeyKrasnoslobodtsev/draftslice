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

from draftslice.common_types import Floats
from draftslice.image_preparation import load_bgr_image
from draftslice.pipeline import extract_views
from draftslice.pipeline_parameters import PipelineParameters
from draftslice.text_detection import TextProbabilityDetector


@dataclass(frozen=True)
class SketchEmbeddings:
    """Эмбеддинги всех видов одного эскиза.

    Attributes
    ----------
    name : str
        Имя файла эскиза, с расширением, без каталога.
    embeddings : Floats
        Эмбеддинги видов, по строке на вид.
    """

    name: str
    embeddings: Floats


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
    detector: TextProbabilityDetector,
    embedder: DinoEmbedder,
    target_size: int,
    parameters: PipelineParameters | None = None,
) -> EmbeddingDataset:
    """Построить датасет эмбеддингов по списку файлов эскизов.

    Parameters
    ----------
    source_files : list[Path]
        Пути к файлам эскизов.
    detector : TextProbabilityDetector
        Загруженная модель детекции текста.
    embedder : DinoEmbedder
        Загруженная модель эмбеддингов.
    target_size : int
        Сторона квадрата кропа вида под вход модели эмбеддинга.
    parameters : PipelineParameters | None
        Пороги пайплайна очистки, None означает значения по умолчанию.

    Returns
    -------
    EmbeddingDataset
        Эмбеддинги видов каждого эскиза, в порядке `source_files`.
    """
    sketches = []
    for source_path in tqdm(source_files, desc="эскизы"):
        image = load_bgr_image(source_path)
        views = extract_views(image, detector, target_size, parameters)
        embeddings = embed_views(embedder, views)
        sketches.append(SketchEmbeddings(name=source_path.name, embeddings=embeddings))
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
