"""Эмбеддинги видов через DINO.

Модель и её препроцессор грузятся один раз и передаются дальше как аргумент, а не создаются внутри
функции извлечения: загрузка весов дорога, а эмбеддинги считаются для многих эскизов подряд.
"""

from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModel

from draftslice.common_types import Floats, Mat

DEFAULT_MODEL_NAME = "facebook/dinov2-small"
"""Модель по умолчанию. DINOv3 на Hugging Face закрыта лицензией с ожиданием одобрения — DINOv2 та
же линейка self-supervised ViT от Meta, но доступна без запроса доступа. Чтобы переключиться на
DINOv3 после одобрения, достаточно передать другое имя в load_dino_embedder."""


@dataclass(frozen=True)
class DinoEmbedder:
    """Загруженная модель DINOv3 вместе с её препроцессором.

    Attributes
    ----------
    processor : AutoImageProcessor
        Препроцессор модели: ресайз и нормализация под конкретный чекпоинт.
    model : AutoModel
        Загруженная модель DINOv3 в режиме инференса.
    device : str
        Устройство, на котором считается инференс.
    """

    processor: AutoImageProcessor
    model: AutoModel
    device: str


def load_dino_embedder(model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> DinoEmbedder:
    """Загрузить модель DINOv3 и её препроцессор.

    Parameters
    ----------
    model_name : str
        Имя чекпоинта модели на Hugging Face Hub.
    device : str
        Устройство для инференса, например "cpu" или "cuda".

    Returns
    -------
    DinoEmbedder
        Модель и препроцессор, готовые к извлечению эмбеддингов.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return DinoEmbedder(processor=processor, model=model, device=device)


def embed_views(embedder: DinoEmbedder, views: list[Mat]) -> Floats:
    """Посчитать эмбеддинг каждого вида одного эскиза.

    Parameters
    ----------
    embedder : DinoEmbedder
        Загруженная модель и препроцессор.
    views : list[Mat]
        Виды одного эскиза как RGB-картинки, из `draftslice.pipeline.extract_views`.

    Returns
    -------
    Floats
        Эмбеддинги, по строке на вид, в том же порядке, что и `views`.

    Notes
    -----
    Эмбеддингом вида берется CLS-токен последнего слоя — стандартное глобальное представление
    картинки для моделей семейства DINO.
    """
    inputs = embedder.processor(images=views, return_tensors="pt").to(embedder.device)
    with torch.no_grad():
        outputs = embedder.model(**inputs)
    cls_tokens = outputs.last_hidden_state[:, 0, :]
    return cls_tokens.cpu().numpy()
