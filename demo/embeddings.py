"""Эмбеддинги видов через DINO.

Модель и её препроцессор грузятся один раз и передаются дальше как аргумент, а не создаются внутри
функции извлечения: загрузка весов дорога, а эмбеддинги считаются для многих эскизов подряд.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_NAME = "facebook/dinov2-small"
"""Модель по умолчанию. DINOv3 на Hugging Face закрыта лицензией с ожиданием одобрения — DINOv2 та
же линейка self-supervised ViT от Meta, но доступна без запроса доступа. Чтобы переключиться на
DINOv3 после одобрения, достаточно передать другое имя в load_dino_embedder."""


@dataclass(frozen=True)
class DinoEmbedder:
    """Загруженная модель DINO вместе с её препроцессором.

    Attributes
    ----------
    processor : AutoImageProcessor
        Препроцессор модели: ресайз и нормализация под конкретный чекпоинт.
    model : AutoModel
        Загруженная модель DINO в режиме инференса.
    device : str
        Устройство, на котором считается инференс.
    """

    processor: AutoImageProcessor
    model: AutoModel
    device: str


def load_dino_embedder(model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu") -> DinoEmbedder:
    """Загрузить модель DINO и её препроцессор.

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


def embed_views(embedder: DinoEmbedder, views: list[np.ndarray]) -> np.ndarray:
    """Посчитать эмбеддинг каждого вида одного эскиза.

    Parameters
    ----------
    embedder : DinoEmbedder
        Загруженная модель и препроцессор.
    views : list[np.ndarray]
        Виды одного эскиза: RGB `(size, size, 3)`, `uint8`, из `Views.crops`.

    Returns
    -------
    np.ndarray
        Эмбеддинги `(n_views, dim)`, по строке на вид, в том же порядке, что и `views`.

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
