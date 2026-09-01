"""Поиск похожих эскизов по эмбеддингам их видов.

У эскиза не один эмбеддинг, а несколько — по одному на вид, и их количество у разных эскизов разное.
Поэтому сходство между двумя эскизами не сводится к одному косинусу, а зависит от того, как свести
набор попарных сходств вид-вид к одному числу.
"""

from collections.abc import Callable
from enum import Enum

import numpy as np

from dataset import EmbeddingDataset
from draftslice.common_types import Floats


class MatchStrategy(Enum):
    """Способ свести попарные сходства видов двух эскизов к одному числу.

    Attributes
    ----------
    BEST_PAIR : str
        Сходство — максимум по всем парам вид-вид: эскизы похожи, если совпал хотя бы один вид.
    MEAN_EMBEDDING : str
        Эмбеддинги видов усредняются в один вектор на эскиз, сходство — косинус между средними.
    MEAN_PAIR : str
        Сходство — среднее по всем парам вид-вид: учитывает совпадение всего набора видов.
    """

    BEST_PAIR = "best_pair"
    MEAN_EMBEDDING = "mean_embedding"
    MEAN_PAIR = "mean_pair"


def normalize_rows(embeddings: Floats) -> Floats:
    """Нормировать строки на единичную длину для косинусного сходства.

    Parameters
    ----------
    embeddings : Floats
        Эмбеддинги, по строке на вектор.

    Returns
    -------
    Floats
        Эмбеддинги той же формы, каждая строка единичной длины.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def score_best_pair(query: Floats, candidate: Floats) -> float:
    """Сходство как максимум по всем парам вид-вид.

    Parameters
    ----------
    query, candidate : Floats
        Эмбеддинги видов запроса и кандидата.

    Returns
    -------
    float
        Косинусное сходство самой похожей пары видов.
    """
    similarity = normalize_rows(query) @ normalize_rows(candidate).T
    return float(similarity.max())


def score_mean_embedding(query: Floats, candidate: Floats) -> float:
    """Сходство между усредненными по видам эмбеддингами.

    Parameters
    ----------
    query, candidate : Floats
        Эмбеддинги видов запроса и кандидата.

    Returns
    -------
    float
        Косинусное сходство средних векторов.
    """
    query_mean = normalize_rows(query.mean(axis=0, keepdims=True))
    candidate_mean = normalize_rows(candidate.mean(axis=0, keepdims=True))
    return float((query_mean @ candidate_mean.T).item())


def score_mean_pair(query: Floats, candidate: Floats) -> float:
    """Сходство как среднее по всем парам вид-вид.

    Parameters
    ----------
    query, candidate : Floats
        Эмбеддинги видов запроса и кандидата.

    Returns
    -------
    float
        Среднее косинусное сходство по всем парам видов.
    """
    similarity = normalize_rows(query) @ normalize_rows(candidate).T
    return float(similarity.mean())


_SCORERS: dict[MatchStrategy, Callable[[Floats, Floats], float]] = {
    MatchStrategy.BEST_PAIR: score_best_pair,
    MatchStrategy.MEAN_EMBEDDING: score_mean_embedding,
    MatchStrategy.MEAN_PAIR: score_mean_pair,
}


def find_similar_sketches(
    query: Floats, dataset: EmbeddingDataset, strategy: MatchStrategy, top_k: int = 5
) -> list[tuple[str, float]]:
    """Найти самые похожие на запрос эскизы в датасете.

    Parameters
    ----------
    query : Floats
        Эмбеддинги видов запроса.
    dataset : EmbeddingDataset
        Эмбеддинги эскизов, среди которых ищем.
    strategy : MatchStrategy
        Способ свести сходство видов к одному числу на пару эскизов.
    top_k : int
        Сколько лучших совпадений вернуть.

    Returns
    -------
    list[tuple[str, float]]
        Имя эскиза и сходство с запросом, по убыванию сходства, не больше `top_k` записей.
    """
    scorer = _SCORERS[strategy]
    scored = [(sketch.name, scorer(query, sketch.embeddings)) for sketch in dataset.sketches]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
