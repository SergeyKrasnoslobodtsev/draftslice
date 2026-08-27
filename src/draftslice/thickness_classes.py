"""Разделение цепей на классы толщины.

На чертеже толщина линии несет смысл: контур детали толстый, размерные и выносные линии тонкие.
Абсолютные пороги для этого не годятся, потому что толщина зависит от разрешения листа и от того, чем
чертеж напечатан. Поэтому границы классов не задаются, а находятся по самому кадру одномерной
кластеризацией толщин.

Обучение идет не по всем цепям. Текстовые цепи исключаются: буквы тонкие и многочисленные, они
стягивают границу вниз и уводят часть контура в младший класс. Цепи с неопределенной толщиной
исключаются тоже. Вклад цепи можно взвесить ее длиной, тогда короткий мусор влияет на границы слабее,
чем длинный контур.

Классы нумеруются по возрастанию толщины, поэтому правило отбора выглядит как выбор класса и всего,
что толще.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from draftslice.chain_merging import StrokeChains
from draftslice.common_types import Floats, Ints, Mask


@dataclass(frozen=True, eq=False)
class ThicknessClasses:
    """Классы толщины, пронумерованные по возрастанию.

    Attributes
    ----------
    label : Ints
        Номер класса для каждой цепи, минус единица для текста и неопределенной толщины.
    centers : Floats
        Центры классов в пикселях, по возрастанию.
    edges : Floats
        Границы между соседними классами.
    """

    label: Ints
    centers: Floats
    edges: Floats

    @property
    def class_count(self) -> int:
        """Число классов."""
        return len(self.centers)


def fit_thickness_classes(
    chains: StrokeChains, class_count: int, text_share_limit: float = 0.5, weight_by_length: bool = True
) -> ThicknessClasses:
    """Найти границы классов толщины по цепям кадра.

    Parameters
    ----------
    chains : StrokeChains
        Цепи с измеренной толщиной.
    class_count : int
        Число классов.
    text_share_limit : float
        Доля текста, выше которой цепь исключается из обучения.
    weight_by_length : bool
        Взвешивать ли вклад цепи ее длиной.

    Returns
    -------
    ThicknessClasses
        Номера классов по цепям, центры и границы.

    Raises
    ------
    ValueError
        Если для обучения не осталось ни одной пригодной цепи.
    """
    usable = np.isfinite(chains.thickness) & (chains.text_share <= text_share_limit)
    if not usable.any():
        raise ValueError("нет цепей с определенной толщиной вне текста")

    model = KMeans(n_clusters=class_count, n_init=10, random_state=0).fit(
        chains.thickness[usable, None],
        sample_weight=chains.length[usable] if weight_by_length else None,
    )
    order = np.argsort(model.cluster_centers_.ravel())
    rank = np.empty(class_count, dtype=int)
    rank[order] = np.arange(class_count)
    centers = model.cluster_centers_.ravel()[order]

    label = np.full(chains.chain_count, -1, dtype=np.int64)
    label[usable] = rank[model.labels_]
    return ThicknessClasses(label=label, centers=centers, edges=(centers[:-1] + centers[1:]) / 2)


def select_chains_by_class(
    chains: StrokeChains, classes: ThicknessClasses, minimum_class: int, text_share_limit: float = 0.5
) -> Mask:
    """Отобрать цепи заданного класса толщины и выше.

    Parameters
    ----------
    chains : StrokeChains
        Цепи кадра.
    classes : ThicknessClasses
        Классы толщины.
    minimum_class : int
        Младший класс, который еще сохраняется.
    text_share_limit : float
        Доля текста, выше которой цепь удаляется независимо от класса.

    Returns
    -------
    Mask
        Маска сохраняемых цепей.
    """
    return (classes.label >= minimum_class) & (chains.text_share <= text_share_limit)
