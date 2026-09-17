"""Объединение мелких компонент чертежа в чанки и удаление краевых артефактов.

Мелкие компоненты (`small_labels`) объединяются через Union-Find одним проходом по всем
парам: два компонента объединяются в один чанк, если их bbox лежат в одной строке
(`same_row` - пересекаются по Y и зазор по X не больше среднего из их ширин) или в
одной колонке (`same_col` - пересекаются по X и зазор по Y не больше среднего из их
высот). Раздельных стадий "сначала строки потом колонки" нет - транзитивность
Union-Find протягивает цепочку объединений сама. Краевые компоненты (`edge_labels`)
зануляются в итоговой карте меток и полностью исключаются из результата.

`.. warning::`

Краевые компоненты могут касаться основных компонентов. Например рамка вокруг чертежа
может быть краевой компонентой, но при этом касаться чертежа внутри через сноски или дополнительные линии.

"""

import numpy as np

from draftslice.core.region_segmentation.comp_classification import ComponentClassification
from draftslice.core.region_segmentation.comp_labeling import ComponentLabels
from draftslice.core.types import Array, Array2D


def _pairwise_overlap(lo: Array[np.int32], hi: Array[np.int32]) -> Array2D[np.bool_]:
    """Матрица: пересекаются ли отрезки [lo_i, hi_i) и [lo_j, hi_j) по одной оси."""
    return np.maximum(lo[:, None], lo[None, :]) < np.minimum(hi[:, None], hi[None, :])


def _pairwise_gap(lo: Array[np.int32], hi: Array[np.int32]) -> Array2D[np.int32]:
    """Матрица зазоров между отрезками [lo_i, hi_i) и [lo_j, hi_j) по одной оси."""
    return np.maximum(lo[:, None], lo[None, :]) - np.minimum(hi[:, None], hi[None, :])


def _pairwise_mean(size: Array[np.int32]) -> Array2D[np.float64]:
    """Матрица среднего размера пары компонент по одной оси."""
    return (size[:, None] + size[None, :]) / 2


def _find_merge_candidates(
    comps: ComponentLabels, small_labels: Array[np.intp]
) -> tuple[Array[np.intp], Array[np.intp]]:
    """Находит все пары мелких компонент, лежащие в одной строке или колонке."""
    small_left = comps.left[small_labels - 1]
    small_top = comps.top[small_labels - 1]
    small_width = comps.width[small_labels - 1]
    small_height = comps.height[small_labels - 1]
    small_right = small_left + small_width
    small_bottom = small_top + small_height

    same_row = _pairwise_overlap(small_top, small_bottom) & (
        _pairwise_gap(small_left, small_right) <= _pairwise_mean(small_width)
    )
    same_col = _pairwise_overlap(small_left, small_right) & (
        _pairwise_gap(small_top, small_bottom) <= _pairwise_mean(small_height)
    )
    should_merge = same_row | same_col

    n = len(small_labels)
    # TODO: skipy.spatial.cKDTree возможно ускорит поиск кандидатов
    # на слияние и уменьшит объем памяти на больших матрицах.
    # Дешевле: sweep-line по компонентам, отсортированным по top,
    # или scipy.spatial.cKDTree вместо плотной O(n²) матрицы.
    # Необходимо замерить производительность и объем памяти
    # для больших изображений.
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
    row_indices, col_indices = np.nonzero(should_merge & upper_triangle)
    return row_indices, col_indices


def _find_chunk_root(label: int, chunk_parent: Array[np.int32]) -> int:
    """Находит представителя чанка для метки со сжатием пути (path halving).

    Parameters
    ----------
    label : int
        Метка компоненты, для которой ищется представитель чанка.
    chunk_parent : Array[np.int32]
        Таблица Union-Find для поиска представителей чанков.

    Returns
    -------
    int
        Метка представителя чанка.
    """
    while chunk_parent[label] != label:
        chunk_parent[label] = chunk_parent[chunk_parent[label]]
        label = chunk_parent[label]
    return label


def merge_into_chunks(comps: ComponentLabels, classification: ComponentClassification) -> Array2D[np.int32]:
    """Объединяет мелкие компоненты чертежа в чанки и удаляет краевые артефакты.

    Parameters
    ----------
    comps : ComponentLabels
        Метки компонентов и их статистики.
    classification : ComponentClassification
        Классификация компонент на мелкие, seed'ы и краевые артефакты.

    Returns
    -------
    Array2D[np.int32]
        Карта меток на всем изображении после объединения мелких компонент в чанки
        и зануления краевых артефактов.
    """
    n = len(comps.area)
    small_labels = classification.small_labels
    chunk_parent = np.arange(n + 1, dtype=np.int32)

    pair_i, pair_j = _find_merge_candidates(comps, small_labels)
    for i, j in zip(pair_i, pair_j, strict=True):
        label_a = int(small_labels[i])
        label_b = int(small_labels[j])
        root_a = _find_chunk_root(label_a, chunk_parent)
        root_b = _find_chunk_root(label_b, chunk_parent)
        if root_a != root_b:
            chunk_parent[root_a] = root_b

    chunk_id = np.arange(n + 1, dtype=np.int32)
    chunk_id[small_labels] = np.array([_find_chunk_root(int(label), chunk_parent) for label in small_labels])
    chunk_id[classification.edge_labels] = 0

    return chunk_id[comps.labels]
