import numpy as np
from skan import Skeleton

from draftslice.core.types import Array, Array2D


def skeleton_thickness(g: Skeleton, dist: Array2D[np.float32]) -> Array[np.float32]:
    """Толщина в каждом пикселе скелета.

    Parameters
    ----------
    g : Skeleton
        Граф скелета skan.
    dist : Array2D[np.float32]
        Расстояние до фона по краске (cv2.distanceTransform), px.

    Returns
    -------
    Array[np.float32]
        (P,) 2·dist по id пикселя skan — одна выборка на весь скелет.
    """
    rc = g.coordinates.astype(np.intp)
    return 2.0 * dist[rc[:, 0], rc[:, 1]]


def junction_free(t, deg):
    """Профиль без пятен стыков (в пятне — по соседям вне пятен) и маска пятен."""
    bad = np.zeros(len(t), bool)
    for j in np.flatnonzero(deg >= 3):
        r = int(np.ceil(t[j] / 2))
        bad[max(0, j - r) : j + r + 1] = True
    if bad.all():
        return t.astype(np.float64), bad
    i = np.arange(len(t))
    return np.interp(i, i[~bad], t[~bad]), bad
