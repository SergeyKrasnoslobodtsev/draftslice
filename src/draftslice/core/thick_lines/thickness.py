from dataclasses import dataclass

import numpy as np
from skan import Skeleton


@dataclass(frozen=True, eq=False)
class Profile:
    """Профиль толщины вдоль пути без пятен стыков.

    Attributes
    ----------
    t : np.ndarray[float64]
        (N,) толщина, px; в пятне стыка — по соседям вне пятен.
    spots : np.ndarray[bool]
        (N,) точка в пятне стыка.
    """

    t: np.ndarray
    spots: np.ndarray


def skeleton_thickness(g: Skeleton, dist: np.ndarray) -> np.ndarray:
    """Толщина в каждом пикселе скелета.

    Parameters
    ----------
    g : Skeleton
        Граф скелета skan.
    dist : np.ndarray[float32]
        Расстояние до фона по краске (cv2.distanceTransform), px.

    Returns
    -------
    np.ndarray[float32]
        (P,) 2·dist по id пикселя skan — одна выборка на весь скелет.
    """
    rc = g.coordinates.astype(np.intp)
    return 2.0 * dist[rc[:, 0], rc[:, 1]]


def junction_free(t: np.ndarray, deg: np.ndarray) -> Profile:
    """Профиль толщины вдоль пути без пятен стыков.

    Пятно стыка — точки пути в радиусе ceil(t/2) от узла (степень ≥ 3). Толщина в пятне — линейная
    интерполяция по соседям вне пятен; если весь путь в пятнах — толщина как есть.

    Parameters
    ----------
    t : np.ndarray[float32]
        (N,) толщина в точках пути по порядку обхода, px.
    deg : np.ndarray[int64]
        (N,) степень тех же точек в графе skan (`Skeleton.degrees`).

    Returns
    -------
    Profile
        Толщина без пятен стыков и маска пятен.
    """
    bad = np.zeros(len(t), bool)
    for j in np.flatnonzero(deg >= 3):
        r = int(np.ceil(t[j] / 2))
        bad[max(0, j - r) : j + r + 1] = True
    if bad.all():
        return Profile(t.astype(np.float64), bad)
    i = np.arange(len(t))
    return Profile(np.interp(i, i[~bad], t[~bad]), bad)
