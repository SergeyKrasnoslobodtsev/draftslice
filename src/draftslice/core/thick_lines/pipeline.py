"""Точка входа `thick_lines`: граф линий по маске краски и деление линий на толстые и тонкие.

lines = run(ink, t_c)
lines.g, lines.dist, lines.cls.thick
lines.plot()  # RGB: толстые красным, тонкие голубым
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skan import Skeleton

from draftslice.core.thick_lines.classify import LineClasses, classify_lines
from draftslice.core.thick_lines.graph import build_graph


@dataclass(frozen=True)
class Lines:
    """Линии листа.

    Attributes
    ----------
    ink : np.ndarray[uint8]
        Маска краски `0/255` `(H, W)`, по которой построен граф.
    dist : np.ndarray[float32]
        Расстояние до фона по `ink` `(H, W)`, px.
    g : Skeleton
        Граф линий.
    cls : LineClasses
        Классы путей графа.
    """

    ink: np.ndarray
    dist: np.ndarray
    g: Skeleton
    cls: LineClasses

    def plot(self, width: int = 2) -> np.ndarray:
        """Линии на краске: толстые красным, тонкие голубым.

        Parameters
        ----------
        width : int
            Толщина линии пути на картинке, px.

        Returns
        -------
        np.ndarray[uint8]
            RGB `(H, W, 3)`: краска серым, пути цветом класса.
        """
        out = np.full((*self.ink.shape, 3), 255, np.uint8)
        out[self.ink > 0] = 200
        for i in range(self.g.n_paths):
            yx = np.rint(self.g.path_coordinates(i)).astype(np.int32)
            color = (255, 0, 0) if self.cls.thick[i] else (0, 190, 255)
            cv2.polylines(out, [yx[:, ::-1].reshape(-1, 1, 2)], False, color, width)
        return out


def run(
    ink: np.ndarray,
    t_c: float,
    max_kink_deg: float = 30.0,
    tol: float = 0.1,
    thick_ratio: float = 1.3,
) -> Lines:
    """Граф линий по маске краски и классы его путей.

    Parameters
    ----------
    ink : np.ndarray[uint8]
        Маска краски `0/255` `(H, W)` без текста.
    t_c : float
        Толщина контура в масштабе `ink`, px.
    max_kink_deg : float
        Наибольший излом при склейке соосных путей графа и мостов классификатора, градусы.
    tol : float
        Допуск совпадения окружностей двух дуг при склейке, доля радиуса (`build_graph`).
    thick_ratio : float
        Отношение толщин, выше которого путь режется на толстую и тонкую части (`build_graph`).

    Returns
    -------
    Lines
        Краска, `dist`, граф и классы путей.
    """
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    g = build_graph(ink, dist, max_kink_deg=max_kink_deg, tol=tol, thick_ratio=thick_ratio)
    return Lines(ink, dist, g, classify_lines(g, dist, t_c, max_kink_deg=max_kink_deg))
