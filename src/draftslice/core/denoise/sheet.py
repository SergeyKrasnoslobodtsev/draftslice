"""Лист для чистки: общее для правил считается один раз, правила - методы, чистая краска - свойство.

    sheet = Sheet(ink, g, dist, cls.thick, t_c, polys)
    sheet.trim(res.mask)
    sheet.remove(sheet.tables())
    sheet.remove(sheet.marks())
    sheet.clean

Пиксели путей и компоненты краски считаются один раз. `clean` и `segments` пересобираются, только когда `remove`
снял пути. Правило отвечает, какие пути удалить, `(P,)` bool; что удалять, решает вызывающий.
"""

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
from skan import Skeleton


@dataclass(frozen=True)
class Segments:
    """Сегменты чистой краски (компоненты связности `clean`) и их признаки; индекс 0 - фон.

    Attributes
    ----------
    labels : np.ndarray[int32]
        (H, W) номер сегмента пикселя, 0 - фон.
    stats : np.ndarray[int32]
        (N, 5) `x y w h area` сегментов (`cv2.connectedComponentsWithStats`).
    holes : np.ndarray[int64]
        (N,) замкнутых контуров (дыр площадью не меньше `t_c^2`).
    free : np.ndarray[int64]
        (N,) дыр без текста: в дыру не попадает центр полигона OCR.
    bends : np.ndarray[int64]
        (N,) изгибов: прямых кусков путей сегмента после упрощения с точностью `t_c`, минус один.
    small : np.ndarray[bool]
        (N,) наибольшая сторона меньше высоты текста.
    circle : np.ndarray[bool]
        (N,) скелет сегмента лежит на окружности с точностью `t_c`.
    n_thin : np.ndarray[int64]
        (N,) тонких путей, хоть один пиксель которых лежит в сегменте (осевые, размерные, штриховка).
    view : np.ndarray[bool]
        (N,) вид: есть дыра без текста, не мелкий, не круг и касается хотя бы одной тонкой линии.
    owner : np.ndarray[int64]
        (N,) вид, которому принадлежит сегмент: больше половины его пикселей - в выпуклой оболочке этого вида
        (отверстия, внутренние линии, вид внутри другого вида); 0 - ни в одной оболочке.
    inside : np.ndarray[bool]
        (N,) сегмент в оболочке вида: `owner > 0`.
    """

    labels: np.ndarray
    stats: np.ndarray
    holes: np.ndarray
    free: np.ndarray
    bends: np.ndarray
    small: np.ndarray
    circle: np.ndarray
    n_thin: np.ndarray
    view: np.ndarray
    owner: np.ndarray
    inside: np.ndarray

    def __len__(self) -> int:
        """Число сегментов без фона."""
        return len(self.stats) - 1


class Sheet:
    """Краска листа, её граф и пути, которые пока остаются.

    Attributes
    ----------
    ink : np.ndarray[uint8]
        Маска краски `0/255`, по которой построен граф.
    g : Skeleton
        Граф линий `thick_lines.graph.build_graph` по `ink`.
    dist : np.ndarray[float32]
        Расстояние до фона по `ink`, px.
    thick : np.ndarray[bool]
        (P,) толстые пути; тонкие - остальные.
    t_c : float
        Толщина контура в масштабе `ink`, px.
    polys : list[np.ndarray[float32]]
        Полигоны OCR `(n, 2)` (x, y) в координатах `ink`.
    keep : np.ndarray[bool]
        (P,) пути, которые остаются; в начале - толстые, `remove` их снимает.
    span : np.ndarray[int64]
        (P, 2) остающийся участок `[a, b)` каждого пути - индексы его пикселей; `trim` обрезает концы.
    max_bends, small_h, circle_tol, min_cells
        Пороги правил, см. `__init__`.
    """

    def __init__(
        self,
        ink: np.ndarray,
        g: Skeleton,
        dist: np.ndarray,
        thick: np.ndarray,
        t_c: float,
        polys: list[np.ndarray] | None = None,
        *,
        max_bends: int = 2,
        small_h: float = 1.0,
        circle_tol: float = 1.0,
        min_cells: int = 2,
    ) -> None:
        """Лист для чистки; в начале остаются толстые пути.

        Parameters
        ----------
        ink : np.ndarray[uint8]
            Маска краски `0/255` (без текста), по которой построен граф.
        g : Skeleton
            Граф линий по `ink`.
        dist : np.ndarray[float32]
            Расстояние до фона по `ink` (`cv2.distanceTransform`, `DIST_L2`), того же размера.
        thick : np.ndarray[bool]
            (P,) толстые пути (`LineClasses.thick`).
        t_c : float
            Толщина контура в масштабе `ink`, px.
        polys : list[np.ndarray[float32]] | None
            Полигоны OCR в координатах `ink`; нужны правилу таблиц.
        max_bends : int
            Открытая фигура вне вида с изгибами не больше max_bends - обозначение (`marks`): 0 - штрих, 1 - полка
            выноски, 2 - знак шероховатости; скобка бокового вида - 3.
        small_h : float
            Сегмент мелкий, если его наибольшая сторона меньше small_h высот текста (`text_h`).
        circle_tol : float
            Сегмент - круг, если точки скелета отходят от окружности не дальше circle_tol толщин контура.
        min_cells : int
            Наименьшее число ячеек таблицы (`tables`): одна ячейка с текстом - это и вид с размером внутри.

        Raises
        ------
        ValueError
            `dist` другого размера, чем `ink`, или длина `thick` не равна числу путей графа.
        """
        if dist.shape != ink.shape:
            raise ValueError(f"dist {dist.shape} и ink {ink.shape} разного размера")
        if len(thick) != g.n_paths:
            raise ValueError(f"thick: {len(thick)} значений на {g.n_paths} путей графа")
        self.ink, self.g, self.dist, self.t_c = ink, g, dist, t_c
        self.thick = np.asarray(thick, bool)
        self.polys = list(polys or [])
        self.max_bends, self.small_h, self.circle_tol, self.min_cells = max_bends, small_h, circle_tol, min_cells
        self.keep = self.thick.copy()
        self.span = np.array([[0, len(c)] for c in self.coords], np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        """Число путей графа."""
        return self.g.n_paths

    def remove(self, paths: np.ndarray) -> "Sheet":
        """Снимает пути с листа; `clean` пересоберётся при следующем обращении.

        Parameters
        ----------
        paths : np.ndarray[bool]
            (P,) пути, которые удалить.

        Returns
        -------
        Sheet
            Этот же лист, для цепочки вызовов.
        """
        self.keep &= ~np.asarray(paths, bool)
        self._reset()
        return self

    def trim(self, mask: np.ndarray) -> "Sheet":
        """Снимает стрелки по маске модели: наконечник с линией - целиком, наконечник на конце линии - обрезкой.

        Наконечник стоит на конце линии, и классификатор не всегда отрезает его от линии (клин толще линии меньше
        чем в `thick_ratio` раз). Маска модели чуть короче стрелки: остриё и основание высовываются из неё.

        - Путь снимается целиком, если вне маски у него только короткие участки (короче трёх толщин контура):
          это наконечник, высунувшийся из маски.
        - Иначе с каждого свободного конца (остриё: к нему ничего не подходит) снимается участок в маске, а
          высунувшийся кончик короче трёх толщин снимается вместе с ним. Так у оси с наконечником остаётся линия.
        - Конец в узле не обрезается: остриё чужой стрелки, упёртое в контур, контур не режет.

        Parameters
        ----------
        mask : np.ndarray[uint8]
            Маска стрелок `ArrowResult.mask` в любом масштабе листа (обычно исходном): пиксель пути `y`
            соответствует строке `floor(y * mask_h / ink_h)` маски.

        Returns
        -------
        Sheet
            Этот же лист, для цепочки вызовов.
        """
        sy, sx = mask.shape[0] / self.ink.shape[0], mask.shape[1] / self.ink.shape[1]
        short = 3 * self.t_c
        for i in np.flatnonzero(self.keep):
            a, b = self.span[i]
            yx = self.coords[i][a:b]
            inm = mask[(yx[:, 0] * sy).astype(np.int64), (yx[:, 1] * sx).astype(np.int64)] > 0
            if not inm.any():
                continue
            closed = _close_runs(inm, short)
            if closed.all():
                self.keep[i] = False
                continue
            pix = self.g.path(i)
            if a == 0 and self.g.degrees[pix[0]] == 1:  # свободное начало
                a += int(np.argmax(~closed))
            if b == len(pix) and self.g.degrees[pix[-1]] == 1:  # свободный конец
                b -= int(np.argmax(~closed[::-1]))
            self.span[i] = a, b
        self._reset()
        return self

    def _reset(self):
        """Сбросить то, что зависит от оставленных путей."""
        self.__dict__.pop("clean", None)
        self.__dict__.pop("segments", None)

    @cached_property
    def coords(self) -> list[np.ndarray]:
        """Пиксели путей `(n, 2)` (y, x), по одному массиву на путь."""
        return [np.rint(self.g.path_coordinates(i)).astype(np.int64) for i in range(self.g.n_paths)]

    @cached_property
    def clean(self) -> np.ndarray:
        """Краска путей `keep`, нарисованная заново: ломаные путей своей толщины, внутри `ink`.

        Удаляемая графика делит пиксели с контуром там, где пересекает его или касается, - вычитание порвало бы
        контур (OQ-03). Поэтому контур рисуется, а не вычитается: каждый оставленный участок пути - ломаная толщиной
        `ceil(2 * median dist) + 1` (своя толщина линии без раздувания `dist` у узлов), удалённые линии и наконечники
        не рисуются вовсе; `& ink` - нарисованное не выходит за краску листа.
        """
        paint = np.zeros(self.ink.shape, np.uint8)
        for i in np.flatnonzero(self.keep):
            yx = self.coords[i][slice(*self.span[i])]
            t = int(np.ceil(2 * np.median(self.dist[yx[:, 0], yx[:, 1]]))) + 1
            if len(yx) == 1:
                cv2.circle(paint, (int(yx[0, 1]), int(yx[0, 0])), max(t // 2, 1), 255, -1)
            else:
                cv2.polylines(paint, [yx[:, ::-1].astype(np.int32).reshape(-1, 1, 2)], False, 255, t)
        return ((paint > 0) & (self.ink > 0)).astype(np.uint8) * 255

    def _segment_of_paths(self, lab):
        """(P,) сегмент каждого пути - по большинству пикселей его остающегося участка, 0 - вне сегментов.

        Не по первому пикселю: он лежит в узле и может попасть в крошку у узла - отдельный мелкий сегмент, и путь
        с длинной дугой ушёл бы вместе с крошкой.
        """
        out = np.zeros(len(self), np.int64)
        for i, (c, (a, b)) in enumerate(zip(self.coords, self.span, strict=True)):
            k = lab[c[a:b, 0], c[a:b, 1]]
            k = k[k > 0]
            if len(k):
                out[i] = np.bincount(k).argmax()
        return out

    def tables(self) -> np.ndarray:
        """Пути таблиц (DEL-07): компонента краски с 2+ ячейками, большинство ячеек - прямоугольники с текстом.

        Таблица стоит отдельно - линии чертежа её не касаются, поэтому группа - компонента связности `ink` с
        внутренними (тонкими) линиями. Ячейки - дыры в краске группы. Ячейка прямоугольная, если её контур после
        `approxPolyDP` с точностью `t_c` - выпуклый четырёхугольник; с текстом - если в неё попадает центр
        полигона OCR. Дыра площадью меньше `t_c^2` - щель на стыке линий, а не ячейка.

        Ячеек нужно хотя бы две: одна дыра с текстом - это и прямоугольный вид с размером внутри контура (его
        удалять нельзя, OQ-04), и рамка вокруг одного текста или кружок позиции (DEL-04, `marks`).

        Returns
        -------
        np.ndarray[bool]
            (P,) путь принадлежит таблице.
        """
        lab = self._ink_labels
        cells: dict[int, list[int]] = {}  # группа -> [ячеек, прямоугольных с текстом]
        for c, (y, x), text in self._holes((self.ink > 0).astype(np.uint8)):
            n = cells.setdefault(int(lab[y, x]), [0, 0])
            n[0] += 1
            n[1] += text and _is_rect(c, self.t_c)
        tables = [k for k, (n_cells, good) in cells.items() if n_cells >= self.min_cells and good > n_cells / 2]
        return np.isin(lab[self._start[:, 0], self._start[:, 1]], tables)

    @cached_property
    def text_h(self) -> float:
        """Высота текста листа, px: медиана меньших сторон повёрнутых рамок полигонов OCR; 0 - текста нет."""
        if not self.polys:
            return 0.0
        return float(np.median([min(cv2.minAreaRect(np.asarray(p, np.float32))[1]) for p in self.polys]))

    @cached_property
    def segments(self) -> Segments:
        """Сегменты текущего `clean` и их признаки; пересобираются после `remove`."""
        paint = (self.clean > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
        holes, free = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for _, (y, x), text in self._holes(paint):
            holes[lab[y, x]] += 1
            free[lab[y, x]] += not text
        seg = self._segment_of_paths(lab)
        kept = np.flatnonzero(self.keep)
        piece = np.array([_pieces(self.coords[i][slice(*self.span[i])], self.t_c) for i in kept], np.float64)
        pieces = np.bincount(seg[kept], weights=piece, minlength=n).astype(np.int64)
        bends = np.maximum(pieces - 1, 0)
        small = stats[:, 2:4].max(1) < self.small_h * self.text_h
        circle = np.zeros(n, bool)
        for k in np.flatnonzero(holes > 0):
            yx = np.concatenate([self.coords[i][slice(*self.span[i])] for i in kept[seg[kept] == k]]).astype(np.float64)
            d = np.hypot(*(yx - yx.mean(0)).T)
            circle[k] = np.abs(d - d.mean()).max() <= self.circle_tol * self.t_c
        n_thin = np.zeros(n, np.int64)
        for i in np.flatnonzero(~self.thick):
            k = np.unique(lab[self.coords[i][:, 0], self.coords[i][:, 1]])
            n_thin[k[k > 0]] += 1
        view = (free > 0) & ~small & ~circle & (n_thin > 0)
        view[0] = False
        owner = self._owner(lab, stats, view)
        return Segments(lab, stats, holes, free, bends, small, circle, n_thin, view, owner, owner > 0)

    @staticmethod
    def _owner(lab, stats, view):
        """Вид-хозяин сегмента: оболочки видов рисуются от меньшего к большему (больший вид забирает вложенный),
        хозяин - самая частая оболочка под пикселями сегмента, если она накрывает больше половины его пикселей."""
        n = len(stats)
        hull = np.zeros(lab.shape, np.int32)
        for k in sorted(np.flatnonzero(view), key=lambda k: stats[k, cv2.CC_STAT_AREA]):
            x, y, w, h = stats[k, :4]
            ys, xs = np.nonzero(lab[y : y + h, x : x + w] == k)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack([xs + x, ys + y]).astype(np.int32)), int(k))
        pix = (lab > 0) & (hull > 0)
        key, cnt = np.unique(lab[pix].astype(np.int64) * n + hull[pix], return_counts=True)
        seg, by = key // n, key % n
        owner = np.zeros(n, np.int64)
        best = np.zeros(n, np.int64)
        for k, h, c in zip(seg, by, cnt, strict=True):
            if c > best[k]:
                best[k], owner[k] = c, h
        owner[best <= stats[:, cv2.CC_STAT_AREA] / 2] = 0
        owner[0] = 0
        return owner

    def marks(self) -> np.ndarray:
        """Пути обозначений (DEL-04, DEL-05) по сегментам `clean`: вне видов - всё, кроме сложных открытых фигур.

        Вид - сегмент с замкнутым контуром без текста, не мелкий (наибольшая сторона не меньше высоты текста), не
        круг и касается хотя бы одной тонкой линии: у вида всегда есть осевые, размеры или штриховка, у отдельной
        буквы, кружка позиции или рамки - нет. Внутри вида - больше половины пикселей сегмента в оболочке вида.

        - Внутри вида удаляется только мелкое без замкнутого контура: точки, чёрточки. Отверстия остаются.
        - Вне вида остаются только виды и открытые (без замкнутого контура) фигуры сложнее 2 изгибов, которых
          касается тонкая линия, - скобка бокового вида после снятия линии обрыва, профиль резьбы выносного
          элемента. Удаляется мелкое; открытое до 2 изгибов (штрихи, полки выносок, знаки шероховатости,
          размерные линии); открытое без тонких линий (буквы обозначения видов крупнее текста размеров, которых
          не нашёл OCR); замкнутое, но не вид (буквы с петлями, кружки позиций, рамки вокруг текста).

        Считается по текущему `clean`, поэтому вызывается после правил, которые снимают стрелки и таблицы.

        Returns
        -------
        np.ndarray[bool]
            (P,) путь остаётся на листе и лежит в сегменте-обозначении.
        """
        sg = self.segments
        complex_open = (sg.holes == 0) & (sg.bends > self.max_bends) & (sg.n_thin > 0)
        outside = sg.small | ~(sg.view | complex_open)
        mark = np.where(sg.inside, sg.small & (sg.holes == 0), outside)
        mark[0] = False  # фон
        return self.keep & mark[self._segment_of_paths(sg.labels)]

    def _holes(self, mask):
        """Дыры маски площадью не меньше t_c^2 (меньше - щель на стыке линий): (контур, пиксель краски вокруг (y, x),
        попадает ли в дыру центр полигона OCR)."""
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        out = []
        for c, (_, _, _, parent) in zip(cnts, hier[0] if hier is not None else [], strict=False):
            if parent < 0 or cv2.contourArea(c) < self.t_c**2:  # внешний контур или щель
                continue
            x, y = cnts[parent][0, 0]
            out.append((c, (y, x), any(cv2.pointPolygonTest(c, p, False) >= 0 for p in self._centers)))
        return out

    @cached_property
    def _centers(self):
        """Центры полигонов OCR (x, y)."""
        return [(float(x), float(y)) for x, y in (np.asarray(p, np.float32).mean(0) for p in self.polys)]

    @cached_property
    def _start(self):
        """(P, 2) первый пиксель каждого пути (y, x)."""
        return np.array([yx[0] for yx in self.coords]).reshape(-1, 2)

    @cached_property
    def _ink_labels(self):
        """Компоненты связности краски `ink` (8-связность)."""
        return cv2.connectedComponents((self.ink > 0).astype(np.uint8), connectivity=8)[1]


def _close_runs(on, short):
    """Участки False короче short между True и у концов, если рядом есть True, - тоже True."""
    out = on.copy()
    idx = np.flatnonzero(np.diff(np.concatenate([[1], on.astype(np.int8), [1]])))  # границы участков False
    for lo, hi in zip(idx[::2], idx[1::2], strict=True):
        if hi - lo < short:
            out[lo:hi] = True
    return out


def _pieces(yx, eps):
    """Прямых кусков участка пути после упрощения с точностью eps; короче трёх точек - один кусок."""
    return max(len(cv2.approxPolyDP(yx[:, ::-1].astype(np.int32).reshape(-1, 1, 2), eps, False)) - 1, 1)


def _is_rect(c, eps):
    """Контур после упрощения с точностью eps - выпуклый четырёхугольник."""
    a = cv2.approxPolyDP(c, eps, True)
    return len(a) == 4 and cv2.isContourConvex(a)
