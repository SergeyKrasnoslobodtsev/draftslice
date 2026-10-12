"""Виды листа для ViT-моделей: группы сегментов чистой краски, вырезанные из исходного листа.

    views = Views(sheet, orig, scale=preprocessed.scale)
    views[0].box, views[0].mask, views[0].img
    views.crops(size=224)
    views.rotations(size=224)

Вид - сегмент-вид `Sheet.segments` вместе со всеми сегментами, которые он накрывает оболочкой (отверстия,
внутренние линии); сегмент вне оболочек видов (открытая скобка бокового вида после снятия линии обрыва) - отдельный
вид. Вид вырезается в исходном масштабе: пиксели исходного листа под чистой маской, фон белый (OQ-02).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from draftslice.core.denoise.sheet import Sheet
from draftslice.core.utils.ops import scale_mask

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class View:
    """Один вид листа в исходном масштабе.

    Attributes
    ----------
    box : tuple[int, int, int, int]
        `x0 y0 x1 y1` рамки вида с отступом в координатах исходного листа; `x1`, `y1` не входят.
    mask : np.ndarray[uint8]
        Чистая маска вида `0/255` размера рамки: только сегменты этого вида.
    img : np.ndarray
        RGB размера рамки: пиксели исходного листа под маской, фон белый.
    """

    box: Box
    mask: np.ndarray
    img: np.ndarray


class Views:
    """Виды листа: группы сегментов `Sheet.segments`, вырезанные из исходного листа.

    Attributes
    ----------
    sheet : Sheet
        Лист после правил чистки (`clean` - краска видов).
    orig : np.ndarray
        Исходный RGB-лист.
    scale : float
        Во сколько раз рабочий масштаб `sheet` больше исходного (`Preprocessed.scale`).
    margin : float
        Отступ вокруг вида в толщинах контура (VIEW-01; по OQ-05 подбирается и утверждается).
    """

    def __init__(self, sheet: Sheet, orig: np.ndarray, scale: float, margin: float = 3.0) -> None:
        """Виды листа.

        Parameters
        ----------
        sheet : Sheet
            Лист после правил чистки.
        orig : np.ndarray
            Исходный RGB-лист `(H, W, 3)`, из которого режутся виды.
        scale : float
            Во сколько раз рабочий масштаб `sheet` больше исходного (`Preprocessed.scale`).
        margin : float
            Отступ вокруг вида в толщинах контура `sheet.t_c`.

        Raises
        ------
        ValueError
            Размер `sheet.ink` не равен размеру `orig`, умноженному на `scale` (с точностью до пикселя).
        """
        h, w = orig.shape[:2]
        if abs(sheet.ink.shape[0] - h * scale) > 1 + scale or abs(sheet.ink.shape[1] - w * scale) > 1 + scale:
            raise ValueError(f"ink {sheet.ink.shape} не равен orig {orig.shape[:2]} x {scale}")
        self.sheet, self.orig, self.scale, self.margin = sheet, orig, scale, margin

    def __len__(self) -> int:
        """Число видов."""
        return len(self.items)

    def __getitem__(self, i: int) -> View:
        """Вид `i`, от крупного к мелкому."""
        return self.items[i]

    def __iter__(self) -> Iterator[View]:
        """Виды от крупного к мелкому."""
        return iter(self.items)

    @cached_property
    def items(self) -> list[View]:
        """Виды от крупного к мелкому по площади рамки, при равной - сверху вниз и слева направо."""
        sg = self.sheet.segments
        group = self._groups(sg)
        lab = group[sg.labels]  # (H, W) номер группы в рабочем масштабе
        pad = self.margin * self.sheet.t_c
        out = []
        for gid in np.unique(group[1:]):
            st = sg.stats[group == gid]
            x0, y0 = st[:, 0].min() - pad, st[:, 1].min() - pad
            x1, y1 = (st[:, 0] + st[:, 2]).max() + pad, (st[:, 1] + st[:, 3]).max() + pad
            out.append(self._view(lab == gid, self._to_orig((x0, y0, x1, y1))))
        return sorted(out, key=lambda v: (-(v.box[2] - v.box[0]) * (v.box[3] - v.box[1]), v.box[1], v.box[0]))

    def crops(self, size: int) -> list[np.ndarray]:
        """Виды в квадрате `size` для ViT, в общем масштабе листа (DUP-05).

        Самый крупный вид листа вписывается в квадрат, остальные уменьшаются тем же множителем: мелкий вид остаётся
        мелким относительно крупного, как на листе. Вид по центру белого квадрата.

        Parameters
        ----------
        size : int
            Сторона квадрата, px: вход модели.

        Returns
        -------
        list[np.ndarray]
            RGB `(size, size, 3)` по виду, в порядке `items`.
        """
        if not self.items:
            return []
        k = size / max(max(v.img.shape[:2]) for v in self.items)
        out = []
        for v in self.items:
            h, w = v.img.shape[:2]
            img = cv2.resize(v.img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
            sq = np.full((size, size, 3), 255, np.uint8)
            top, left = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
            sq[top : top + img.shape[0], left : left + img.shape[1]] = img
            out.append(sq)
        return out

    def rotations(self, size: int) -> list[list[np.ndarray]]:
        """Повороты квадратов `crops` на 0, 90, 180 и 270 градусов против часовой стрелки (DUP-06).

        Parameters
        ----------
        size : int
            Сторона квадрата, px.

        Returns
        -------
        list[list[np.ndarray]]
            По виду - четыре поворота, `np.rot90(crop, k)` для k = 0..3.
        """
        return [[np.rot90(c, k) for k in range(4)] for c in self.crops(size)]

    def _groups(self, sg):
        """Вид сегмента: вид-хозяин или сам сегмент; сегменты ближе двух максимальных толщин листа `2 * max(dist)` -
        один вид (контур, разорванный зазором вокруг стрелки или растром): каждый расширяется на одну толщину."""
        r = int(np.ceil(2 * self.sheet.dist.max()))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        near = cv2.connectedComponents(cv2.dilate((sg.labels > 0).astype(np.uint8), ker), connectivity=8)[1]
        _, first = np.unique(sg.labels.ravel(), return_index=True)  # пиксель каждого сегмента, 0 - фон
        parent = np.arange(len(sg.stats))

        def root(k):
            while parent[k] != k:
                k = parent[k]
            return k

        by_near = {}
        for k in range(1, len(sg.stats)):
            for j in (sg.owner[k], by_near.setdefault(near.ravel()[first[k]], k)):
                if j > 0:
                    parent[root(k)] = root(j)
        group = np.array([root(k) for k in range(len(sg.stats))])
        group[0] = 0
        return group

    def _to_orig(self, box):
        """Рамка рабочего масштаба -> исходного: наружу до целого пикселя, по краю листа."""
        h, w = self.orig.shape[:2]
        x0, y0, x1, y1 = box
        s = self.scale
        return (
            max(0, int(np.floor(x0 / s))),
            max(0, int(np.floor(y0 / s))),
            min(w, int(np.ceil(x1 / s))),
            min(h, int(np.ceil(y1 / s))),
        )

    def _view(self, member, box):
        """Вид в рамке исходного листа: пиксель (Y, X) берёт пиксель рабочей маски (floor(Y*s), floor(X*s))."""
        x0, y0, x1, y1 = box
        mask = scale_mask(member & (self.sheet.clean > 0), self.orig.shape[:2], self.scale, box).astype(np.uint8) * 255
        img = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
        img[mask > 0] = self.orig[y0:y1, x0:x1][mask > 0]
        return View((x0, y0, x1, y1), mask, img)
