"""Единая точка входа сегментации чертежа на виды.

`run()` размечает большие сегменты через `seg.segment` и возвращает по кропу на каждый.
Кроп не копирует пикселы: хранит срез исходного изображения (вид numpy на тот же массив) и
маску своего сегмента в боксе. Очищенное изображение собирается свойством `Crop.img` только
при обращении, поэтому на весь лист в памяти остаётся одно исходное изображение.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from draftslice.core.region_seg.seg import segment
from draftslice.core.types import Array2D, Mat


@dataclass(frozen=True, eq=False)
class Crop:
    """Один большой сегмент (вид детали) в боксе по его собственной границе.

    Attributes
    ----------
    label : int
        Номер сегмента.
    x : int
        Левая граница бокса в пикселях исходного изображения.
    y : int
        Верхняя граница бокса в пикселях исходного изображения.
    w : int
        Ширина бокса.
    h : int
        Высота бокса.
    mask : Array2D[np.bool_]
        Пикселы сегмента внутри бокса, форма `(h, w)`.
    src : Mat
        Срез исходного изображения по боксу - вид на исходный массив, не копия.
    """

    label: int
    x: int
    y: int
    w: int
    h: int
    mask: Array2D[np.bool_]
    src: Mat

    @property
    def img(self) -> Mat:
        """Кроп исходного изображения: пикселы сегмента как в исходнике, остальное белое."""
        out = np.full_like(self.src, 255)
        out[self.mask] = self.src[self.mask]
        return out


def run(
    img: Mat,
    sheet_cov: float = 0.85,
    thick_factor: float = 5.0,
    area_factor: float = 2.0,
    min_extent: float = 0.1,
    inside: float = 0.5,
) -> list[Crop]:
    """Сегментирует чертёж на виды и возвращает кроп каждого вида.

    Parameters
    ----------
    img : Mat
        Изображение чертежа, RGB.
    sheet_cov, thick_factor, area_factor, min_extent, inside : float
        Параметры сегментации, передаются в `seg.segment` - смысл описан там.

    Returns
    -------
    list[Crop]
        Кропы больших сегментов в порядке номеров. Пустой список, если на листе нет ни
        одного вида.

    Raises
    ------
    DraftsliceTypeError
        Если `img` не трёхканальное.
    """
    seg = segment(img, sheet_cov, thick_factor, area_factor, min_extent, inside)

    crops = []
    # find_objects даёт бокс каждой метки за один проход; None - номер, ушедший при слиянии.
    for label, box in enumerate(ndimage.find_objects(seg), start=1):
        if box is None:
            continue
        rows, cols = box
        crops.append(
            Crop(
                label=label,
                x=cols.start,
                y=rows.start,
                w=cols.stop - cols.start,
                h=rows.stop - rows.start,
                mask=seg[box] == label,
                src=img[box],
            )
        )
    return crops
