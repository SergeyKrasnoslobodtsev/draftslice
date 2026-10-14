"""Точка входа `denoise`: правила чистки листа - стрелки, таблицы, обозначения.

```python
sheet = run(lines, arrows.mask, t_c, polys)
sheet.clean             # чистая краска 0/255 в рабочем масштабе
```
"""

import numpy as np

from draftslice.core.denoise.sheet import Sheet
from draftslice.core.thick_lines.pipeline import Lines
from draftslice.core.utils.checks import check_image


def run(lines: Lines, arrow_mask: np.ndarray, t_c: float, polys: list[np.ndarray]) -> Sheet:
    """Снимает с листа стрелки, таблицы и обозначения.

    Parameters
    ----------
    lines : Lines
        Краска без текста, граф и классы линий в рабочем масштабе (`thick_lines.run`).
    arrow_mask : np.ndarray[uint8]
        Маска стрелок `0/255` `(h, w)` в любом масштабе листа (`ArrowResult.mask`).
    t_c : float
        Толщина контура в рабочем масштабе, px.
    polys : list[np.ndarray]
        Полигоны OCR `(4, 2)`, точки `(x, y)` в рабочем масштабе.

    Returns
    -------
    Sheet
        Лист после правил; `Sheet.clean` - чистая краска.

    Raises
    ------
    TypeError
        `arrow_mask` не `(h, w)` uint8.
    """
    check_image(arrow_mask, channels=1, dtype=np.uint8)
    sheet = Sheet(lines.ink, lines.g, lines.dist, lines.cls.thick, t_c, polys)
    sheet.trim(arrow_mask).remove(sheet.tables())
    return sheet.remove(sheet.marks())
