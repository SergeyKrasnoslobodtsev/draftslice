"""Предобработка изображения: серое, бинаризация и нормализация масштаба по толщине контура.

`preprocess()` - единственная публичная функция файла, шаги - приватные функции:

1. `_gray` - серое минимумом по каналам.
2. `_binarize` - бинаризация Otsu, краска 255.
3. `_thickness` - толщина контура по маске краски: вершина верхней горки толщины скелета.
4. `_rescale` - масштаб, при котором толщина контура равна `target`.

Файл ничего не знает о содержимом чертежа (текст, размеры). Нормализация нужна только для анализа, результат
чистки возвращается на исходные пиксели через `Preprocessed.scale`, см. SCALE-01.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from draftslice.core.exceptions import DraftsliceValueError
from draftslice.core.types import Mat
from draftslice.core.validators import validate_image


@dataclass(frozen=True)
class Preprocessed:
    """Изображение, серое и маска краски в одном масштабе.

    Attributes
    ----------
    img : Mat
        Изображение RGB после масштабирования; то же, что на входе, если масштаб не менялся.
    gray : Mat
        Серое изображение того же размера, `(H, W)`.
    binary : Mat
        Маска краски `0/255` того же размера, `(H, W)`.
    scale : float
        Во сколько раз изображение увеличено относительно исходного. Результат анализа в этом масштабе
        возвращается на исходное изображение делением координат на `scale`.
    t_c : float | None
        Толщина контура на исходном изображении, px; `None`, если нормализация выключена.
    """

    img: Mat
    gray: Mat
    binary: Mat
    scale: float
    t_c: float | None


@validate_image(channels=3)
def preprocess(
    img: Mat,
    target: float | None = 6.0,
    tol: float = 0.15,
    min_t: float = 8.0,
    max_scale: float = 4.0,
    top_share: float = 0.06,
) -> Preprocessed:
    """Переводит изображение в серое, бинаризует и приводит масштаб так, чтобы толщина контура была `target`.

    Контур 3 px в исходнике и контур 10 px в большом скане после нормализации выглядят одинаково для всех шагов
    анализа. Толщину меряют по маске из самого изображения. Если она меньше `min_t`, замер повторяют на изображении,
    увеличенном вдвое: интерполяция серого меняет ширину границы, и по бинарной маске исходника толщина после
    увеличения не предсказывается (на размытом скане расходится почти вдвое). Толщину контура оценивают по скелету:
    толщина в точке - удвоенное расстояние до фона, контур - самая толстая из толщин (с округлением до px), на
    которую приходится не меньше `top_share` пикселей скелета. Горка выше контура (наконечники) всегда реже.

    Parameters
    ----------
    img : Mat
        Изображение RGB, `(H, W, 3)`.
    target : float | None
        Нужная толщина контура на выходе, px. `None` - не масштабировать, вернуть серое и маску исходника.
    tol : float
        Допуск масштаба: если `|scale - 1| <= tol`, изображение не трогают и `scale` равен 1.
    min_t : float
        Толщина контура, px, ниже которой замер повторяют на увеличенном вдвое изображении.
    max_scale : float
        Верхняя граница увеличения: защита от тонкого (почти невидимого) контура.
    top_share : float
        Доля пикселей скелета, с которой толщина считается горкой контура. На 17 листах контур занимает от 9 %,
        самая частая толщина выше контура - не больше 4.3 %.

    Returns
    -------
    Preprocessed
        Изображение, серое, маска краски, масштаб и толщина контура на исходнике.

    Raises
    ------
    DraftsliceTypeError
        У изображения не три канала.
    DraftsliceValueError
        `target`, `tol`, `min_t` не положительны (`tol` не отрицателен), `max_scale` меньше 1, `top_share` вне
        (0, 1) или в изображении нет краски при включённой нормализации.
    """
    if target is not None and (target <= 0 or tol < 0 or min_t <= 0 or max_scale < 1 or not 0 < top_share < 1):
        raise DraftsliceValueError(f"Некорректные параметры: {target=}, {tol=}, {min_t=}, {max_scale=}, {top_share=}.")
    gray = _gray(img)
    binary = _binarize(gray)
    if target is None:
        return Preprocessed(img, gray, binary, 1.0, None)
    t_c = _thickness(binary, top_share)
    if t_c < min_t:
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR_EXACT)
        t_c = _thickness(_binarize(up), top_share) / 2.0
    scale = min(target / t_c, max_scale)
    if abs(scale - 1.0) <= tol:
        return Preprocessed(img, gray, binary, 1.0, t_c)
    interp = cv2.INTER_LINEAR_EXACT if scale > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    gray = _gray(img)
    return Preprocessed(img, gray, _binarize(gray), scale, t_c)


def _gray(img: Mat) -> Mat:
    """Серое минимумом по каналам: светлые цветные линии (жёлтая, оранжевая осевая) остаются тёмными."""
    c0, c1, c2 = cv2.split(img)
    return cv2.min(cv2.min(c0, c1), c2)


def _binarize(gray: Mat) -> Mat:
    """Бинаризация Otsu с инверсией: краска 255, фон 0."""
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _thickness(binary: Mat, top_share: float) -> float:
    """Толщина контура: среднее по самой толстой горке скелета, на которую приходится не меньше top_share, px."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = 2.0 * dist[cv2.ximgproc.thinning(binary) > 0]
    if t.size == 0:
        raise DraftsliceValueError("В маске нет краски: нечего мерить.")
    bins, cnt = np.unique(np.round(t), return_counts=True)
    ok = bins[cnt >= top_share * t.size]
    top = ok.max() if ok.size else bins[np.argmax(cnt)]
    return float(t[np.round(t) == top].mean())
