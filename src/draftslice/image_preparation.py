"""Загрузка чертежа, приведение масштаба и бинаризация штрихов.

Масштаб задается один раз и дальше не меняется: все пиксельные величины пайплайна, от длины ядра
замыкания до порогов длины ребер, измеряются в пикселях этого кадра. Увеличение делается бикубикой,
потому что она восстанавливает по серой кайме вокруг штриха плавный градиент, порог режет край
субпиксельно, и толщины перестают слипаться при квантовании distance transform. Уменьшение идет
через INTER_AREA, которая усредняет, а не выбирает пиксель.

Бинаризация опирается на то, что штрих темный на светлом, отсюда инверсия. Билатеральный фильтр
гасит зерно jpeg, сохраняя границы: без него тонкие линии рвутся, а каждый разрыв плодит в графе
пару ложных свободных концов. Порог ставит Otsu по гистограмме всего кадра, отдельного параметра
для него нет.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from draftslice.common_types import Mask, Mat


@dataclass(frozen=True, eq=False)
class BinarizedStrokes:
    """Результат бинаризации кадра.

    Attributes
    ----------
    gray_image : Mat
        Кадр в градациях серого до фильтрации.
    strokes_mask : Mask
        Маска штрихов, True это чернила.
    threshold_value : float
        Порог, выбранный методом Otsu.
    """

    gray_image: Mat
    strokes_mask: Mask
    threshold_value: float


def load_bgr_image(image_path: Path) -> Mat:
    """Прочитать картинку по пути, который может содержать кириллицу.

    Parameters
    ----------
    image_path : Path
        Путь к файлу.

    Returns
    -------
    Mat
        Кадр в порядке каналов BGR.

    Raises
    ------
    ValueError
        Если файл не читается как изображение.

    Notes
    -----
    cv2.imread не открывает пути с кириллицей на части систем, поэтому файл читается через numpy и
    декодируется из памяти.
    """
    raw_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"не читается как изображение: {image_path}")
    return image


def resize_to_long_side(image: Mat, long_side: int) -> Mat:
    """Привести длинную сторону кадра к заданному размеру.

    Parameters
    ----------
    image : Mat
        Исходный кадр.
    long_side : int
        Требуемый размер длинной стороны в пикселях.

    Returns
    -------
    Mat
        Кадр с сохраненными пропорциями.
    """
    current_side = max(image.shape[:2])
    if current_side == long_side:
        return image

    factor = long_side / float(current_side)
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    width = round(image.shape[1] * factor)
    height = round(image.shape[0] * factor)
    return cv2.resize(image, (width, height), interpolation=interpolation)


def scale_image(image: Mat, factor: float) -> Mat:
    """Масштабировать кадр с сохранением пропорций.

    Parameters
    ----------
    image : Mat
        Исходный кадр.
    factor : float
        Множитель размера.

    Returns
    -------
    Mat
        Масштабированный кадр.
    """
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_LINEAR)


def scale_mask(mask: Mask, factor: float) -> Mask:
    """Масштабировать бинарную маску тем же способом, что и кадр.

    Parameters
    ----------
    mask : Mask
        Исходная маска.
    factor : float
        Множитель размера.

    Returns
    -------
    Mask
        Масштабированная маска.

    Notes
    -----
    Интерполяция билинейная, а порог берет любое ненулевое значение, поэтому при увеличении маска
    подрастает на пиксель по своей границе. Для текстовой области это безопасно: лишний захват
    дешевле пропуска.
    """
    return scale_image(mask.astype(np.uint8) * 255, factor) > 0


def binarize_strokes(
    image: Mat, bilateral_diameter: int = 5, sigma_color: float = 15.0, sigma_space: float = 15.0
) -> BinarizedStrokes:
    """Выделить штрихи чертежа порогом Otsu по сглаженному кадру.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.
    bilateral_diameter : int
        Диаметр окрестности билатерального фильтра в пикселях.
    sigma_color : float
        Разброс по яркости, в пределах которого пиксели считаются похожими.
    sigma_space : float
        Разброс по расстоянию для билатерального фильтра.

    Returns
    -------
    BinarizedStrokes
        Серый кадр, маска штрихов и выбранный порог.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray_image, d=bilateral_diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    threshold_value, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return BinarizedStrokes(gray_image=gray_image, strokes_mask=binary > 0, threshold_value=float(threshold_value))
