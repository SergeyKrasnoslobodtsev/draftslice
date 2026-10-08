"""Бинаризация чертежа и смыкание микроразрывов контура.

Основной шаг препроцессинга чертежа перед сегментацией регионов.
На данном этапе необходимо перевести изображение в одноканальное представление (градации серого)
и бинаризовать его. Дилатацию делаем для того чтобы сомкнуть микроразрывы контуров.
Даже визуально идеальные изображения имеют на круговых контурах микроразрывы.
Также дилатация помогает подтянуть текст к линиям выносок, что улучшает последующую сегментацию регионов.

Notes
-----
Отсечка минимального размера ядра дилатации: `scale = max(width // 300, 1)` подобрана
имперически. Возможно именно этот параметр будет неустойчивым для чертежей
с очень высоким или очень низким разрешением.
"""

from draftslice.core.exceptions import DraftsliceValueError
from draftslice.core.preprocess import preprocess
from draftslice.core.types import Mat
from draftslice.core.utils import image_utils as imutils


def preprocess_drawing(image_rgb: Mat, dilation_kernel_scale_divisor: int = 300) -> Mat:
    """Препроцессинг чертежа для последующей сегментации регионов.

    Notes
    -----
    Преобразует RGB изображение в градации серого, бинаризует его с помощью Otsu и инверсии,
    а затем дилатирует бинарную маску эллиптическим ядром, размер которого пропорционален
    ширине изображения.

    Parameters
    ----------
    image_rgb : Mat
        Исходное растровое изображение чертежа в формате RGB.
    dilation_kernel_scale_divisor : int
        Делитель для вычисления масштаба ядра дилатации.

    Returns
    -------
    Mat
        Бинарная маска после дилатации.
    Raises
    ------
    DraftsliceValueError
        Если входное изображение равно None.
    DraftsliceValueError
        Если делитель для масштаба ядра дилатации не является положительным числом.
    """
    if image_rgb is None:
        raise DraftsliceValueError("Входное изображение не должно быть None.")

    if dilation_kernel_scale_divisor <= 0:
        raise DraftsliceValueError("Делитель для масштаба ядра дилатации должен быть положительным числом.")

    binary_mask = preprocess(image_rgb, target=None).binary

    image_width: int = binary_mask.shape[1]
    dilation_kernel_scale: int = max(image_width // dilation_kernel_scale_divisor, 1)
    dilated_mask = imutils.dilate_image(binary_mask, imutils.MorphShape.ELLIPSE, dilation_kernel_scale)

    return dilated_mask
