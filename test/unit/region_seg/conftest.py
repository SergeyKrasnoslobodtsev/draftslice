import cv2
import numpy as np
import pytest


@pytest.fixture
def sheet() -> np.ndarray:
    """Белый RGB-лист 600x800 с «текстом» внизу - 40 мелких прямоугольников.

    Пороги сегментации считаются от среднего и медианы по компонентам листа, поэтому без
    мелочи единственный вид сам задал бы порог и не прошёл бы его.
    """
    img = np.full((600, 800, 3), 255, np.uint8)
    for i in range(40):
        x, y = 60 + (i % 10) * 65, 520 + (i // 10) * 18
        cv2.rectangle(img, (x, y), (x + 8, y + 10), (0, 0, 0), 1)
    return img
