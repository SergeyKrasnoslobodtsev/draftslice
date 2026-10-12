import cv2
import numpy as np
import pytest

from draftslice.core.denoise.arrows import ArrowResult

INK = (30, 30, 30)  # цвет линий листа, RGB


class FakeOCR:
    """OCREngine без моделей: текста на листе нет."""

    def recognize(self, image):
        return []


class FakeArrows:
    """ArrowModel без модели: стрелок на листе нет."""

    def __call__(self, img, conf=0.25, overlap=0.2):
        h, w = img.shape[:2]
        return ArrowResult(np.zeros((h, w), np.uint8), np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


@pytest.fixture
def fakes():
    """(arrow_model, ocr_engine) без весов - для пайплайна и CLI."""
    return FakeArrows(), FakeOCR()


@pytest.fixture
def sheet_img():
    """(лист RGB, маска контура): вид - толстый прямоугольник с отверстием, тонкие осевые и размерная линия."""
    contour = np.zeros((400, 600), np.uint8)
    cv2.rectangle(contour, (150, 100), (450, 300), 255, 4)
    cv2.circle(contour, (300, 200), 50, 255, 4)
    thin = np.zeros_like(contour)
    cv2.line(thin, (120, 200), (480, 200), 255, 1)  # осевые
    cv2.line(thin, (300, 70), (300, 330), 255, 1)
    cv2.line(thin, (150, 300), (150, 360), 255, 1)  # выносные и размерная
    cv2.line(thin, (450, 300), (450, 360), 255, 1)
    cv2.line(thin, (150, 350), (450, 350), 255, 1)
    img = np.full((*contour.shape, 3), 255, np.uint8)
    img[(contour > 0) | (thin > 0)] = INK
    return img, contour
