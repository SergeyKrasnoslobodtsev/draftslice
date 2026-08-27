"""Детекция текстовых областей моделью DBNet и сборка маски текста.

Текст на чертеже ищется не для того, чтобы его стереть, а чтобы буквы не участвовали в оценке
границ классов толщины и не тянули статистику вниз. Поэтому маска может быть грубой: цена лишнего
захвата линии мала, цена пропущенной надписи велика.

Сеть выдает карту вероятностей того, что пиксель принадлежит тексту. Карта снимается на нескольких
масштабах и усредняется: мелкие надписи ловятся на крупном масштабе, крупные заголовки на мелком.
Размер входа должен быть кратен тридцати двум, и это добирается паддингом белым, а не ресайзом,
чтобы масштаб остался единичным и карта легла на штрихи попиксельно.

Дальше повторяется логика DBPostProcess: по бинаризованной карте берутся контуры, для каждого
считается средняя вероятность внутри его минимального прямоугольника, и решение принимается по этой
средней, а не по площади. Принятые ядра раздуваются на расстояние, пропорциональное отношению
площади к периметру: раздувается сам контур, а не описанный прямоугольник, поэтому маска идет вдоль
наклонных надписей и захватывает меньше лишнего.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from draftslice.common_types import Floats, Ints, Mask, Mat
from draftslice.image_preparation import resize_to_long_side

DETECTOR_STRIDE = 32
"""Кратность размера входа, которую требует DBNet."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Среднее нормализации из PP-OCRv5_server_det.yml."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Стандартное отклонение нормализации из PP-OCRv5_server_det.yml."""


@dataclass(frozen=True, eq=False)
class PaddedImage:
    """Кадр, добитый белым до кратности stride.

    Attributes
    ----------
    image : Mat
        Кадр после паддинга.
    added_rows : int
        Сколько строк дописано снизу.
    added_cols : int
        Сколько столбцов дописано справа.
    """

    image: Mat
    added_rows: int
    added_cols: int


@dataclass(frozen=True, eq=False)
class TextCandidates:
    """Кандидаты в текстовые ядра и их метрики.

    Attributes
    ----------
    contours : list[Ints]
        Контуры кандидатов.
    score : Floats
        Средняя вероятность внутри минимального прямоугольника контура.
    short_side : Floats
        Короткая сторона минимального прямоугольника.
    area : Floats
        Площадь контура.
    perimeter : Floats
        Периметр контура.
    """

    contours: list[Ints]
    score: Floats
    short_side: Floats
    area: Floats
    perimeter: Floats

    def __len__(self) -> int:
        return len(self.contours)


@dataclass(frozen=True, eq=False)
class TextKernels:
    """Принятые текстовые ядра.

    Attributes
    ----------
    contours : list[Ints]
        Контуры принятых кандидатов.
    area : Floats
        Площадь каждого контура.
    perimeter : Floats
        Периметр каждого контура.
    """

    contours: list[Ints]
    area: Floats
    perimeter: Floats

    def __len__(self) -> int:
        return len(self.contours)


def pad_to_stride(image: Mat, stride: int = DETECTOR_STRIDE) -> PaddedImage:
    """Добить кадр белым до кратности stride.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.
    stride : int
        Требуемая кратность размеров.

    Returns
    -------
    PaddedImage
        Кадр после паддинга и число дописанных строк и столбцов.

    Notes
    -----
    Паддинг вместо ресайза сохраняет масштаб равным единице, поэтому карта вероятностей ложится на
    исходные пиксели без пересчета координат.
    """
    height, width = image.shape[:2]
    added_rows = (-height) % stride
    added_cols = (-width) % stride
    if added_rows == 0 and added_cols == 0:
        return PaddedImage(image=image, added_rows=0, added_cols=0)

    padded = cv2.copyMakeBorder(image, 0, added_rows, 0, added_cols, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return PaddedImage(image=padded, added_rows=added_rows, added_cols=added_cols)


def build_detector_blob(image: Mat) -> npt.NDArray[np.float32]:
    """Подготовить вход сети: нормализация ImageNet и порядок осей NCHW.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.

    Returns
    -------
    npt.NDArray[np.float32]
        Массив формы (1, 3, высота, ширина).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]


def resize_probability_map(probability_map: Floats, shape: tuple[int, int]) -> Floats:
    """Привести карту вероятностей к заданному размеру.

    Parameters
    ----------
    probability_map : Floats
        Карта вероятностей.
    shape : tuple[int, int]
        Требуемый размер в порядке высота, ширина.

    Returns
    -------
    Floats
        Карта требуемого размера.
    """
    interpolation = cv2.INTER_AREA if probability_map.shape[0] > shape[0] else cv2.INTER_LINEAR
    resized = cv2.resize(probability_map.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)
    return resized.astype(np.float64)


class TextProbabilityDetector:
    """Обертка над onnx-сессией DBNet, выдающая карту вероятностей текста.

    Parameters
    ----------
    model_path : Path
        Путь к файлу модели в формате onnx.

    Attributes
    ----------
    input_name : str
        Имя входного тензора модели.
    """

    def __init__(self, model_path: Path) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"модель не найдена: {model_path}")
        self._session = ort.InferenceSession(str(model_path))
        self.input_name: str = self._session.get_inputs()[0].name

    def detect_probability_map(self, image: Mat) -> Floats:
        """Прогнать кадр через сеть один раз.

        Parameters
        ----------
        image : Mat
            Кадр в порядке каналов BGR.

        Returns
        -------
        Floats
            Карта вероятностей того же размера, что и кадр.
        """
        padded = pad_to_stride(image)
        raw = self._session.run(None, {self.input_name: build_detector_blob(padded.image)})[0]
        height, width = raw.shape[-2:]
        return raw[0, 0, : height - padded.added_rows, : width - padded.added_cols].astype(np.float64)

    def detect_multiscale_maps(self, image: Mat, long_sides: tuple[int, ...]) -> Floats:
        """Снять карты вероятностей на нескольких масштабах.

        Parameters
        ----------
        image : Mat
            Кадр в порядке каналов BGR.
        long_sides : tuple[int, ...]
            Размеры длинной стороны, на которых прогоняется сеть.

        Returns
        -------
        Floats
            Стек карт формы (число масштабов, высота, ширина) в разрешении исходного кадра.

        Notes
        -----
        Мелкие надписи различимы на крупном масштабе, крупные заголовки на мелком, поэтому карты
        снимаются несколько раз и усредняются вызывающей стороной.
        """
        maps: list[Floats] = []
        for long_side in long_sides:
            small = resize_to_long_side(image, long_side)
            maps.append(resize_probability_map(self.detect_probability_map(small), image.shape[:2]))
        return np.stack(maps)


def measure_box_score(probability_map: Floats, box: Floats) -> float:
    """Средняя вероятность внутри четырехугольника.

    Parameters
    ----------
    probability_map : Floats
        Карта вероятностей.
    box : Floats
        Четыре угла прямоугольника в порядке x, y.

    Returns
    -------
    float
        Средняя вероятность внутри прямоугольника.
    """
    height, width = probability_map.shape[:2]
    corners = box.copy()
    left, right = np.clip([np.floor(corners[:, 0].min()), np.ceil(corners[:, 0].max())], 0, width - 1).astype(np.int32)
    top, bottom = np.clip([np.floor(corners[:, 1].min()), np.ceil(corners[:, 1].max())], 0, height - 1).astype(np.int32)

    stencil = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
    corners[:, 0] -= left
    corners[:, 1] -= top
    cv2.fillPoly(stencil, corners.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(probability_map[top : bottom + 1, left : right + 1], stencil)[0])


def find_text_candidates(probability_map: Floats, binary_threshold: float, max_count: int = 1000) -> TextCandidates:
    """Найти кандидатов в текстовые ядра по бинаризованной карте.

    Parameters
    ----------
    probability_map : Floats
        Карта вероятностей.
    binary_threshold : float
        Порог бинаризации карты, отвечает только за связность контуров.
    max_count : int
        Верхняя граница числа рассматриваемых контуров.

    Returns
    -------
    TextCandidates
        Контуры и их метрики. Решение о приеме принимается отдельно.
    """
    found, _ = cv2.findContours(
        (probability_map > binary_threshold).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = list(found[:max_count])
    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    sides = np.array([rectangle[1] for rectangle in rectangles], dtype=np.float64).reshape(-1, 2)
    return TextCandidates(
        contours=contours,
        score=np.array([measure_box_score(probability_map, cv2.boxPoints(r)) for r in rectangles], dtype=np.float64),
        short_side=sides.min(axis=1) if len(sides) else np.zeros(0),
        area=np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float64),
        perimeter=np.array([cv2.arcLength(contour, True) for contour in contours], dtype=np.float64),
    )


def accept_candidates(candidates: TextCandidates, box_threshold: float, minimum_short_side: float) -> Mask:
    """Отобрать кандидатов по средней вероятности и минимальной стороне.

    Parameters
    ----------
    candidates : TextCandidates
        Кандидаты и их метрики.
    box_threshold : float
        Нижняя граница средней вероятности.
    minimum_short_side : float
        Нижняя граница короткой стороны прямоугольника в пикселях.

    Returns
    -------
    Mask
        Маска принятых кандидатов.
    """
    return (candidates.score >= box_threshold) & (candidates.short_side >= minimum_short_side)


def take_accepted_kernels(candidates: TextCandidates, accepted: Mask) -> TextKernels:
    """Оставить только принятых кандидатов.

    Parameters
    ----------
    candidates : TextCandidates
        Кандидаты и их метрики.
    accepted : Mask
        Маска принятых кандидатов.

    Returns
    -------
    TextKernels
        Контуры принятых ядер с площадью и периметром.
    """
    indices = np.flatnonzero(accepted)
    return TextKernels(
        contours=[candidates.contours[index] for index in indices],
        area=candidates.area[indices],
        perimeter=candidates.perimeter[indices],
    )


def compute_unclip_distance(area: float, perimeter: float, unclip_ratio: float) -> int:
    """Расстояние, на которое раздувается контур ядра.

    Parameters
    ----------
    area : float
        Площадь контура.
    perimeter : float
        Периметр контура.
    unclip_ratio : float
        Коэффициент раздувания.

    Returns
    -------
    int
        Расстояние в пикселях.
    """
    return int(round(area * unclip_ratio / perimeter)) if perimeter > 0 else 0


def dilate_contour_mask(contour: Ints, distance: int, shape: tuple[int, int]) -> Mask:
    """Залить контур и раздуть заливку на заданное расстояние.

    Parameters
    ----------
    contour : Ints
        Контур ядра.
    distance : int
        Расстояние раздувания в пикселях.
    shape : tuple[int, int]
        Размер кадра в порядке высота, ширина.

    Returns
    -------
    Mask
        Маска раздутого ядра.

    Notes
    -----
    Работа идет в габарите контура, поэтому стоимость не зависит от размера листа.
    """
    height, width = shape
    left, top, box_width, box_height = cv2.boundingRect(contour)
    margin = distance + 2
    x_start, y_start = max(0, left - margin), max(0, top - margin)
    x_stop, y_stop = min(width, left + box_width + margin), min(height, top + box_height + margin)

    patch = np.zeros((y_stop - y_start, x_stop - x_start), np.uint8)
    cv2.drawContours(patch, [contour - np.array([[x_start, y_start]])], -1, 255, cv2.FILLED)
    if distance > 0:
        outside = cv2.distanceTransform(
            np.where(patch > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        patch = np.where((outside <= distance) | (patch > 0), 255, 0).astype(np.uint8)

    region = np.zeros(shape, dtype=bool)
    region[y_start:y_stop, x_start:x_stop] = patch > 0
    return region


def build_text_region_mask(kernels: TextKernels, unclip_ratio: float, shape: tuple[int, int]) -> Mask:
    """Собрать маску текстовой области из раздутых ядер.

    Parameters
    ----------
    kernels : TextKernels
        Принятые текстовые ядра.
    unclip_ratio : float
        Коэффициент раздувания контуров.
    shape : tuple[int, int]
        Размер кадра в порядке высота, ширина.

    Returns
    -------
    Mask
        Объединение раздутых ядер.
    """
    region = np.zeros(shape, dtype=bool)
    for contour, area, perimeter in zip(kernels.contours, kernels.area, kernels.perimeter, strict=True):
        region |= dilate_contour_mask(contour, compute_unclip_distance(area, perimeter, unclip_ratio), shape)
    return region
