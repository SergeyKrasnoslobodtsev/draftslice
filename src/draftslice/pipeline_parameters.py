"""Параметры всех этапов пайплайна в одном месте.

Пороги подбираются в этапных блокнотах по одному кадру, но прогон по датасету требует единого набора
значений, заданного один раз. Поэтому параметры собраны в неизменяемый датакласс: его можно
сериализовать вместе с результатом, сравнить два прогона и понять, чем они отличались.

Значения по умолчанию это то, на чем сошелся ручной подбор. Единственный параметр, который остается
незаданным, это длина ядра замыкания: она зависит от разрешения исходника, и если оставить None, она
выбирается по кадру.
"""

from dataclasses import dataclass, field

from draftslice.chain_merging import ChainMergeParameters


@dataclass(frozen=True)
class PipelineParameters:
    """Пороги всех этапов.

    Attributes
    ----------
    detector_scales : tuple[int, ...]
        Размеры длинной стороны, на которых прогоняется детектор текста.
    text_binary_threshold : float
        Порог бинаризации карты вероятностей, отвечает за связность контуров.
    text_box_threshold : float
        Нижняя граница средней вероятности внутри прямоугольника кандидата.
    text_minimum_short_side : float
        Нижняя граница короткой стороны кандидата в пикселях.
    text_unclip_ratio : float
        Коэффициент раздувания контуров текстовых ядер.
    working_scale : float
        Множитель рабочего масштаба кадра.
    text_share_limit : float
        Доля текста, выше которой цепь считается текстовой.
    chain : ChainMergeParameters
        Параметры склейки ребер в цепи.
    thickness_class_count : int
        Число классов толщины.
    weight_classes_by_length : bool
        Взвешивать ли вклад цепи ее длиной при поиске классов.
    minimum_thickness_class : int
        Младший класс толщины, который еще сохраняется.
    radius_tolerance : float
        Допуск при перерисовке линий от оси.
    closing_kernel_length : int | None
        Длина ядра замыкания, None означает выбор по разрешению исходника.
    closing_orientation_count : int
        Число ориентаций линейного ядра.
    minimum_inside_area : int
        Внутренняя площадь, начиная с которой компонента считается замкнутой фигурой.
    maximum_thickness_spread : float
        Разброс толщины, выше которого компонента считается клином.
    minimum_tortuosity : float
        Извилистость, начиная с которой незамкнутая компонента считается осмысленной линией.
    maximum_inside_text_share : float
        Доля внутренней площади под текстом, выше которой компонента считается выноской.
    repeat_tolerance : float
        Относительный допуск на совпадение размеров при поиске повторов.
    minimum_repeat_count : int
        Размер группы, начиная с которого компоненты считаются шаблонными.
    twig_length_share : float
        Доля длины компоненты, ниже которой висячее ребро срезается.
    text_size_quantile : float
        Квантиль по габаритам кусков текстовой области: 0.5 медиана, 1.0 максимум.
    text_size_tolerance : float
        Множитель к характерному габариту текста, ноль отключает правило.
    enclosed_inside_share : float
        Доля пикселей компоненты внутри контура детали, при которой она считается вложенной.
    host_area_share : float
        Доля от наибольшей внутренней площади, начиная с которой контур считается телом детали.
    """

    detector_scales: tuple[int, ...] = (320, 640, 2000)
    text_binary_threshold: float = 0.50
    text_box_threshold: float = 0.60
    text_minimum_short_side: float = 3.0
    text_unclip_ratio: float = 1.5

    working_scale: float = 2.0
    text_share_limit: float = 0.5

    chain: ChainMergeParameters = field(default_factory=ChainMergeParameters)
    thickness_class_count: int = 2
    weight_classes_by_length: bool = True
    minimum_thickness_class: int = 1
    radius_tolerance: float = 0.5

    closing_kernel_length: int | None = None
    closing_orientation_count: int = 8

    minimum_inside_area: int = 1000
    maximum_thickness_spread: float = 2.5
    minimum_tortuosity: float = 1.6
    maximum_inside_text_share: float = 0.10
    repeat_tolerance: float = 0.05
    minimum_repeat_count: int = 3
    twig_length_share: float = 0.05

    text_size_quantile: float = 0.5
    text_size_tolerance: float = 1.0
    enclosed_inside_share: float = 0.9
    host_area_share: float = 0.2
