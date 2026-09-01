"""Сквозной прогон всех этапов очистки чертежа.

Модуль ничего не решает сам, он только соединяет этапы в том порядке, в котором они разбирались в
этапных блокнотах: текст, бинаризация, граф, цепи, классы толщины, перерисовка, замыкание, второй
граф, признаки компонент, правила отбора, итоговая перерисовка.

Промежуточные результаты возвращаются целиком, а не выбрасываются: метрики снимаются снаружи, и по
одному прогону можно судить обо всех этапах сразу. Печати здесь нет, поэтому функцию одинаково удобно
вызывать из блокнота и из скрипта, считающего таблицу по датасету.
"""

from dataclasses import dataclass

import numpy as np

from draftslice.chain_merging import StrokeChains, build_stroke_chains
from draftslice.common_types import Mask, Mat
from draftslice.features import (
    ComponentFeatures,
    build_component_features,
    expand_selection_to_paths,
    find_enclosed_components,
    find_host_contours,
    find_repeated_components,
    find_text_sized_components,
    group_similar_components,
    measure_text_component_scale,
    select_components,
    trim_short_twigs,
)
from draftslice.image_preparation import binarize_strokes, scale_image, scale_mask
from draftslice.morphology_closing import close_along_orientations, suggest_kernel_length
from draftslice.pipeline_parameters import PipelineParameters
from draftslice.stroke_graph import StrokeGraph, build_stroke_graph
from draftslice.stroke_painting import paint_selected_paths
from draftslice.text_detection import (
    TextProbabilityDetector,
    accept_candidates,
    build_text_region_mask,
    find_text_candidates,
    take_accepted_kernels,
)
from draftslice.thickness_classes import ThicknessClasses, fit_thickness_classes, select_chains_by_class
from draftslice.view_extraction import group_mask_components, normalize_view_crops, render_view


@dataclass(frozen=True, eq=False)
class PipelineArtifacts:
    """Все промежуточные результаты одного прогона.

    Attributes
    ----------
    scaled_image : Mat
        Кадр в рабочем масштабе.
    text_region_mask : Mask
        Маска текста в рабочем масштабе.
    strokes_mask : Mask
        Маска штрихов после бинаризации.
    strokes_graph : StrokeGraph
        Граф скелета исходных штрихов.
    chains : StrokeChains
        Цепи, собранные из ребер.
    thickness_classes : ThicknessClasses
        Классы толщины цепей.
    clean_mask : Mask
        Линии, сохраненные по классу толщины.
    closing_kernel_length : int
        Длина ядра, которой выполнено замыкание.
    closed_mask : Mask
        Маска после замыкания разрывов.
    closed_graph : StrokeGraph
        Граф скелета замкнутой маски.
    features : ComponentFeatures
        Признаки компонент замкнутого графа.
    kept_paths : Mask
        Ребра, оставшиеся после правил по компонентам.
    final_paths : Mask
        Ребра, оставшиеся после обрезки хвостов.
    part_mask : Mask
        Итоговая маска детали, нарисованная по оригинальным штрихам.
    """

    scaled_image: Mat
    text_region_mask: Mask
    strokes_mask: Mask
    strokes_graph: StrokeGraph
    chains: StrokeChains
    thickness_classes: ThicknessClasses
    clean_mask: Mask
    closing_kernel_length: int
    closed_mask: Mask
    closed_graph: StrokeGraph
    features: ComponentFeatures
    kept_paths: Mask
    final_paths: Mask
    part_mask: Mask


def detect_text_region(image: Mat, detector: TextProbabilityDetector, parameters: PipelineParameters) -> Mask:
    """Построить маску текстовой области в масштабе исходного кадра.

    Parameters
    ----------
    image : Mat
        Кадр в порядке каналов BGR.
    detector : TextProbabilityDetector
        Загруженная модель детекции текста.
    parameters : PipelineParameters
        Параметры пайплайна.

    Returns
    -------
    Mask
        Маска текстовой области.
    """
    probability_map = detector.detect_multiscale_maps(image, parameters.detector_scales).mean(axis=0)
    candidates = find_text_candidates(probability_map, parameters.text_binary_threshold)
    accepted = accept_candidates(candidates, parameters.text_box_threshold, parameters.text_minimum_short_side)
    kernels = take_accepted_kernels(candidates, accepted)
    return build_text_region_mask(kernels, parameters.text_unclip_ratio, image.shape[:2])


def build_clean_mask(
    graph: StrokeGraph,
    chains: StrokeChains,
    classes: ThicknessClasses,
    strokes_mask: Mask,
    parameters: PipelineParameters,
) -> Mask:
    """Нарисовать линии, прошедшие по классу толщины.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета исходных штрихов.
    chains : StrokeChains
        Цепи кадра.
    classes : ThicknessClasses
        Классы толщины.
    strokes_mask : Mask
        Маска штрихов, источник пикселей.
    parameters : PipelineParameters
        Параметры пайплайна.

    Returns
    -------
    Mask
        Маска сохраненных линий.
    """
    kept_chains = select_chains_by_class(
        chains, classes, parameters.minimum_thickness_class, parameters.text_share_limit
    )
    kept_paths = chains.expand_selection_to_paths(kept_chains)
    return paint_selected_paths(graph, kept_paths, strokes_mask, parameters.radius_tolerance)


def select_part_paths(
    graph: StrokeGraph,
    features: ComponentFeatures,
    text_region_mask: Mask,
    parameters: PipelineParameters,
) -> Mask:
    """Отобрать ребра, относящиеся к детали, по правилам для компонент.

    Parameters
    ----------
    graph : StrokeGraph
        Граф скелета замкнутой маски.
    features : ComponentFeatures
        Признаки компонент.
    text_region_mask : Mask
        Маска текстовой области, задает масштаб шрифта на листе.
    parameters : PipelineParameters
        Параметры пайплайна.

    Returns
    -------
    Mask
        Маска ребер, прошедших правила.

    Notes
    -----
    Правила накладываются последовательно: сначала отбор по форме, затем снятие повторяющихся
    шаблонов, затем снятие компонент размера текста. Последнее правило не применяется к тому, что
    лежит внутри большого контура: отверстия и обозначения внутри тела по размеру неотличимы от
    надписей. Внутренности мелких замкнутых объектов вроде рамки допуска это не касается, там как
    раз текст и находится.
    """
    kept_components = select_components(
        features,
        minimum_inside_area=parameters.minimum_inside_area,
        maximum_thickness_spread=parameters.maximum_thickness_spread,
        minimum_tortuosity=parameters.minimum_tortuosity,
        maximum_inside_text_share=parameters.maximum_inside_text_share,
    )
    groups = group_similar_components(features, parameters.repeat_tolerance)
    repeated = find_repeated_components(features, groups, parameters.minimum_repeat_count)
    text_scale = measure_text_component_scale(text_region_mask, parameters.text_size_quantile)
    text_sized = find_text_sized_components(features, text_scale, parameters.text_size_tolerance)
    hosts = find_host_contours(features, parameters.host_area_share)
    enclosed = find_enclosed_components(graph, features, hosts, parameters.enclosed_inside_share)
    return expand_selection_to_paths(features, kept_components & ~repeated & ~(text_sized & ~enclosed))


def run_pipeline(
    image: Mat, detector: TextProbabilityDetector, parameters: PipelineParameters | None = None
) -> PipelineArtifacts:
    """Прогнать кадр через все этапы очистки.

    Parameters
    ----------
    image : Mat
        Кадр чертежа в порядке каналов BGR.
    detector : TextProbabilityDetector
        Загруженная модель детекции текста.
    parameters : PipelineParameters | None
        Параметры пайплайна, None означает значения по умолчанию.

    Returns
    -------
    PipelineArtifacts
        Все промежуточные результаты вместе с итоговой маской детали.
    """
    parameters = parameters or PipelineParameters()

    text_region_mask = detect_text_region(image, detector, parameters)
    scaled_image = scale_image(image, parameters.working_scale)
    scaled_text_region = scale_mask(text_region_mask, parameters.working_scale)
    strokes = binarize_strokes(scaled_image)

    strokes_graph = build_stroke_graph(strokes.strokes_mask, scaled_text_region)
    chains = build_stroke_chains(strokes_graph, parameters.chain)
    classes = fit_thickness_classes(
        chains, parameters.thickness_class_count, parameters.text_share_limit, parameters.weight_classes_by_length
    )
    clean_mask = build_clean_mask(strokes_graph, chains, classes, strokes.strokes_mask, parameters)

    kernel_length = parameters.closing_kernel_length or suggest_kernel_length(float(classes.edges[0]))
    closed_mask = close_along_orientations(clean_mask, kernel_length, parameters.closing_orientation_count)

    closed_graph = build_stroke_graph(closed_mask, scaled_text_region)
    all_paths = np.ones(closed_graph.path_count, dtype=bool)
    features = build_component_features(closed_graph, all_paths, scaled_text_region)
    kept_paths = select_part_paths(closed_graph, features, scaled_text_region, parameters)
    final_paths = trim_short_twigs(closed_graph, kept_paths, features.path_component, parameters.twig_length_share)
    part_mask = paint_selected_paths(closed_graph, final_paths, strokes.strokes_mask, parameters.radius_tolerance)

    return PipelineArtifacts(
        scaled_image=scaled_image,
        text_region_mask=scaled_text_region,
        strokes_mask=strokes.strokes_mask,
        strokes_graph=strokes_graph,
        chains=chains,
        thickness_classes=classes,
        clean_mask=clean_mask,
        closing_kernel_length=kernel_length,
        closed_mask=closed_mask,
        closed_graph=closed_graph,
        features=features,
        kept_paths=kept_paths,
        final_paths=final_paths,
        part_mask=part_mask,
    )


def extract_views(
    image: Mat, detector: TextProbabilityDetector, target_size: int, parameters: PipelineParameters | None = None
) -> list[Mat]:
    """Прогнать кадр через пайплайн очистки и разбить итоговую маску детали на виды.

    Parameters
    ----------
    image : Mat
        Кадр чертежа в порядке каналов BGR.
    detector : TextProbabilityDetector
        Загруженная модель детекции текста.
    target_size : int
        Сторона итогового квадрата кропа вида в пикселях, под вход конкретной модели эмбеддинга.
    parameters : PipelineParameters | None
        Пороги пайплайна, None означает значения по умолчанию.

    Returns
    -------
    list[Mat]
        Виды детали как RGB-картинки, все в едином масштабе и вписанные в квадрат `target_size`, от
        самого крупного вида к самому мелкому — готовые к подаче в модель эмбеддинга (CLIP, DINOv3).
    """
    parameters = parameters or PipelineParameters()

    text_region_mask = detect_text_region(image, detector, parameters)
    scaled_image = scale_image(image, parameters.working_scale)
    scaled_text_region = scale_mask(text_region_mask, parameters.working_scale)
    strokes = binarize_strokes(scaled_image)

    strokes_graph = build_stroke_graph(strokes.strokes_mask, scaled_text_region)
    line_thickness = strokes_graph.median_thickness
    chains = build_stroke_chains(strokes_graph, parameters.chain)
    classes = fit_thickness_classes(
        chains, parameters.thickness_class_count, parameters.text_share_limit, parameters.weight_classes_by_length
    )
    clean_mask = build_clean_mask(strokes_graph, chains, classes, strokes.strokes_mask, parameters)

    kernel_length = parameters.closing_kernel_length or suggest_kernel_length(float(classes.edges[0]))
    closed_mask = close_along_orientations(clean_mask, kernel_length, parameters.closing_orientation_count)

    closed_graph = build_stroke_graph(closed_mask, scaled_text_region)
    all_paths = np.ones(closed_graph.path_count, dtype=bool)
    features = build_component_features(closed_graph, all_paths, scaled_text_region)
    kept_paths = select_part_paths(closed_graph, features, scaled_text_region, parameters)
    final_paths = trim_short_twigs(closed_graph, kept_paths, features.path_component, parameters.twig_length_share)
    part_mask = paint_selected_paths(closed_graph, final_paths, strokes.strokes_mask, parameters.radius_tolerance)

    groups = group_mask_components(part_mask, line_thickness, parameters.view_merge_gap_thickness)
    crops = normalize_view_crops(part_mask, groups, parameters.view_padding_thickness * line_thickness, target_size)
    return [render_view(crop) for crop in crops]
