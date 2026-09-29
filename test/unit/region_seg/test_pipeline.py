import cv2
import numpy as np
import pytest

from draftslice.core.region_seg.pipeline import run

BLACK = (0, 0, 0)
U_VIEW = np.array([(400, 60), (430, 60), (430, 230), (570, 230), (570, 60), (600, 60), (600, 260), (400, 260)])


@pytest.mark.req("SCN-03")
def test_crop_box_is_tight_around_view(sheet):
    view = np.zeros(sheet.shape[:2], np.uint8)
    cv2.rectangle(view, (100, 100), (300, 250), 1, 3)
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)
    rows, cols = np.nonzero(view)

    (crop,) = run(sheet)

    assert (crop.x, crop.y) == (cols.min(), rows.min())
    assert (crop.w, crop.h) == (cols.max() - cols.min() + 1, rows.max() - rows.min() + 1)
    assert crop.mask.shape == (crop.h, crop.w)


def test_crop_img_keeps_segment_and_whitens_foreign_ink(sheet):
    cv2.polylines(sheet, [U_VIEW], isClosed=True, color=BLACK, thickness=2)
    cv2.rectangle(sheet, (480, 120), (488, 130), BLACK, 1)  # мелочь в вырезе U - внутри бокса, не в сегменте

    (crop,) = run(sheet)
    img = crop.img
    dy, dx = 120 - crop.y, 480 - crop.x

    assert crop.src[dy, dx].tolist() == [0, 0, 0]
    assert img[dy, dx].tolist() == [255, 255, 255]
    assert (img[crop.mask] == crop.src[crop.mask]).all()


def test_crop_src_is_view_of_input_not_copy(sheet):
    cv2.rectangle(sheet, (100, 100), (300, 250), BLACK, 3)

    (crop,) = run(sheet)

    assert np.shares_memory(crop.src, sheet)


@pytest.mark.req("VIEW-02")
def test_sheet_without_views_gives_no_crops(sheet):
    assert run(sheet) == []
