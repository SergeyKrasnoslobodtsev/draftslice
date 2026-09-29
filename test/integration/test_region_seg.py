from pathlib import Path

import pytest

from draftslice.core.io import load_image
from draftslice.core.region_seg.pipeline import run

ROOT = Path(__file__).parents[2]
APPENDIX = ROOT / "docs/canon/source/appendix1"
IMAGES = ROOT / "data/images"
DATA = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if IMAGES.is_dir() else []

pytestmark = pytest.mark.integration


@pytest.mark.req("OUT-04", "KEEP-01")
@pytest.mark.parametrize(
    ("name", "n_views"),
    [
        ("ex1_input.jpg", 4),
        pytest.param(
            "ex2_input.jpg",
            2,
            marks=pytest.mark.xfail(strict=True, reason="таблица проходит как вид, DEL-07 открыто"),
        ),
        ("ex3_input.jpg", 3),
    ],
)
def test_appendix_view_count(name, n_views):
    assert len(run(load_image(str(APPENDIX / name)))) == n_views


@pytest.mark.req("VIEW-01")
@pytest.mark.parametrize("path", DATA, ids=lambda p: p.name)
def test_drawing_crops_are_inside_sheet(path):
    img = load_image(str(path))
    img_h, img_w = img.shape[:2]

    crops = run(img)

    assert crops
    for c in crops:
        assert 0 <= c.x and c.x + c.w <= img_w
        assert 0 <= c.y and c.y + c.h <= img_h
        assert c.img.shape == (c.h, c.w, 3)
