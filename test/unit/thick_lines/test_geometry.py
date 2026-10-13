import numpy as np
import pytest

from draftslice.core.thick_lines.geometry import fit_circle, is_arc


def _arc(cy, cx, r, deg):
    """Точки дуги окружности на сетке пикселей, без повторов."""
    a = np.deg2rad(np.linspace(0, deg, int(r * np.deg2rad(deg) * 2)))
    p = np.rint(np.c_[cy + r * np.sin(a), cx + r * np.cos(a)]).astype(np.int64)
    return p[np.r_[True, (np.diff(p, axis=0) != 0).any(axis=1)]]


@pytest.mark.req("KEEP-01")
def test_straight_line():
    p = np.c_[np.arange(50), np.arange(50) // 10]  # ступеньки сетки на пологой прямой
    c = fit_circle(p)
    assert np.isinf(c.r) and np.isnan(c.center).all() and c.rel_err == 0
    assert not is_arc(p, c)


@pytest.mark.req("KEEP-01")
def test_arc_recovers_circle():
    """Дуга 60°: Гаусс–Ньютон снимает занижение радиуса Каса."""
    c = fit_circle(_arc(200, 300, 80, 60))
    assert c.r == pytest.approx(80, rel=0.01)
    np.testing.assert_allclose(c.center, (200, 300), atol=1.0)
    assert is_arc(_arc(200, 300, 80, 60), c)


@pytest.mark.req("KEEP-01")
def test_nearly_straight_branch_is_finite():
    """Прямая со ступенькой 2 px: радиус ~20 тыс. px, решение устойчиво, дугой не считается."""
    x = np.arange(93)
    p = np.c_[1239 + x, 6714 + 2 * (x > 46)]
    c = fit_circle(p)
    assert np.isfinite(c.r) and c.r > 1000
    assert not is_arc(p, c)
