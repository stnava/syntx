import numpy as np
import pytest

from syntx.core.losses import NGFReference, ngf_alignment


def _old_formula(a, b, sp, m, nf=0.02):
    """Verbatim copy of the antsxdwi implementation this module replaced (the bit-equality reference)."""
    ga = np.stack(np.gradient(a, *sp), axis=-1)
    gb = np.stack(np.gradient(b, *sp), axis=-1)
    na, nb = np.linalg.norm(ga, axis=-1), np.linalg.norm(gb, axis=-1)
    ea, eb = nf * float(na[m].mean()), nf * float(nb[m].mean())
    cos2 = (ga * gb).sum(-1) ** 2 / ((na**2 + ea**2) * (nb**2 + eb**2))
    w = (na**2 / (na**2 + ea**2)) * (nb**2 / (nb**2 + eb**2))
    return float(cos2[m].sum() / w[m].sum())


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    x, y, z = np.meshgrid(*[np.linspace(0, 1, 16)] * 3, indexing="ij")
    a = np.exp(-((x - 0.5) ** 2 + (y - 0.4) ** 2 + (z - 0.5) ** 2) / 0.05)
    b = np.exp(-((x - 0.55) ** 2 + (y - 0.4) ** 2 + (z - 0.5) ** 2) / 0.07) + 0.01 * rng.normal(size=a.shape)
    m = a > 0.05
    return a, b, m


def test_matches_original_formula(pair):
    a, b, m = pair
    sp = (1.5, 2.0, 1.0)
    assert ngf_alignment(a, b, sp, m) == pytest.approx(_old_formula(a, b, sp, m), rel=1e-12)


def test_reference_reuse_matches_one_shot(pair):
    a, b, m = pair
    ref = NGFReference(a, (1.0, 1.0, 1.0), m)
    assert ref.alignment(b) == ngf_alignment(a, b, (1.0, 1.0, 1.0), m)


def test_identity_inversion_and_scale(pair):
    a, _, m = pair
    sp = (1.0, 1.0, 1.0)
    assert ngf_alignment(a, a, sp, m) == pytest.approx(1.0)
    assert ngf_alignment(a, -3.0 * a + 2.0, sp, m) == pytest.approx(1.0)


def test_errors(pair):
    a, b, m = pair
    with pytest.raises(ValueError, match="empty mask"):
        ngf_alignment(a, b, (1, 1, 1), np.zeros_like(m))
    with pytest.raises(ValueError, match="no gradient"):
        ngf_alignment(np.ones_like(a), b, (1, 1, 1), m)
