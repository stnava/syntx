"""
ANTs <-> syntx interoperability contract.

Goal: register with syntx, then use classical ants tools (apply_transforms,
apply_transforms_to_points, label transfer), or register with ants and consume the
result in syntx -- with no accuracy penalty in either direction, INCLUDING at the
field-of-view boundary.

Two scenarios per test:
  - 'interior':  anatomy sits well inside the FOV (zero background margin at the edges).
  - 'tight_fov': content fills the whole volume and the motion pushes it across the edges,
                 so boundary/extrapolation semantics are exercised directly.

Each test reports full-image / interior / edge-shell numbers so failures localise.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import ants

import syntx
from syntx.core.affine import compute_initial_grid
from syntx.core.grid import grid_sample_nd

# ----------------------------------------------------------------------------------------
# Contract tolerances
# ----------------------------------------------------------------------------------------
REL_INTENSITY_TOL = 1e-3      # max |a-b| / intensity range, per region
LABEL_AGREEMENT_TOL = 0.999   # fraction of voxels with identical label
ROUNDTRIP_EXCESS_TOL = 0.05   # syntx round-trip error may exceed ants' own by <= this (voxels)

EDGE = 2      # voxels from any face counted as "edge shell"
INTERIOR = 4  # voxels trimmed from every face for "interior"

SHAPE = (32, 32, 32)
SPACING = (1.2, 1.0, 1.4)
ORIGIN = (5.0, -3.0, 2.0)
DIRECTION = np.diag([1.0, -1.0, 1.0])


# ----------------------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------------------
def _grids():
    x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in SHAPE], indexing="ij")
    return x, y, z


def _make_pair(case):
    x, y, z = _grids()
    c = (np.array(SHAPE) - 1) / 2.0
    if case == "interior":
        def blob(cx, cy, cz):
            r2 = ((x - cx) / 7.0) ** 2 + ((y - cy) / 6.0) ** 2 + ((z - cz) / 5.0) ** 2
            return (100.0 * np.exp(-r2) * (1.0 + 0.3 * np.sin(x / 2.0))).astype(np.float32)
        f = blob(*c)
        m = blob(c[0] + 1.5, c[1] - 1.0, c[2] + 0.8)
    elif case == "tight_fov":
        def tex(sx, sy, sz):
            return (60.0 + 25.0 * np.sin((x + sx) / 3.0) * np.cos((y + sy) / 4.0)
                    + 15.0 * np.sin((z + sz) / 2.5) + 0.8 * (x + sx)).astype(np.float32)
        f = tex(0.0, 0.0, 0.0)
        m = tex(2.3, -1.7, 1.2)   # content crosses every face
    else:
        raise ValueError(case)

    mk = lambda a: ants.from_numpy(a, origin=ORIGIN, spacing=SPACING, direction=DIRECTION)
    fi, mi = mk(f), mk(m)
    # label map on the moving image: 3 intensity-quantile classes (background = 0)
    thr = np.quantile(m[m > 0], [0.33, 0.66]) if (m > 0).any() else [0, 0]
    lab = np.zeros_like(m, dtype=np.float32)
    lab[m > 0.05 * m.max()] = 1
    lab[m > thr[0]] = 2
    lab[m > thr[1]] = 3
    return fi, mi, mk(lab)


def _regions():
    interior = tuple(slice(INTERIOR, n - INTERIOR) for n in SHAPE)
    edge = np.zeros(SHAPE, dtype=bool)
    for ax, n in enumerate(SHAPE):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, EDGE); edge[tuple(sl)] = True
        sl[ax] = slice(n - EDGE, n); edge[tuple(sl)] = True
    return interior, edge


def _rel_diffs(a, b):
    """max |a-b| / range over full / interior / edge shell."""
    rng = max(float(np.ptp(b)), 1e-8)
    d = np.abs(a - b)
    interior, edge = _regions()
    return {
        "full": float(d.max() / rng),
        "interior": float(d[interior].max() / rng),
        "edge": float(d[edge].max() / rng),
        "mean_full": float(d.mean() / rng),
    }


def _fmt(tag, r):
    return (f"[{tag}] rel max|diff| full={r['full']:.3e} interior={r['interior']:.3e} "
            f"edge={r['edge']:.3e} (mean full={r['mean_full']:.3e})")


# ----------------------------------------------------------------------------------------
# Registrations (computed once per case)
# ----------------------------------------------------------------------------------------
_CACHE = {}


def _syntx_reg(case):
    key = ("syntx", case)
    if key not in _CACHE:
        fi, mi, lab = _make_pair(case)
        torch.manual_seed(0)
        reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 10, 5], device="cpu",
                        verbose=False)
        _CACHE[key] = (fi, mi, lab, reg)
    return _CACHE[key]


def _ants_reg(case):
    key = ("ants", case)
    if key not in _CACHE:
        fi, mi, lab = _make_pair(case)
        reg = ants.registration(fixed=fi, moving=mi, type_of_transform="SyN",
                                reg_iterations=(20, 10, 5), random_seed=1)
        _CACHE[key] = (fi, mi, lab, reg)
    return _CACHE[key]


CASES = ["interior", "tight_fov"]


# ----------------------------------------------------------------------------------------
# 1. syntx registers -> ants applies (image, forward)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_syntx_native_forward_warp_matches_ants_application(case):
    """The warp syntx optimised against (its own native resampler) must be what ants
    reproduces when applying syntx's exported transform files."""
    fi, mi, _, reg = _syntx_reg(case)
    native = reg["model"].forward(mi).numpy()
    via_ants = ants.apply_transforms(fixed=fi, moving=mi,
                                     transformlist=reg["fwdtransforms"]).numpy()
    r = _rel_diffs(native, via_ants)
    print(_fmt(f"{case} fwd syntx-native vs ants", r))
    assert r["full"] < REL_INTENSITY_TOL, _fmt(case, r)


# ----------------------------------------------------------------------------------------
# 2. syntx registers -> ants applies (image, inverse)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_syntx_native_inverse_warp_matches_ants_application(case):
    fi, mi, _, reg = _syntx_reg(case)
    native = reg["model"].forward_inverse(fi).numpy()
    via_ants = ants.apply_transforms(fixed=mi, moving=fi,
                                     transformlist=reg["invtransforms"],
                                     whichtoinvert=reg["whichtoinvert_inv"]).numpy()
    r = _rel_diffs(native, via_ants)
    print(_fmt(f"{case} inv syntx-native vs ants", r))
    assert r["full"] < REL_INTENSITY_TOL, _fmt(case, r)


# ----------------------------------------------------------------------------------------
# 3. Label transfer: syntx native nearest/genericLabel vs ants nearestNeighbor/genericLabel
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("ants_interp", ["nearestNeighbor", "genericLabel"])
def test_label_transfer_syntx_vs_ants(case, ants_interp):
    fi, mi, lab, reg = _syntx_reg(case)
    model = reg["model"]
    old = model.interpolator
    model.interpolator = "nearest" if ants_interp == "nearestNeighbor" else "genericLabel"
    try:
        native = np.rint(model.forward(lab).numpy())
    finally:
        model.interpolator = old
    via_ants = np.rint(ants.apply_transforms(fixed=fi, moving=lab,
                                             transformlist=reg["fwdtransforms"],
                                             interpolator=ants_interp).numpy())
    interior, edge = _regions()
    agree = float((native == via_ants).mean())
    agree_int = float((native[interior] == via_ants[interior]).mean())
    agree_edge = float((native[edge] == via_ants[edge]).mean())
    print(f"[{case} labels vs ants {ants_interp}] agreement full={agree:.5f} "
          f"interior={agree_int:.5f} edge={agree_edge:.5f}")
    assert agree >= LABEL_AGREEMENT_TOL


# ----------------------------------------------------------------------------------------
# 4. ants registers -> syntx consumes (syntx's own ingestion path)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_ants_transforms_ingested_by_syntx(case):
    """syntx's path for consuming an ants transform list (compute_initial_grid, as used by
    registration(initial_transform=...)) followed by syntx sampling must reproduce ants'
    own warpedmovout."""
    fi, mi, _, reg = _ants_reg(case)
    grid = torch.from_numpy(compute_initial_grid(fi, mi, reg["fwdtransforms"]))
    mov = torch.from_numpy(np.ascontiguousarray(
        np.transpose(mi.numpy().astype(np.float32), (2, 1, 0))))[None, None]
    warped = grid_sample_nd(mov, grid, padding_mode="itk")[0, 0].permute(2, 1, 0).numpy()
    r = _rel_diffs(warped, reg["warpedmovout"].numpy())
    print(_fmt(f"{case} ants-tx ingested by syntx vs ants", r))
    assert r["full"] < REL_INTENSITY_TOL, _fmt(case, r)


def _origin_inside_pair():
    """Moving image whose physical (0,0,0) lies INSIDE the volume, non-cubic, oblique-ish
    header, bright everywhere -- so a point wrongly sent to (0,0,0) samples tissue."""
    shape = (30, 26, 22)
    x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    arr = (50.0 + 20.0 * np.sin(x / 3.0) + 10.0 * np.cos(y / 4.0) + 0.5 * z).astype(np.float32)
    kw = dict(origin=(-12.0, 10.0, -9.0), spacing=(1.0, 1.1, 0.9),
              direction=np.diag([1.0, -1.0, 1.0]))
    return ants.from_numpy(arr.copy(), **kw), ants.from_numpy(arr.copy(), **kw)


@pytest.mark.parametrize("shift", [(14.0, 0.0, 0.0), (-9.0, 6.0, 5.0)])
def test_ingested_transform_out_of_domain_is_background(shift, tmp_path):
    """Fixed points that an ingested ants transform maps OUTSIDE the moving image must
    sample background (ants semantics) -- never tissue at physical (0,0,0)."""
    fi, mi = _origin_inside_pair()
    idx0 = np.linalg.solve(np.asarray(mi.direction) @ np.diag(mi.spacing),
                           -np.asarray(mi.origin))
    assert np.all(idx0 > 1) and np.all(idx0 < np.array(mi.shape) - 2), idx0
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3,
                                    translation=shift)
    txf = str(tmp_path / "shift.mat")
    ants.write_transform(tx, txf)

    grid = torch.from_numpy(compute_initial_grid(fi, mi, [txf]))
    mov = torch.from_numpy(np.ascontiguousarray(
        np.transpose(mi.numpy().astype(np.float32), (2, 1, 0))))[None, None]
    warped = grid_sample_nd(mov, grid, padding_mode="itk")[0, 0].permute(2, 1, 0).numpy()
    ref = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[txf]).numpy()
    outside = ref == 0
    assert outside.mean() > 0.05  # the shift really leaves the FOV
    assert np.abs(warped[outside]).max() == 0.0
    rng = float(np.ptp(ref))
    assert float(np.abs(warped - ref).max()) / rng < REL_INTENSITY_TOL


# ----------------------------------------------------------------------------------------
# 4c. Distinct, non-cubic fixed/moving headers: every native syntx apply path vs ants
# ----------------------------------------------------------------------------------------
def _distinct_header_reg():
    key = ("syntx", "distinct_headers")
    if key not in _CACHE:
        def img(shape, spacing, origin, direction, shift):
            x, y, z = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
            c = (np.array(shape) - 1) / 2.0 + np.array(shift)
            r2 = ((x - c[0]) / 8.0) ** 2 + ((y - c[1]) / 6.0) ** 2 + ((z - c[2]) / 5.0) ** 2
            a = (100.0 * np.exp(-r2) * (1.0 + 0.3 * np.sin(x / 2.0))).astype(np.float32)
            return ants.from_numpy(a, origin=origin, spacing=spacing, direction=direction)
        fi = img((34, 28, 24), (1.1, 1.3, 1.5), (4.0, -6.0, 3.0), np.diag([1.0, -1.0, 1.0]), (0, 0, 0))
        mi = img((30, 32, 26), (1.2, 1.1, 1.4), (6.0, -4.0, 1.5), np.diag([1.0, -1.0, 1.0]), (1.0, -1.0, 0.5))
        torch.manual_seed(0)
        reg = syntx.syn(fixed=fi, moving=mi, reg_iterations=[20, 10, 5], device="cpu",
                        verbose=False)
        _CACHE[key] = (fi, mi, reg)
    return _CACHE[key]


def _to_zyx(img):
    return torch.from_numpy(np.ascontiguousarray(
        np.transpose(img.numpy().astype(np.float32), (2, 1, 0))))[None, None]


def test_distinct_headers_native_forward_and_inverse_match_ants():
    fi, mi, reg = _distinct_header_reg()
    model = reg["model"]
    fwd_ants = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"]).numpy()
    inv_ants = ants.apply_transforms(fixed=mi, moving=fi, transformlist=reg["invtransforms"],
                                     whichtoinvert=reg["whichtoinvert_inv"]).numpy()
    fwd_native = model.forward(mi).numpy()
    inv_native = model.forward_inverse(fi).numpy()
    assert fwd_native.shape == fi.shape and inv_native.shape == mi.shape
    for tag, a, b in [("fwd", fwd_native, fwd_ants), ("inv", inv_native, inv_ants)]:
        rel = float(np.abs(a - b).max() / np.ptp(b))
        print(f"[distinct headers {tag} native vs ants] rel max|diff|={rel:.3e}")
        assert rel < REL_INTENSITY_TOL, (tag, rel)


def test_distinct_headers_transform_apply_matches_ants():
    fi, mi, reg = _distinct_header_reg()
    tx = reg["model"].to_transform(fixed=fi, moving=mi)
    applied = tx.apply(_to_zyx(mi))[0, 0].permute(2, 1, 0).numpy()
    ref = ants.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"]).numpy()
    assert applied.shape == ref.shape
    rel = float(np.abs(applied - ref).max() / np.ptp(ref))
    print(f"[distinct headers SyNToTransform.apply vs ants] rel max|diff|={rel:.3e}")
    assert rel < REL_INTENSITY_TOL


# ----------------------------------------------------------------------------------------
# 5. syntx's own composite (single-field) export vs ants applying the transform list
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_syntx_composite_export_matches_transform_list(case, tmp_path):
    fi, mi, _, reg = _syntx_reg(case)
    tx = reg["model"].to_transform(fixed=fi, moving=mi)
    comp = tx.to_composite_warp(str(tmp_path / "composite.nii.gz"))
    via_list = ants.apply_transforms(fixed=fi, moving=mi,
                                     transformlist=reg["fwdtransforms"]).numpy()
    via_comp = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[comp]).numpy()
    r = _rel_diffs(via_comp, via_list)
    print(_fmt(f"{case} composite export vs list", r))
    assert r["full"] < REL_INTENSITY_TOL, _fmt(case, r)


# ----------------------------------------------------------------------------------------
# 6. Round trip fwd o inv through ants point tools: syntx transforms vs ants' own
# ----------------------------------------------------------------------------------------
def _roundtrip_error_vox(fi, fwd, inv, whichinv):
    x, y, z = _grids()
    sub = (slice(None, None, 2),) * 3
    idx = np.stack([x[sub].ravel(), y[sub].ravel(), z[sub].ravel()], axis=1)
    phys = (DIRECTION @ (idx * np.array(SPACING)).T).T + np.array(ORIGIN)
    pts = pd.DataFrame(phys, columns=["x", "y", "z"])
    p1 = ants.apply_transforms_to_points(3, pts, fwd, whichtoinvert=[False] * len(fwd))
    p2 = ants.apply_transforms_to_points(3, p1, inv, whichtoinvert=whichinv)
    err_mm = np.linalg.norm(p2[["x", "y", "z"]].values - phys, axis=1)
    err_vox = err_mm / min(SPACING)
    n = np.array(SHAPE)
    dist_to_face = np.minimum(idx, (n - 1) - idx).min(axis=1)
    return err_vox, dist_to_face


@pytest.mark.parametrize("case", CASES)
def test_roundtrip_via_ants_point_tools(case):
    fi, _, _, sreg = _syntx_reg(case)
    _, _, _, areg = _ants_reg(case)
    s_err, dist = _roundtrip_error_vox(fi, sreg["fwdtransforms"], sreg["invtransforms"],
                                       sreg["whichtoinvert_inv"])
    a_err, _ = _roundtrip_error_vox(fi, areg["fwdtransforms"], areg["invtransforms"],
                                    [True, False])
    edge = dist < EDGE
    inner = dist >= INTERIOR
    rep = {}
    for name, e in [("syntx", s_err), ("ants", a_err)]:
        rep[name] = (np.median(e[inner]), np.percentile(e[inner], 95),
                     np.median(e[edge]), np.percentile(e[edge], 95), e.max())
        print(f"[{case} roundtrip {name}] interior median={rep[name][0]:.4f} "
              f"p95={rep[name][1]:.4f} | edge median={rep[name][2]:.4f} "
              f"p95={rep[name][3]:.4f} | max={rep[name][4]:.4f} (voxels)")
    assert rep["syntx"][1] <= rep["ants"][1] + ROUNDTRIP_EXCESS_TOL
    assert rep["syntx"][3] <= rep["ants"][3] + ROUNDTRIP_EXCESS_TOL
