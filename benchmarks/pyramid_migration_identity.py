"""RECORD of what changed (not a gate since Pass 3c, which deliberately moved the solver onto the corrected ``mm_iso`` pyramid; the
Pass 3a migration was bit-identical, a40cd5e vs 1c34d1b..cb0b8ae). Bit-identity evidence for the Pass 3a migration (robust_affine / motion_batched onto the central pyramid).

Records the outputs of the affine and motion callers on fixed inputs, on CPU, and compares two recordings byte for byte.
Run it once with ``PYTHONPATH=<old worktree>/src`` (the commit before the migration) and once with the migrated tree::

    PYTHONPATH=$OLD/src python benchmarks/pyramid_migration_identity.py record OLD_DIR
    PYTHONPATH=src      python benchmarks/pyramid_migration_identity.py record NEW_DIR
    python benchmarks/pyramid_migration_identity.py compare OLD_DIR NEW_DIR

Cases: robust_affine (rigid) on the thin-slab pairs of tests/test_robust_affine_anisotropy.py::_slab_pair at 1:1, 2:1, 4:1, 8:1;
robust_affine with smoothing and an affine dof on an isotropic pair; the phase-correlation candidates directly (isotropic and
anisotropic fixed); the real neuromelanin pairs sub-01 and sub-07; motion_batched (single-group, adaptive, group-bias). A recording
is a directory of ``<case>.npz`` files holding the returned transform parameters (float64) and a hash of the raw bytes.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

REAL = "/Users/stnava/data/nm_registration_debug/subjects"
SEED = 42


def _tx_arrays(paths):
    import ants
    out = {}
    for i, p in enumerate(paths if isinstance(paths, (list, tuple)) else [paths]):
        if str(p).endswith(".mat"):
            tx = ants.read_transform(p)
            out[f"p{i}"] = np.asarray(tx.parameters, dtype=np.float64)
            out[f"f{i}"] = np.asarray(tx.fixed_parameters, dtype=np.float64)
            out[f"bytes{i}"] = np.frombuffer(hashlib.sha256(open(p, "rb").read()).digest(), dtype=np.uint8)
    return out


def _phantom_pair(shape=(40, 44, 36), spacing=(1.0, 1.0, 1.0), seed=3):
    import ants
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.random(shape).astype(np.float32), 3.0)
    a = (a - a.min()) / (a.max() - a.min())
    fixed = ants.from_numpy(a, spacing=spacing, origin=(-20.0, -22.0, -18.0))
    mov = ants.from_numpy(np.roll(a, (2, -1, 1), axis=(0, 1, 2)).copy(), spacing=spacing, origin=(-20.0, -22.0, -18.0))
    return fixed, mov


def cases():
    import ants
    import syntx
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    from test_robust_affine_anisotropy import _slab_pair
    RA = __import__("syntx.robust_affine", fromlist=["x"])
    for r in (1, 2, 4, 8):
        def run(r=r):
            fixed, mov, _ = _slab_pair(r)
            reg = syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", seed=SEED)
            return _tx_arrays(reg["fwdtransforms"])
        yield f"slab_{r}to1_rigid", run
    def run_iso_aff():
        fixed, mov = _phantom_pair()
        reg = syntx.robust_affine(fixed, mov, mode="pytorch", dof="affine", device="cpu", seed=SEED, smooth_sigma_per_level=0.5)
        return _tx_arrays(reg["fwdtransforms"])
    yield "iso_affine_smoothed", run_iso_aff
    def run_iso_rigid_nosmooth():
        fixed, mov = _phantom_pair((48, 52, 44))
        reg = syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", seed=SEED)
        return _tx_arrays(reg["fwdtransforms"])
    yield "iso_rigid", run_iso_rigid_nosmooth
    def run_pc_iso():
        fixed, mov = _phantom_pair((96, 100, 88))
        return {f"c{i}": np.asarray(c, dtype=np.float64) for i, c in enumerate(RA._phase_correlation_translation_candidates(fixed, mov))}
    yield "phase_corr_iso", run_pc_iso
    def run_pc_iso_moving_other_grid():
        fixed, _ = _phantom_pair((96, 100, 88))
        _, mov = _phantom_pair((60, 64, 56), spacing=(1.5, 1.5, 1.5), seed=5)
        return {f"c{i}": np.asarray(c, dtype=np.float64) for i, c in enumerate(RA._phase_correlation_translation_candidates(fixed, mov))}
    yield "phase_corr_iso_other_grid", run_pc_iso_moving_other_grid
    for r in (2, 4, 8):
        def run_pc_an(r=r):
            fixed, mov, _ = _slab_pair(r)
            return {f"c{i}": np.asarray(c, dtype=np.float64) for i, c in enumerate(RA._phase_correlation_translation_candidates(fixed, mov))}
        yield f"phase_corr_aniso_{r}to1", run_pc_an
    for sub in ("sub-01", "sub-07"):
        def run_real(sub=sub):
            fixed = ants.image_read(f"{REAL}/{sub}/fixed_nm_n4.nii.gz")
            mov = ants.image_read(f"{REAL}/{sub}/moving_t1_slab_crop.nii.gz")
            reg = syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", seed=SEED)
            return _tx_arrays(reg["fwdtransforms"])
        yield f"real_{sub}_rigid", run_real
        def run_real_pc(sub=sub):
            fixed = ants.image_read(f"{REAL}/{sub}/fixed_nm_n4.nii.gz")
            mov = ants.image_read(f"{REAL}/{sub}/moving_t1_slab_crop.nii.gz")
            return {f"c{i}": np.asarray(c, dtype=np.float64) for i, c in enumerate(RA._phase_correlation_translation_candidates(fixed, mov))}
        yield f"real_{sub}_phase_corr", run_real_pc
    # motion_batched
    from test_motion_batched import _create_3d_phantom
    from syntx.motion_batched import (batched_rigid_register_pass, batched_rigid_register_pass_adaptive,
                                      batched_group_bias_register_pass)
    def frames(n=4, size=48, sp=(1.0, 1.0, 1.0)):
        img, _ = _create_3d_phantom(num_frames=n, nx=size, ny=size, nz=size,
                                    shifts=[(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0), (0.5, 0.3, -0.4)][:n])
        fr = ants.ndimage_to_list(img)
        return [ants.from_numpy(f.numpy(), spacing=sp) for f in fr]
    def mb_pass(**kw):
        fr = frames()
        fwd, inv, *_ = batched_rigid_register_pass(fr[0], fr[1:], device="cpu", **kw)[:3]
        return {f"{i}_{k}": v for i, f in enumerate(fwd) for k, v in _tx_arrays(f).items()}
    yield "motion_batched_wide", lambda: mb_pass()
    def mb_aniso():
        fr = frames(sp=(1.0, 1.0, 1.0))
        fr = [ants.from_numpy(f.numpy(), spacing=(0.8, 0.8, 1.6)) for f in fr]
        fwd, inv, *_ = batched_rigid_register_pass(fr[0], fr[1:], device="cpu")[:3]
        return {f"{i}_{k}": v for i, f in enumerate(fwd) for k, v in _tx_arrays(f).items()}
    yield "motion_batched_aniso", mb_aniso
    def mb_adaptive():
        fr = frames()
        res = batched_rigid_register_pass_adaptive(fr[0], fr[1:], device="cpu")
        fwd = res[0]
        return {f"{i}_{k}": v for i, f in enumerate(fwd) for k, v in _tx_arrays(f).items()}
    yield "motion_batched_adaptive", mb_adaptive
    def mb_group():
        fr = frames()
        fwd, inv, R, t, el = batched_group_bias_register_pass(fr[0], fr[1:], [False, True, True], device="cpu")
        d = {f"{i}_{k}": v for i, f in enumerate(fwd) for k, v in _tx_arrays(f).items()}
        d["R"] = np.asarray(R, dtype=np.float64); d["t"] = np.asarray(t, dtype=np.float64)
        return d
    yield "motion_batched_group_bias", mb_group


def record(outdir, only=None):
    os.makedirs(outdir, exist_ok=True)
    import syntx
    print("syntx from", syntx.__file__, flush=True)
    for name, fn in cases():
        if only and name not in only:
            continue
        if os.path.exists(os.path.join(outdir, name + ".npz")):
            continue
        t = time.time()
        res = fn()
        np.savez(os.path.join(outdir, name + ".npz"), **res)
        print(f"recorded {name} ({time.time() - t:.1f} s, {len(res)} arrays)", flush=True)


def compare(a, b):
    names = sorted(f[:-4] for f in os.listdir(a) if f.endswith(".npz"))
    bad = 0
    for n in names:
        pb = os.path.join(b, n + ".npz")
        if not os.path.exists(pb):
            print(f"MISSING {n}"); bad += 1; continue
        x, y = np.load(os.path.join(a, n + ".npz")), np.load(pb)
        same = sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        mx = max((float(np.max(np.abs(x[k] - y[k]))) for k in x.files if k in y.files and x[k].shape == y[k].shape
                  and x[k].dtype.kind == "f"), default=float("nan"))
        print(f"{'IDENTICAL' if same else 'DIFFERENT'} {n} (max abs diff {mx:.3e})")
        bad += 0 if same else 1
    print("ALL IDENTICAL" if bad == 0 else f"{bad} case(s) differ")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3:] or None)
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
