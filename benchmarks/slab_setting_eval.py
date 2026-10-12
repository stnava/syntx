"""Evaluation of ``robust_affine(setting='intermodality_slab')`` against the default (docs/RELEASE_NOTES_6.3.0.md).

Sub-commands (JSON lines under --out, resumable; every record carries the device and the wall time):
    real       the ten real ds001242 NM-slab / T1 pairs: pose error (max probe-point displacement vs the ANTs rigid reference) and native MI
    synthetic  thin-slab known-rigid sweep (benchmarks/pyramid_compare.slab_pair, pose A by default): mean TRE per ratio
    stress     capture stress on the real pairs. The centre of mass (the start of every default candidate) is displaced by a prescribed
               distance along a physical axis by CROPPING the T1 at one end (the pose to find is unchanged; pyramid_compare's origin
               shift would move the truth and the centre of mass together and leave the start error unchanged: ``--control`` runs that)
Run with PYTHONPATH=src. Pose error and TRE are in mm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import syntx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyramid_compare as PC  # noqa: E402

SUBJ = "/Users/stnava/data/nm_registration_debug/subjects"
SETTINGS = ["default", "intermodality_slab", "zshift"]   # zshift: slab_capture strategy="z_shift" (default solver + 8 z-shifted starts)
AXES = {"x": 0, "y": 1, "z": 2}


def _call(fixed, moving, setting, device, seed, **kw):
    import syntx
    t = time.time()
    try:
        if setting == "zshift":
            from syntx.slab_capture import run_slab_capture
            return run_slab_capture(fixed, moving, device=device, seed=seed, strategy="z_shift", **kw), time.time() - t, None
        reg = syntx.robust_affine(fixed, moving, mode="auto", dof="rigid", device=device, seed=seed, setting=setting, **kw)
        return reg, time.time() - t, None
    except Exception as e:                     # a failure is a measured outcome, recorded verbatim
        return None, time.time() - t, f"{type(e).__name__}: {str(e)[:200]}"


def _com(img):
    from syntx.core.utils import normalize_image
    from syntx.robust_affine import robust_center_of_mass
    p, n = robust_center_of_mass(normalize_image(img, method="auto", force=True), weighted=True)
    return np.asarray(p if p is not None else n, dtype=np.float64)


def crop_to_shift_com(moving, axis: str, d_mm: float):
    """Crop ``moving`` at its low-physical end along the physical axis ``axis`` until the centre of mass has moved by ~``d_mm`` in +axis.
    Returns (cropped image, achieved shift in mm, voxels removed) or (None, 0, 0) when no crop of up to 70 percent reaches 0.5 d."""
    import ants
    a = AXES[axis]
    Dm = np.asarray(moving.direction).reshape(3, 3)
    k = int(np.argmax(np.abs(Dm[a, :])))
    sgn = np.sign(Dm[a, k])
    n = moving.shape[k]
    c0 = _com(moving)[a]
    best = (None, 0.0, 0)
    for cut in range(0, int(0.7 * n)):
        lo = [0, 0, 0]; hi = [moving.shape[0], moving.shape[1], moving.shape[2]]
        if sgn > 0:
            lo[k] = cut
        else:
            hi[k] = n - cut
        img = ants.crop_indices(moving, lo, hi) if cut else moving
        sh = _com(img)[a] - c0
        best = (img, float(sh), cut)
        if sh >= d_mm:
            break
    if best[1] < 0.5 * d_mm:
        return None, best[1], best[2]
    return best


def cmd_real(a):
    import ants
    out = os.path.join(a.out, "real.jsonl"); done = PC._done_keys(out)
    for sub in a.subjects:
        d = os.path.join(SUBJ, sub)
        fixed = ants.image_read(f"{d}/fixed_nm_n4.nii.gz"); moving = ants.image_read(f"{d}/moving_t1_slab_crop.nii.gz")
        ref = f"{d}/ants_rigid_reference.mat"
        for st in a.settings:
            key = f"{sub}|{st}|{a.device}"
            if key in done:
                continue
            reg, dt, err = _call(fixed, moving, st, a.device, a.seed)
            rec = dict(key=key, subject=sub, setting=st, device=a.device, time=dt, error=err)
            if reg is not None:
                rep = syntx.pose_error_report(fixed, reg["fwdtransforms"][0], ref)
                rec["pose_err_max"], rec["pose_err_z"] = rep.max_error_mm, rep.z_mean_error_mm
                rec["mi_native_ants_yardstick"] = PC.native_mi(fixed, moving, reg["fwdtransforms"][0])
                rec["mi_native_reference"] = PC.native_mi(fixed, moving, ref)
                if "slab_capture" in reg:
                    rec["winner"] = reg["slab_capture"]["winner"]; rec["n_rejected"] = len(reg["slab_capture"]["rejected"])
            PC._append(out, rec); print(json.dumps(rec), flush=True)


def cmd_synthetic(a):
    out = os.path.join(a.out, "synthetic.jsonl"); done = PC._done_keys(out)
    for r in a.ratios:
        for pose in a.poses:
            fixed, mov, tx = PC.slab_pair(r, pose)
            for st in a.settings:
                key = f"{r}|{pose}|{st}|{a.seed}|{a.device}"
                if key in done:
                    continue
                reg, dt, err = _call(fixed, mov, st, a.device, a.seed)
                rec = dict(key=key, ratio=r, pose=pose, setting=st, seed=a.seed, device=a.device, time=dt, error=err,
                           tre=None if reg is None else PC.tre(reg, tx))
                PC._append(out, rec); print(json.dumps(rec), flush=True)


def cmd_stress(a):
    import ants
    out = os.path.join(a.out, "stress.jsonl"); done = PC._done_keys(out)
    for sub in a.subjects:
        d = os.path.join(SUBJ, sub)
        fixed = ants.image_read(f"{d}/fixed_nm_n4.nii.gz"); moving0 = ants.image_read(f"{d}/moving_t1_slab_crop.nii.gz")
        ref = f"{d}/ants_rigid_reference.mat"
        for axis in a.axes:
            for dist in a.dists:
                if a.control:
                    o = list(moving0.origin); o[AXES[axis]] += dist
                    mov = ants.image_clone(moving0); mov.set_origin(tuple(o)); got, cut = dist, 0
                    # truth moves with the shifted image: y' = y + shift
                    shift = np.zeros(3); shift[AXES[axis]] = dist
                else:
                    mov, got, cut = crop_to_shift_com(moving0, axis, dist); shift = np.zeros(3)
                for st in a.settings:
                    key = f"{sub}|{axis}|{dist}|{st}|{'ctl' if a.control else 'crop'}|{a.device}"
                    if key in done:
                        continue
                    rec = dict(key=key, subject=sub, axis=axis, requested_mm=dist, com_shift_mm=got, vox_cut=cut, setting=st, device=a.device,
                               control=a.control)
                    if mov is None:
                        rec["error"] = "cannot displace the centre of mass by this much by cropping"
                    else:
                        reg, dt, err = _call(fixed, mov, st, a.device, a.seed)
                        rec.update(time=dt, error=err)
                        if reg is not None:
                            e = syntx.probe_displacements(fixed, reg["fwdtransforms"][0]) - (syntx.probe_displacements(fixed, ref) + shift)
                            rec["pose_err_max"] = float(np.linalg.norm(e, axis=1).max())
                    PC._append(out, rec); print(json.dumps(rec), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--out", required=True); p.add_argument("--device", default="mps"); p.add_argument("--seed", type=int, default=42)
        p.add_argument("--settings", nargs="+", default=SETTINGS, choices=SETTINGS)
    p = sub.add_parser("real"); common(p); p.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)]); p.set_defaults(fn=cmd_real)
    p = sub.add_parser("synthetic"); common(p)
    p.add_argument("--ratios", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10]); p.add_argument("--poses", nargs="+", default=["A"])
    p.set_defaults(fn=cmd_synthetic)
    p = sub.add_parser("stress"); common(p); p.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    p.add_argument("--axes", nargs="+", default=["z", "x", "y"], choices=list(AXES)); p.add_argument("--dists", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    p.add_argument("--control", action="store_true"); p.set_defaults(fn=cmd_stress)
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True); a.fn(a)


if __name__ == "__main__":
    main()
