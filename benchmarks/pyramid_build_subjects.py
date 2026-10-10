"""Build the T1-slab -> neuromelanin (NM) registration pairs for ds001242 subjects, as antsxfunctional's run_melanin does.

Per subject, under ``<out>/sub-XX/``: ``fixed_nm_n4.nii.gz`` (N4-corrected NM), ``moving_t1_slab_crop.nii.gz`` (whole-head T1
cropped to the midbrain slab and min-max normalised) and ``ants_rigid_reference.mat`` (plain ANTs rigid Mattes registration of the
pair, random_seed=1). Needs antsxfunctional and antstorch (the recipe of /Users/stnava/data/nm_registration_debug/).

    python benchmarks/pyramid_build_subjects.py --subjects 02 03 ... --out /Users/stnava/data/nm_registration_debug/subjects
"""
import argparse
import os
import shutil
import time

import ants

DATA = "/Users/stnava/data/repos/antsxfunctional/data_cache/ds001242"


def build(sub: str, out: str, device: str) -> None:
    import antstorch
    from antsxfunctional.melanin.atlas import compute_midbrain_priors
    d = os.path.join(DATA, f"sub-{sub}", "anat")
    o = os.path.join(out, f"sub-{sub}")
    os.makedirs(o, exist_ok=True)
    t1_head = ants.image_read(os.path.join(d, "T1w.nii.gz"))
    nm = ants.image_read(os.path.join(d, "melanin_T1w.nii.gz"))
    t1_brain = t1_head * ants.threshold_image(antstorch.brain_extraction(t1_head, "t1", verbose=False), 0.5, 1.0)
    pri = compute_midbrain_priors(t1_head, t1_brain=t1_brain, device=device)
    n4m = ants.threshold_image(ants.iMath(nm, "Normalize"), 0.05, 1.0)
    fixed = ants.n4_bias_field_correction(nm, mask=n4m)
    moving = ants.crop_image(t1_head, pri.slab_mask_t1).iMath("Normalize")
    ants.image_write(fixed, os.path.join(o, "fixed_nm_n4.nii.gz"))
    ants.image_write(moving, os.path.join(o, "moving_t1_slab_crop.nii.gz"))
    t = time.time()
    r = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid", aff_metric="mattes", random_seed=1)
    shutil.copy(r["fwdtransforms"][0], os.path.join(o, "ants_rigid_reference.mat"))
    print(f"sub-{sub}: fixed {fixed.shape} {tuple(round(x, 3) for x in fixed.spacing)}, moving {moving.shape}, "
          f"ANTs rigid {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    for s in a.subjects:
        build(s, a.out, a.device)
