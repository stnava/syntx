#!/usr/bin/env python3
import os
import ants

def warp_labels_for_pair(pair_dir, aff_mat):
    print(f"Warping labels for {pair_dir}...", flush=True)
    fi = ants.image_read(os.path.join(pair_dir, "fixed_target.nii.gz"))
    ml_dkt = ants.image_read(os.path.join(pair_dir, "moving_label_dkt.nii.gz"))
    ml_aseg = ants.image_read(os.path.join(pair_dir, "moving_label_aseg.nii.gz"))

    # Affine Warped Labels (Original label mapped to target anatomy)
    print("  -> Warping Affine labels...", flush=True)
    aff_dkt = ants.apply_transforms(fi, ml_dkt, [aff_mat], interpolator="genericLabel")
    ants.image_write(aff_dkt, os.path.join(pair_dir, "warped_dkt_affine.nii.gz"))
    aff_aseg = ants.apply_transforms(fi, ml_aseg, [aff_mat], interpolator="genericLabel")
    ants.image_write(aff_aseg, os.path.join(pair_dir, "warped_aseg_affine.nii.gz"))

    # Deformable Warped Labels
    models = ["sobolev", "syn_hyperelastic", "syn_divcurl"]
    for m in models:
        warp_file = os.path.join(pair_dir, f"warp_{m}.nii.gz")
        tx_list = [warp_file, aff_mat]
        print(f"  -> Warping [{m}] DKT labels...", flush=True)
        w_dkt = ants.apply_transforms(fi, ml_dkt, tx_list, interpolator="genericLabel")
        ants.image_write(w_dkt, os.path.join(pair_dir, f"warped_dkt_{m}.nii.gz"))

if __name__ == "__main__":
    warp_labels_for_pair("results/anatomical_evidence/pair_044", "results/canonical_affines/pair_044_pt7_affine.mat")
    warp_labels_for_pair("results/anatomical_evidence/pair_008", "results/canonical_affines/pair_008_pt7_affine.mat")
    print("All labels successfully transformed and saved!", flush=True)
