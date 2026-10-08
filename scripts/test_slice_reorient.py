#!/usr/bin/env python3
import os
import numpy as np
import ants
from syntx.viz import extract_oriented_slice

p44 = "results/anatomical_evidence/pair_044"
fi = ants.image_read(os.path.join(p44, "fixed_target.nii.gz"))
fl_dkt = ants.image_read(os.path.join(p44, "fixed_label_dkt.nii.gz"))

# Prepare reoriented LPI image to find optimal slices
fi_lpi = ants.reorient_image2(fi, "LPI")
fl_lpi = ants.reorient_image2(fl_dkt, "LPI")
dkt_lpi_arr = fl_lpi.numpy()

for name, lbls in [("MTG", [1015, 2015]), ("STG", [1030, 2030]), ("CAC", [1002, 2002])]:
    # LPI planes: 0=sagittal, 1=coronal, 2=axial
    mask = np.isin(dkt_lpi_arr, lbls)
    sag_sum = [mask[x, :, :].sum() for x in range(mask.shape[0])]
    cor_sum = [mask[:, y, :].sum() for y in range(mask.shape[1])]
    ax_sum = [mask[:, :, z].sum() for z in range(mask.shape[2])]
    print(f"{name} in LPI coordinates:")
    print(f"  Best Sagittal (axis 0): {np.argmax(sag_sum)} (voxels={np.max(sag_sum)})")
    print(f"  Best Coronal  (axis 1): {np.argmax(cor_sum)} (voxels={np.max(cor_sum)})")
    print(f"  Best Axial    (axis 2): {np.argmax(ax_sum)} (voxels={np.max(ax_sum)})")

