#!/usr/bin/env python3
import os
import numpy as np
import ants

p44 = "results/anatomical_evidence/pair_044"
fl_aseg = ants.image_read(os.path.join(p44, "fixed_label_aseg.nii.gz"))
fl_lpi = ants.reorient_image2(fl_aseg, "LPI")
aseg_lpi = fl_lpi.numpy()

mask = np.isin(aseg_lpi, [4, 14, 43])
print("Ventricles in LPI coordinates:")
print("  Best Sagittal (axis 0):", np.argmax([mask[x, :, :].sum() for x in range(mask.shape[0])]))
print("  Best Coronal  (axis 1):", np.argmax([mask[:, y, :].sum() for y in range(mask.shape[1])]))
print("  Best Axial    (axis 2):", np.argmax([mask[:, :, z].sum() for z in range(mask.shape[2])]))

