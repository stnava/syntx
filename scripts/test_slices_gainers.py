#!/usr/bin/env python3
import os
import numpy as np
import ants
from syntx.viz import extract_oriented_slice

p44 = "results/anatomical_evidence/pair_044"
fi = ants.image_read(os.path.join(p44, "fixed_target.nii.gz"))
fl_dkt = ants.image_read(os.path.join(p44, "fixed_label_dkt.nii.gz"))
fl_aseg = ants.image_read(os.path.join(p44, "fixed_label_aseg.nii.gz"))

dkt_arr = fl_dkt.numpy()
aseg_arr = fl_aseg.numpy()

# 1. MTG (1015/2015)
mtg_mask = np.isin(dkt_arr, [1015, 2015])
# Find slices with maximal MTG area
cor_counts_mtg = [np.isin(dkt_arr[:, y, :], [1015, 2015]).sum() for y in range(dkt_arr.shape[1])]
ax_counts_mtg = [np.isin(dkt_arr[:, :, z], [1015, 2015]).sum() for z in range(dkt_arr.shape[2])]
best_cor_mtg = int(np.argmax(cor_counts_mtg))
best_ax_mtg = int(np.argmax(ax_counts_mtg))

# 2. STG (1030/2030)
cor_counts_stg = [np.isin(dkt_arr[:, y, :], [1030, 2030]).sum() for y in range(dkt_arr.shape[1])]
ax_counts_stg = [np.isin(dkt_arr[:, :, z], [1030, 2030]).sum() for z in range(dkt_arr.shape[2])]
best_cor_stg = int(np.argmax(cor_counts_stg))
best_ax_stg = int(np.argmax(ax_counts_stg))

# 3. Cingulate (1002/2002)
sag_counts_cac = [np.isin(dkt_arr[x, :, :], [1002, 2002]).sum() for x in range(dkt_arr.shape[0])]
cor_counts_cac = [np.isin(dkt_arr[:, y, :], [1002, 2002]).sum() for y in range(dkt_arr.shape[1])]
best_sag_cac = int(np.argmax(sag_counts_cac))
best_cor_cac = int(np.argmax(cor_counts_cac))

# 4. Ventricles (4/14/43)
cor_counts_vent = [np.isin(aseg_arr[:, y, :], [4, 14, 43]).sum() for y in range(aseg_arr.shape[1])]
ax_counts_vent = [np.isin(aseg_arr[:, :, z], [4, 14, 43]).sum() for z in range(aseg_arr.shape[2])]
best_cor_vent = int(np.argmax(cor_counts_vent))
best_ax_vent = int(np.argmax(ax_counts_vent))

print(f"MTG Peak Area: Coronal Y={best_cor_mtg} (count={cor_counts_mtg[best_cor_mtg]}), Axial Z={best_ax_mtg} (count={ax_counts_mtg[best_ax_mtg]})")
print(f"STG Peak Area: Coronal Y={best_cor_stg} (count={cor_counts_stg[best_cor_stg]}), Axial Z={best_ax_stg} (count={ax_counts_stg[best_ax_stg]})")
print(f"CAC Peak Area: Sagittal X={best_sag_cac} (count={sag_counts_cac[best_sag_cac]}), Coronal Y={best_cor_cac} (count={cor_counts_cac[best_cor_cac]})")
print(f"Vent Peak Area: Coronal Y={best_cor_vent} (count={cor_counts_vent[best_cor_vent]}), Axial Z={best_ax_vent} (count={ax_counts_vent[best_ax_vent]})")

