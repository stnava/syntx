#!/usr/bin/env python3
import os
import numpy as np
import ants

pair_dir_44 = "results/anatomical_evidence/pair_044"
fl_dkt_44 = ants.image_read(os.path.join(pair_dir_44, "fixed_label_dkt.nii.gz"))
fl_aseg_44 = ants.image_read(os.path.join(pair_dir_44, "fixed_label_aseg.nii.gz"))
dkt_arr_44 = fl_dkt_44.numpy()
aseg_arr_44 = fl_aseg_44.numpy()

labels_to_check = {
    "Middle Temporal (1015/2015)": [1015, 2015],
    "Superior Temporal (1030/2030)": [1030, 2030],
    "Caudal Anterior Cingulate (1002/2002)": [1002, 2002],
    "Supramarginal (1031/2031)": [1031, 2031],
    "Ventricles (4/14/43)": [4, 14, 43],
}

print("=== Pair 44 Gainers Coordinates ===")
for name, lbls in labels_to_check.items():
    arr = aseg_arr_44 if "Ventricles" in name else dkt_arr_44
    mask = np.isin(arr, lbls)
    if mask.sum() > 0:
        coords = np.where(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        z_min, z_max = coords[2].min(), coords[2].max()
        cx, cy, cz = int(np.mean(coords[0])), int(np.mean(coords[1])), int(np.mean(coords[2]))
        print(f"{name}:")
        print(f"  Shape: {mask.sum()} voxels")
        print(f"  Center (x, y, z): ({cx}, {cy}, {cz})")
        print(f"  Bounding Box: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}], Z=[{z_min}, {z_max}]")

