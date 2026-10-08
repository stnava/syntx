#!/usr/bin/env python3
import os
import numpy as np
import ants

pair_dir_08 = "results/anatomical_evidence/pair_008"
fl_dkt_08 = ants.image_read(os.path.join(pair_dir_08, "fixed_label_dkt.nii.gz"))
fl_aseg_08 = ants.image_read(os.path.join(pair_dir_08, "fixed_label_aseg.nii.gz"))
dkt_arr_08 = fl_dkt_08.numpy()
aseg_arr_08 = fl_aseg_08.numpy()

labels_to_check = {
    "Caudal Anterior Cingulate (1002/2002)": [1002, 2002],
    "3rd Ventricle (14)": [14],
    "Precuneus (1025/2025)": [1025, 2025],
    "Precentral (1024/2024)": [1024, 2024],
}

print("=== Pair 08 Gainers Coordinates ===")
for name, lbls in labels_to_check.items():
    arr = aseg_arr_08 if "Ventricle" in name else dkt_arr_08
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

