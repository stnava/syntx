#!/usr/bin/env python3
"""
Generate publication-quality figures visualizing the "Biggest Gainers" in Continuum Mechanics
using the formalized primitives in syntx.viz:
- visualize_segmentation_on_anatomy
- visualize_segmentation_pair_on_anatomy
- visualize_flow_on_anatomy
- visualize_flow_differential_on_anatomy
- render_gainer_anatomical_dissection
- AnatomicalOverlayVisualizer

Strictly adheres to GEMINI.md Rules 1-6 (light theme, physical space, syntx.viz).
"""

import os
import numpy as np
import ants

import syntx.viz as viz
from syntx.viz import (
    extract_oriented_slice,
    render_gainer_anatomical_dissection,
    AnatomicalOverlayVisualizer,
)

FIG_DIR = "docs/reports/figures_cm_evidence"
os.makedirs(FIG_DIR, exist_ok=True)
p44 = "results/anatomical_evidence/pair_044"
p08 = "results/anatomical_evidence/pair_008"

print("Loading Pair 44 images and warps via syntx.viz pipeline...", flush=True)
fi_44 = ants.image_read(os.path.join(p44, "fixed_target.nii.gz"))
fl_dkt_44 = ants.image_read(os.path.join(p44, "fixed_label_dkt.nii.gz"))
fl_aseg_44 = ants.image_read(os.path.join(p44, "fixed_label_aseg.nii.gz"))

aff_dkt_44 = ants.image_read(os.path.join(p44, "warped_dkt_affine.nii.gz"))
sob_dkt_44 = ants.image_read(os.path.join(p44, "warped_dkt_sobolev.nii.gz"))
hyp_dkt_44 = ants.image_read(os.path.join(p44, "warped_dkt_syn_hyperelastic.nii.gz"))

aff_aseg_44 = ants.image_read(os.path.join(p44, "warped_aseg_affine.nii.gz"))
sob_aseg_44 = ants.image_read(os.path.join(p44, "warped_aseg_sobolev.nii.gz"))
hyp_aseg_44 = ants.image_read(os.path.join(p44, "warped_aseg_syn_hyperelastic.nii.gz"))

warp_sob_44 = os.path.join(p44, "warp_sobolev.nii.gz")
warp_hyp_44 = os.path.join(p44, "warp_syn_hyperelastic.nii.gz")

# =========================================================================
# 1. FIGURE 7: Middle Temporal Gyrus (#1 Gainer: +6.93% Dice Gain)
# =========================================================================
print("\n--- Generating Figure 7 via render_gainer_anatomical_dissection (MTG) ---", flush=True)
z_mtg = 133
fl_dkt_slice, _ = extract_oriented_slice(fl_dkt_44, slice_axis=2, slice_idx=z_mtg)
mtg_left = (fl_dkt_slice == 1015)
coords = np.where(mtg_left)
ymin, ymax = max(0, coords[0].min() - 15), min(fl_dkt_slice.shape[0], coords[0].max() + 15)
xmin, xmax = max(0, coords[1].min() - 15), min(fl_dkt_slice.shape[1], coords[1].max() + 15)
roi_mtg = (ymin, ymax, xmin, xmax)

models_cfg_mtg = {
    'affine': {'image': aff_dkt_44, 'badge': 'Initial Dice: 0.312', 'color': '#94A3B8', 'edge': '#475569'},
    'sobolev': {'image': sob_dkt_44, 'badge': 'Sobolev Dice: 0.5443', 'color': '#0284C7', 'edge': '#0369A1'},
    'continuum': {'image': hyp_dkt_44, 'badge': 'Hyperelastic Dice: 0.6136 (+6.93%)', 'color': '#059669', 'edge': '#047857'},
}

fig7_path = os.path.join(FIG_DIR, "fig7_mtg_biggest_gainer_flows.png")
render_gainer_anatomical_dissection(
    anatomy=fi_44,
    target_label=fl_dkt_44,
    models_config=models_cfg_mtg,
    flow_sobolev=warp_sob_44,
    flow_continuum=warp_hyp_44,
    slice_axis=2,
    slice_idx=z_mtg,
    target_label_ids=[1015, 2015],
    roi=roi_mtg,
    structure_name="Middle Temporal Gyrus",
    gain_pct=6.93,
    sobolev_name="Sobolev SyN",
    continuum_name="Hyperelastic SyN",
    flow_vmax=15.0,
    delta_vmax=6.0,
    output_filename=fig7_path
)
print(f"--> Successfully generated Figure 7: {fig7_path}", flush=True)


# =========================================================================
# 2. FIGURE 8: Caudal Anterior Cingulate Gyrus (+4.37% Dice Gain)
# =========================================================================
print("\n--- Generating Figure 8 via render_gainer_anatomical_dissection (Cingulate) ---", flush=True)
x_cac = 98
fl_dkt_sag, _ = extract_oriented_slice(fl_dkt_44, slice_axis=0, slice_idx=x_cac)
cac_target = np.isin(fl_dkt_sag, [1002, 2002])
coords_cac = np.where(cac_target)
ymin_c, ymax_c = max(0, coords_cac[0].min() - 25), min(fl_dkt_sag.shape[0], coords_cac[0].max() + 25)
xmin_c, xmax_c = max(0, coords_cac[1].min() - 25), min(fl_dkt_sag.shape[1], coords_cac[1].max() + 25)
roi_cac = (ymin_c, ymax_c, xmin_c, xmax_c)

models_cfg_cac = {
    'affine': {'image': aff_dkt_44, 'badge': 'Initial Dice: 0.284', 'color': '#94A3B8', 'edge': '#475569'},
    'sobolev': {'image': sob_dkt_44, 'badge': 'Sobolev Dice: 0.5349', 'color': '#0284C7', 'edge': '#0369A1'},
    'continuum': {'image': hyp_dkt_44, 'badge': 'Hyperelastic Dice: 0.5786 (+4.37%)', 'color': '#059669', 'edge': '#047857'},
}

fig8_path = os.path.join(FIG_DIR, "fig8_cingulate_gainer_flows.png")
render_gainer_anatomical_dissection(
    anatomy=fi_44,
    target_label=fl_dkt_44,
    models_config=models_cfg_cac,
    flow_sobolev=warp_sob_44,
    flow_continuum=warp_hyp_44,
    slice_axis=0,
    slice_idx=x_cac,
    target_label_ids=[1002, 2002],
    roi=roi_cac,
    structure_name="Caudal Anterior Cingulate Gyrus",
    gain_pct=4.37,
    sobolev_name="Sobolev SyN",
    continuum_name="Hyperelastic SyN",
    flow_vmax=12.0,
    delta_vmax=4.0,
    output_filename=fig8_path
)
print(f"--> Successfully generated Figure 8: {fig8_path}", flush=True)


# =========================================================================
# 3. FIGURE 9: Ventricular System (+89% Volume Shift & Compressive Flows)
# =========================================================================
print("\n--- Generating Figure 9 via render_gainer_anatomical_dissection (Ventricles) ---", flush=True)
y_vent = 89
fl_aseg_cor, _ = extract_oriented_slice(fl_aseg_44, slice_axis=1, slice_idx=y_vent)
vent_target = np.isin(fl_aseg_cor, [4, 14, 43])
coords_vent = np.where(vent_target)
ymin_v, ymax_v = max(0, coords_vent[0].min() - 25), min(fl_aseg_cor.shape[0], coords_vent[0].max() + 25)
xmin_v, xmax_v = max(0, coords_vent[1].min() - 35), min(fl_aseg_cor.shape[1], coords_vent[1].max() + 35)
roi_vent = (ymin_v, ymax_v, xmin_v, xmax_v)

models_cfg_vent = {
    'affine': {'image': aff_aseg_44, 'badge': 'Source Ventricle: 22.1 mL', 'color': '#EF4444', 'edge': '#B91C1C'},
    'sobolev': {'image': sob_aseg_44, 'badge': 'Sobolev J_min = 0.448', 'color': '#0284C7', 'edge': '#0369A1'},
    'continuum': {'image': hyp_aseg_44, 'badge': 'Hyperelastic J_min = 0.318', 'color': '#059669', 'edge': '#047857'},
}

fig9_path = os.path.join(FIG_DIR, "fig9_ventricles_gainer_flows.png")
render_gainer_anatomical_dissection(
    anatomy=fi_44,
    target_label=fl_aseg_44,
    models_config=models_cfg_vent,
    flow_sobolev=warp_sob_44,
    flow_continuum=warp_hyp_44,
    slice_axis=1,
    slice_idx=y_vent,
    target_label_ids=[4, 14, 43],
    roi=roi_vent,
    structure_name="Ventricular System (+89% Volume Shift)",
    gain_pct=89.2,
    sobolev_name="Sobolev SyN",
    continuum_name="Hyperelastic SyN",
    flow_vmax=18.0,
    delta_vmax=6.0,
    output_filename=fig9_path
)
print(f"--> Successfully generated Figure 9: {fig9_path}", flush=True)


# =========================================================================
# 4. FIGURE 10: Superior Temporal Gyrus (#2 Gainer: +5.16% Dice Gain)
# =========================================================================
print("\n--- Generating Figure 10 via render_gainer_anatomical_dissection (STG) ---", flush=True)
z_stg = 148
fl_dkt_ax_stg, _ = extract_oriented_slice(fl_dkt_44, slice_axis=2, slice_idx=z_stg)
stg_mask = (fl_dkt_ax_stg == 1030)  # Left STG
coords_stg = np.where(stg_mask)
ymin_s, ymax_s = max(0, coords_stg[0].min() - 20), min(fl_dkt_ax_stg.shape[0], coords_stg[0].max() + 20)
xmin_s, xmax_s = max(0, coords_stg[1].min() - 20), min(fl_dkt_ax_stg.shape[1], coords_stg[1].max() + 20)
roi_stg = (ymin_s, ymax_s, xmin_s, xmax_s)

models_cfg_stg = {
    'affine': {'image': aff_dkt_44, 'badge': 'Initial Dice: 0.441', 'color': '#94A3B8', 'edge': '#475569'},
    'sobolev': {'image': sob_dkt_44, 'badge': 'Sobolev Dice: 0.6810', 'color': '#0284C7', 'edge': '#0369A1'},
    'continuum': {'image': hyp_dkt_44, 'badge': 'Hyperelastic Dice: 0.7326 (+5.16%)', 'color': '#059669', 'edge': '#047857'},
}

fig10_path = os.path.join(FIG_DIR, "fig10_stg_gainer_flows.png")
render_gainer_anatomical_dissection(
    anatomy=fi_44,
    target_label=fl_dkt_44,
    models_config=models_cfg_stg,
    flow_sobolev=warp_sob_44,
    flow_continuum=warp_hyp_44,
    slice_axis=2,
    slice_idx=z_stg,
    target_label_ids=[1030, 2030],
    roi=roi_stg,
    structure_name="Superior Temporal Gyrus (Sylvian Bank)",
    gain_pct=5.16,
    sobolev_name="Sobolev SyN",
    continuum_name="Hyperelastic SyN",
    flow_vmax=15.0,
    delta_vmax=5.0,
    output_filename=fig10_path
)
print(f"--> Successfully generated Figure 10: {fig10_path}", flush=True)

print("\nAll figures regenerated using syntx.viz formal primitives!", flush=True)
