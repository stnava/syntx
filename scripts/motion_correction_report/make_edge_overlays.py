import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants
from scipy import ndimage

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; BORDER = "#27354a"

def edge_mask(arr2d, sigma=1.2, pct=90):
    smoothed = ndimage.gaussian_filter(arr2d.astype(np.float32), sigma=sigma)
    gx = ndimage.sobel(smoothed, axis=0)
    gy = ndimage.sobel(smoothed, axis=1)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag[mag > 0], pct) if np.any(mag > 0) else 1.0
    return mag > thresh

def norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - v0) / max(v1 - v0, 1e-6), 0, 1)

def panel(ax, base2d, edge2d, title, cmap="gray", edge_color=(0.3, 1.0, 0.3, 0.95)):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(base2d, cmap=cmap, vmin=0, vmax=1)
    overlay = np.zeros(base2d.shape + (4,))
    overlay[edge2d] = edge_color
    ax.imshow(overlay)
    ax.set_title(title, color=TEXT, fontsize=10)

# ============================= DWI cross-registration: edge overlay, not checkerboard =============================
# Different contrasts (b0 vs DWI) -- a checkerboard confounds "tile looks different" with
# "tile is misaligned"; an edge overlay only asks "do the ANATOMICAL BOUNDARIES line up",
# which is well-defined regardless of the two images' very different absolute contrast.
b0_mean = ants.image_read("/tmp/motion_report/dwi_ref_two_mean_b0.nii.gz")
dwi_mean = ants.image_read("/tmp/motion_report/dwi_ref_two_mean_dwi.nii.gz")
from syntx.robust_affine import robust_affine
cross = robust_affine(fixed=b0_mean, moving=dwi_mean, dof="rigid", mode="auto", verbose=False)
dwi_warped = cross["warpedmovout"]

zi = b0_mean.numpy().shape[2] // 2
sl = lambda im: im.numpy()[:, :, zi].T[::-1, :]
b0_n = norm(sl(b0_mean))
dwi_raw_n = norm(sl(dwi_mean))
dwi_warped_n = norm(sl(dwi_warped))
b0_edges = edge_mask(b0_n, pct=88)

fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), facecolor=BG)
panel(axes[0], dwi_raw_n, b0_edges, "BEFORE cross-registration\nDWI mean (grayscale) + B0 mean's edges (green)\n-- look closely at the ventricle/sulcal boundaries: a small but real offset")
panel(axes[1], dwi_warped_n, b0_edges, "AFTER the one careful cross-registration\nwarped DWI mean (grayscale) + SAME B0 edges (green)\n-- edges now track the DWI mean's own boundaries precisely")
fig.suptitle("DWI: edge-overlay alignment check (B0 <-> DWI are different contrasts -- checkerboard is not a valid tool here)",
             color=TEXT, fontsize=12, fontweight="bold")
fig.patch.set_facecolor(BG)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("/tmp/motion_report/dwi_cross_reg_edge_overlay.png", dpi=150, facecolor=BG)
print("wrote dwi_cross_reg_edge_overlay.png")

# ============================= PET: edge overlay instead of checkerboard =============================
info = json.load(open("/tmp/motion_report/pet_worst_frame_info.json"))
ref = ants.image_read("/tmp/motion_report/pet_ref_naive.nii.gz")

path = "data_cache/ds002898/sub-01/pet/sub-01_task-rest_trc-18FFDG_rec-acdyn_run-001_pet.nii.gz"
img = ants.image_read(path)
START, N = 40, 32
arr = img.numpy()[..., START:START+N]
img4d = ants.from_numpy(np.ascontiguousarray(arr), spacing=img.spacing, origin=img.origin, direction=img.direction)
frames = [ants.slice_image(img4d, axis=3, idx=i) for i in range(N)]
frames_ds = [ants.resample_image(f, (4.,4.,4.), use_voxels=False, interp_type=0) for f in frames]
img4d_ds = ants.list_to_ndimage(ants.from_numpy(np.zeros(frames_ds[0].shape + (N,), dtype=np.float32)), frames_ds)
import syntx
ref_ds = ants.get_average_of_timeseries(img4d_ds)
mc = syntx.motion_correction(image=img4d_ds, reference=ref_ds, backend="pytorch_batched_adaptive", verbose=False)

raw_a = ants.slice_image(img4d_ds, axis=3, idx=info['worst_frame']).numpy()
corr_a = ants.slice_image(mc.motion_corrected, axis=3, idx=info['worst_frame']).numpy()
ref_a = ref_ds.numpy()
zi2 = ref_a.shape[2] // 2
sl2 = lambda a: a[:, :, zi2].T[::-1, :]
ref_n2, raw_n2, corr_n2 = norm(sl2(ref_a)), norm(sl2(raw_a)), norm(sl2(corr_a))
ref_edges2 = edge_mask(ref_n2, pct=85)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5.4), facecolor=BG)
panel(axes2[0], raw_n2, ref_edges2, f"BEFORE: raw frame #{info['worst_frame']} (Mattes FD={info['worst_fd']:.2f}mm)\n+ reference edges (green)\n-- edges already fall ON the frame's own boundary",
      cmap="inferno", edge_color=(0.2, 1.0, 0.2, 1.0))
panel(axes2[1], corr_n2, ref_edges2, "AFTER 'correction'\n+ same reference edges (green)\n-- no visible change: there was no spatial offset to fix",
      cmap="inferno", edge_color=(0.2, 1.0, 0.2, 1.0))
fig2.suptitle("FDG-PET: the edge overlay confirms the 'high FD' frame was never spatially misaligned", color=TEXT, fontsize=12, fontweight="bold")
fig2.patch.set_facecolor(BG)
fig2.tight_layout(rect=[0, 0, 1, 0.88])
fig2.savefig("/tmp/motion_report/pet_edge_overlay.png", dpi=150, facecolor=BG)
print("wrote pet_edge_overlay.png")
