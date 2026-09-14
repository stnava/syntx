#!/usr/bin/env python3
"""
scripts/benchmark_landmarks.py
================================
Test syntx.landmarks detectors on:
  1. Mindboggle-hard (mbhard) — one intra-cohort brain MRI pair
  2. Medical Segmentation Decathlon tasks — one pair per available task

For each pair:
  - Run all 4 detectors (LoG, DoG, SIFT2D, SIFT3D)
  - Match with Lowe ratio + RANSAC
  - Report: n_keypoints, n_matches, n_inliers, TRE before/after RANSAC
  - Render a 3-panel figure: fixed with landmarks, matching overlay, inlier overlay

Usage:
  python3 scripts/benchmark_landmarks.py
  python3 scripts/benchmark_landmarks.py --datasets mbhard decathlon
  python3 scripts/benchmark_landmarks.py --max-dim 128 --no-sift3d
"""
import argparse, os, sys, time, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import ants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from pathlib import Path

import syntx
from syntx.landmarks import (
    detect_blobs_log, detect_blobs_dog,
    detect_sift3d,
    compute_mind, extract_mind_at_points,
    match_landmarks, ransac_filter, compute_tre,
)
from syntx.landmarks import spatial as S   # all mm <-> display projections go through here

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'figure.dpi': 160, 'savefig.dpi': 160,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
COL_F = '#2166AC'; COL_M = '#D6604D'; COL_IN = '#1B7837'; COL_OUT = '#D6604D'


def _norm(arr):
    lo, hi = arr.min(), arr.max()
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def _scatter_pts(ax, img, pts_mm, view, color, s=10, marker='o', alpha=0.75, slab_mm=None, x_offset_mm=0.0):
    """Project physical mm coords onto a display view spec (from extract_ortho_slices)."""
    u, v, m = S.project_to_slice(img, pts_mm[:, :3], view=view,
                                 slab_half_mm=slab_mm if slab_mm is not None else 3 * view["spacing_u"])
    ax.scatter((u[m] + 0.5) * view["spacing_u"] + x_offset_mm, (v[m] + 0.5) * view["spacing_v"],
               c=color, s=s, marker=marker, alpha=alpha, linewidths=0.3, edgecolors='white', zorder=5)
    return int(m.sum())


def preprocess(img, max_dim=256):
    """Resample so max dimension ≤ max_dim, then foreground percentile norm."""
    shape = img.shape
    max_shape = max(shape)
    if max_shape > max_dim:
        factor = max_dim / max_shape
        new_sp = tuple(s / factor for s in img.spacing)
        img = ants.resample_image(img, new_sp, use_voxels=False, interp_type=1)
    return img


def run_pair(pair_id, fixed, moving, out_dir, max_dim=256,
             do_sift3d=True, n_slices=6, max_kpts=200):
    """
    Run all landmark detectors on a pair. Returns a results dict.
    """
    print(f"\n  {'─'*55}")
    print(f"  Pair: {pair_id}  |  fixed {fixed.shape} @ {fixed.spacing}")

    t0 = time.perf_counter()

    # Preprocess
    fi = preprocess(fixed,  max_dim)
    mi = preprocess(moving, max_dim)

    results = {'pair_id': pair_id, 'detectors': {}}

    detectors = [
        ('LoG',    lambda img: (detect_blobs_log(img, max_keypoints=max_kpts), None)),
        ('DoG',    lambda img: (detect_blobs_dog(img, max_keypoints=max_kpts), None)),
        # SIFT2D is banned on 3D volumetric data (GEMINI.md §23): only true 3D detectors here.
    ]
    if do_sift3d:
        detectors.append(('SIFT3D', lambda img: detect_sift3d(img, max_keypoints=max_kpts)))

    best_det = None; best_inliers = -1; best_result = None

    for det_name, det_fn in detectors:
        t_det = time.perf_counter()
        try:
            out_f = det_fn(fi); out_m = det_fn(mi)
            # LoG/DoG return ([N,4], None) — no descriptors; skip matching
            if out_f[1] is None:
                c_f, d_f = out_f[0], None
                c_m, d_m = out_m[0], None
                n_match = n_in = 0
                tre_raw = tre_in = float('nan')
            else:
                c_f, d_f = out_f
                c_m, d_m = out_m

                # CoM pre-alignment: translate moving keypoints to match
                # fixed CoM — handles large initial displacement (e.g. brain
                # images not in the same coordinate frame)
                if c_f.shape[0] > 0 and c_m.shape[0] > 0:
                    com_f = c_f[:, :3].mean(axis=0)
                    com_m = c_m[:, :3].mean(axis=0)
                    c_m_aligned = c_m.copy()
                    c_m_aligned[:, :3] += (com_f - com_m)
                else:
                    c_m_aligned = c_m

                # Mutual-check ratio matching (wider ratio for cross-subject)
                matches_fwd = match_landmarks(c_f, c_m_aligned, d_f, d_m,
                                              ratio_thresh=0.82)
                matches_bwd = match_landmarks(c_m_aligned, c_f, d_m, d_f,
                                              ratio_thresh=0.82)
                # Keep only mutually consistent pairs
                bwd_set = {(b[1], b[0]) for b in matches_bwd.tolist()}
                mutual = np.array([m for m in matches_fwd.tolist()
                                   if tuple(m) in bwd_set], dtype=np.int32)
                if mutual.shape[0] == 0:
                    mutual = matches_fwd  # fall back to unidirectional

                n_match = len(mutual)

                # Adaptive inlier threshold: 3× max voxel spacing
                inlier_thresh = max(6.0, 3.0 * max(fi.spacing))

                if n_match >= 4:
                    matches_in, M = ransac_filter(
                        c_f, c_m_aligned, mutual,
                        model='affine', inlier_thresh_mm=inlier_thresh,
                        min_inliers=4, max_iter=2000)
                    n_in = len(matches_in)
                    tre_raw = (compute_tre(c_f[mutual[:, 0], :3],
                                          c_m_aligned[mutual[:, 1], :3])
                               if n_match > 0 else float('nan'))
                    tre_in  = (compute_tre(c_f[matches_in[:, 0], :3],
                                           c_m_aligned[matches_in[:, 1], :3])
                               if n_in > 0 else float('nan'))
                    if n_in > best_inliers:
                        best_inliers = n_in
                        best_det = det_name
                        best_result = (c_f, d_f, c_m_aligned, d_m,
                                       mutual, matches_in)
                else:
                    matches_in = np.zeros((0, 2), dtype=np.int32)
                    n_in = 0; tre_raw = tre_in = float('nan')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [{det_name}] ERROR: {e}")
            c_f = np.zeros((0, 4)); n_match = n_in = 0
            tre_raw = tre_in = float('nan')

        t_det = time.perf_counter() - t_det
        n_f = len(c_f)
        n_m = len(det_fn(mi)[0]) if det_fn(mi)[1] is None else len(det_fn(mi)[0])

        inlier_pct = 100 * n_in / max(1, n_match)
        print(f"    [{det_name:6s}] fixed={n_f:4d} pts  "
              f"matches={n_match:4d}  inliers={n_in:4d} ({inlier_pct:.0f}%)  "
              f"TRE {tre_raw:.1f}→{tre_in:.1f}mm  ({t_det*1e3:.0f}ms)")

        results['detectors'][det_name] = {
            'n_fixed': n_f, 'n_matches': n_match, 'n_inliers': n_in,
            'inlier_pct': round(inlier_pct, 1),
            'tre_before': round(tre_raw, 2) if not np.isnan(tre_raw) else None,
            'tre_after':  round(tre_in, 2)  if not np.isnan(tre_in)  else None,
            'time_ms': round(t_det * 1e3, 1),
        }

    results['total_time_s'] = round(time.perf_counter() - t0, 2)
    results['best_detector'] = best_det

    # ── Figure: 3-panel for best descriptor detector ─────────────────────────
    fig_path = out_dir / f"{pair_id.replace('/', '_')}_landmarks.png"
    try:
        _render_figure(pair_id, fi, mi, best_det, best_result, fig_path)
        results['figure'] = str(fig_path)
        print(f"    Figure → {fig_path.name}")
    except Exception as e:
        print(f"    Figure failed: {e}")

    return results


def _render_figure(pair_id, fi, mi, best_det, best_result, fig_path):
    """Axial views in medical display orientation (radiological), native arrays, mm extents.
    Points/lines are projected with syntx.landmarks.spatial so overlays follow the direction matrix."""
    if best_result is None or best_det is None:
        return
    c_f, d_f, c_m, d_m, matches, matches_in = best_result
    sl_f = S.extract_ortho_slices(fi)
    sl_m = S.extract_ortho_slices(mi)
    vf, vm = sl_f["views"]["ax"], sl_m["views"]["ax"]
    ax_f = _norm(sl_f["ax"]); ax_m = _norm(sl_m["ax"])
    wf, hf = vf["n_u"] * vf["spacing_u"], vf["n_v"] * vf["spacing_v"]
    wm, hm = vm["n_u"] * vm["spacing_u"], vm["n_v"] * vm["spacing_v"]
    gap = 10.0
    slab = 3.0 * max(max(fi.spacing), max(mi.spacing))
    inlier_set = set(map(tuple, matches_in.tolist()))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8),
                              gridspec_kw={'wspace': 0.06})
    fig.patch.set_facecolor('#111')

    def _both(ax):
        ax.imshow(ax_f, cmap='gray', origin='lower', extent=[0, wf, 0, hf])
        ax.imshow(ax_m, cmap='gray', origin='lower', extent=[wf + gap, wf + gap + wm, 0, hm])
        ax.axvline(wf + gap / 2, color='white', lw=0.7, ls='--', alpha=0.5)
        ax.set_xlim(0, wf + gap + wm); ax.set_ylim(0, max(hf, hm)); ax.set_aspect('equal')

    def _lines(ax, pairs, colour_fn, lw_fn, la_fn):
        uf, vvf, mf_ = S.project_to_slice(fi, c_f[pairs[:, 0], :3], view=vf, slab_half_mm=slab)
        um, vvm, mm_ = S.project_to_slice(mi, c_m[pairs[:, 1], :3], view=vm, slab_half_mm=slab)
        for k in np.where(mf_ & mm_)[0]:
            key = tuple(pairs[k].tolist())
            ax.plot([(uf[k] + 0.5) * vf["spacing_u"], wf + gap + (um[k] + 0.5) * vm["spacing_u"]],
                    [(vvf[k] + 0.5) * vf["spacing_v"], (vvm[k] + 0.5) * vm["spacing_v"]],
                    '-', color=colour_fn(key), lw=lw_fn(key), alpha=la_fn(key))

    # Panel 1: fixed with keypoints
    axes[0].imshow(ax_f, cmap='gray', origin='lower', extent=[0, wf, 0, hf])
    _scatter_pts(axes[0], fi, c_f, vf, COL_F, s=12, slab_mm=slab)
    lab = vf["labels"]
    axes[0].set_title(f'Fixed  ({len(c_f)} keypoints)  [{lab["left"]}|{lab["right"]}, {lab["top"]} up]',
                      color='white', fontsize=8)

    # Panel 2: raw matches
    _both(axes[1])
    _lines(axes[1], matches[::max(1, len(matches)//60)],
           lambda k: COL_IN if k in inlier_set else '#FF7F00',
           lambda k: 0.8 if k in inlier_set else 0.4,
           lambda k: 0.75 if k in inlier_set else 0.3)
    axes[1].set_title(
        f'{best_det}: {len(matches)} raw → {len(matches_in)} inliers',
        color='white', fontsize=8)

    # Panel 3: inliers only
    _both(axes[2])
    _lines(axes[2], matches_in[::max(1, len(matches_in)//80)],
           lambda k: COL_IN, lambda k: 0.9, lambda k: 0.7)
    n_in = len(matches_in); n_raw = len(matches)
    pct = 100*n_in/max(1,n_raw)
    axes[2].set_title(
        f'RANSAC inliers: {n_in}/{n_raw} ({pct:.0f}%)', color='white', fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)

    fig.suptitle(f'{pair_id}  |  {best_det} landmark matching (axial mid-slice)',
                 color='white', fontsize=9, y=1.01)
    fig.savefig(fig_path, facecolor='#111')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['mbhard', 'decathlon'])
    ap.add_argument('--max-dim', type=int, default=192)
    ap.add_argument('--no-sift3d', action='store_true')
    ap.add_argument('--n-slices', type=int, default=8,
                    help='n_slices_per_axis for SIFT2D')
    ap.add_argument('--max-kpts', type=int, default=256)
    args = ap.parse_args()

    out_dir = Path('/Users/stnava/.gemini/antigravity-cli/brain/ed9af813-1310-43df-bc86-a32aeec400a0/scratch/landmark_benchmark')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. mbhard
    # ──────────────────────────────────────────────────────────────────────────
    if 'mbhard' in args.datasets:
        print('\n' + '═'*60)
        print('  DATASET: Mindboggle-Hard (mbhard) — intra-cohort brain MRI')
        print('═'*60)
        ds = syntx.benchmark_data('mbhard')
        res = run_pair(
            pair_id='mbhard_pair00',
            fixed=ds['fixed'], moving=ds['moving'],
            out_dir=out_dir,
            max_dim=args.max_dim,
            do_sift3d=not args.no_sift3d,
            n_slices=args.n_slices,
            max_kpts=args.max_kpts,
        )
        all_results['mbhard_pair00'] = res

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Decathlon tasks
    # ──────────────────────────────────────────────────────────────────────────
    if 'decathlon' in args.datasets:
        DECATH_ROOT = Path('/Users/stnava/data/decathlon')
        tasks = sorted(DECATH_ROOT.glob('Task*'))

        print('\n' + '═'*60)
        print(f'  DATASET: Medical Segmentation Decathlon ({len(tasks)} tasks)')
        print('═'*60)

        for task_dir in tasks:
            img_dir = task_dir / 'imagesTr'
            if not img_dir.exists():
                print(f"  [SKIP] {task_dir.name} — no imagesTr/")
                continue
            imgs = sorted(img_dir.glob('*.nii.gz'))
            if len(imgs) < 2:
                print(f"  [SKIP] {task_dir.name} — < 2 images")
                continue

            task_name = task_dir.name
            print(f'\n  ── {task_name} ──')

            try:
                # Skip macOS hidden files
                real_imgs = [p for p in imgs if not p.name.startswith('._')]
                if len(real_imgs) < 2:
                    print(f"  [SKIP] {task_name} — < 2 readable images")
                    continue
                fi_raw = ants.image_read(str(real_imgs[0]))
                mi_raw = ants.image_read(str(real_imgs[1]))

                # Multi-channel volumes (e.g. BRATS 4-channel): take channel 0
                # ANTsImage uses .components or shape[-1] to detect channels
                if len(fi_raw.shape) == 4:
                    fi_arr = fi_raw.numpy()[..., 0]
                    fi_raw = ants.from_numpy(fi_arr, origin=fi_raw.origin[:3],
                                             spacing=fi_raw.spacing[:3],
                                             direction=fi_raw.direction[:3, :3])
                if len(mi_raw.shape) == 4:
                    mi_arr = mi_raw.numpy()[..., 0]
                    mi_raw = ants.from_numpy(mi_arr, origin=mi_raw.origin[:3],
                                             spacing=mi_raw.spacing[:3],
                                             direction=mi_raw.direction[:3, :3])

                res = run_pair(
                    pair_id=task_name,
                    fixed=fi_raw, moving=mi_raw,
                    out_dir=out_dir,
                    max_dim=args.max_dim,
                    do_sift3d=not args.no_sift3d,
                    n_slices=args.n_slices,
                    max_kpts=args.max_kpts,
                )
                all_results[task_name] = res
            except Exception as e:
                import traceback
                print(f"    ERROR: {e}")
                traceback.print_exc()
                all_results[task_name] = {'error': str(e)}

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / 'landmark_benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Results → {json_path}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '═'*80)
    print(f'  {"Pair / Task":<30}  {"Best":<7}  '
          f'{"Matches":>8}  {"Inliers":>8}  {"Pct%":>5}  {"TRE→mm":>8}')
    print('  ' + '─'*78)
    for pid, res in all_results.items():
        if 'error' in res:
            print(f'  {pid:<30}  ERROR: {res["error"][:40]}')
            continue
        best = res.get('best_detector', '—')
        if best and best in res['detectors']:
            d = res['detectors'][best]
            tre_str = (f"{d['tre_before']:.1f}→{d['tre_after']:.1f}"
                       if d['tre_after'] is not None else 'N/A')
            print(f'  {pid:<30}  {best:<7}  '
                  f'{d["n_matches"]:>8d}  {d["n_inliers"]:>8d}  '
                  f'{d["inlier_pct"]:>5.1f}  {tre_str:>8}')
    print('═'*80)


if __name__ == '__main__':
    main()
