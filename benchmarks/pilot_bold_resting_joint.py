"""Pilot of the integrated resting-BOLD motion model on one 4D series (docs/MOTION_JOINT_OPTIMIZATION_PLAN.md, pilot).

One objective over all frames, solved jointly with L-BFGS from a zero pose:

    J(m, a) = sum_t sum_x  huber( [ I_t( y_t(x) + p * a_{t,g(x)} ) - c(x) ] / sigma(x) )
            + lambda_h * sum_t sqrt( delta^2 + |D2 m_t|^2 )                      (head: Charbonnier on 2nd differences)
            + 1/2 * sum_{t,g} a_{t,g}^2 / sigma_a^2                               (field: shrinkage toward zero)

- y_t(x) = c0 + R(w_t)(x - c0) + t_t is the rigid pose (6 DOF per frame) applied to the common point set x.
- a_{t,g} is a phase-encode field displacement per frame and per slice-timing group g (uniform within the group):
  a respiratory field shift differs between slice groups acquired at different respiratory phases, a rigid pose does not.
  For an acquisition with one slice group the term is a pure PE translation and only the two priors separate it.
- c(x) is the design-profiled series model; with no design in resting BOLD it is the robust temporal mean at each point.
- sigma(x) is the per-point robust residual scale (the motion-specificity weight): points whose residual stays large after
  alignment, i.e. pulsatile CSF and brainstem, carry little weight so the pose cannot be lowered by chasing them.
- the priors are tie-breakers in natural (nat) units; with ~2e4 points the data term dominates every identifiable
  direction, and the priors decide only the nullspace (uniform field vs PE translation).

Ablations: ``rigid`` (no field term) and ``uniform`` (no per-point scale). Baseline: syntx ``motion_correction`` with
the ``bold`` preset (the previous default). Evidence is image-based: temporal SD at tissue edges after correction, the
correlation of DVARS with the reported FD, and the reported FD itself with a notch at the alternation band.

Usage: python benchmarks/pilot_bold_resting_joint.py <bold.nii.gz> <out_dir> [--points 20000] [--groups 4] [--skip-baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

LAMBDA_HEAD = 20.0      # nats per mm of second-difference jump beyond the knee (Laplace scale 0.05 mm)
DELTA_HEAD = 0.05       # Charbonnier knee (mm)
SIGMA_FIELD = 0.5       # prior SD of the per-group field shift (mm)
HUBER = 1.5             # Huber transition in units of sigma(x)
ROT_RADIUS = 50.0       # mm, Power FD convention


def _rotmat(w):
    """Rotation matrices (T, 3, 3) from rotation vectors (T, 3) via Rodrigues (torch)."""
    import torch
    th = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros((w.shape[0], 3, 3), dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s = torch.sin(th)[:, :, None]
    c = torch.cos(th)[:, :, None]
    I = torch.eye(3, dtype=w.dtype, device=w.device)[None]
    return I + s * K + (1 - c) * (K @ K)


def _fd_power(t, w):
    """Power FD (mm) from translations (T,3) mm and rotation vectors (T,3) rad."""
    d = np.abs(np.diff(t, axis=0)).sum(1) + ROT_RADIUS * np.abs(np.diff(w, axis=0)).sum(1)
    return np.r_[0.0, d]


def _bandstop(X, tr, band):
    T = len(X)
    F = np.fft.rfft(X - X.mean(0), axis=0)
    fr = np.fft.rfftfreq(T, d=tr)
    F[(fr >= band[0]) & (fr <= band[1])] = 0
    return np.fft.irfft(F, n=T, axis=0) + X.mean(0)


def _alternation_band(tr):
    """Frequency band of the near-Nyquist physiological alternation for this TR (Hz)."""
    nyq = 0.5 / tr
    return (max(0.1, 0.5 * nyq), nyq)


class Pilot:
    def __init__(self, img, sidecar: dict, n_points: int, n_groups: int, device: str = "cpu", seed: int = 0):
        import ants
        import torch
        from syntx.spatial import get_spatial_coordinate_grid, physical_to_normalized_torch

        self.torch = torch
        self.dev = torch.device(device)
        self.img = img
        arr = img.numpy().astype(np.float32)                      # (X, Y, Z, T)
        self.T = arr.shape[-1]
        self.tr = float(sidecar.get("RepetitionTime", img.spacing[3]))
        frame0 = ants.slice_image(img, 3, 0)
        self.frame0 = frame0
        mean = arr.mean(-1)
        hi = float(np.percentile(mean, 99))
        self.fg = mean > 0.25 * hi
        # per-frame global gain (spatially constant intensity drift) removed up front
        gains = np.array([np.median(arr[..., k][self.fg]) for k in range(self.T)], dtype=np.float32)
        self.gains = gains / np.median(gains)
        arr = arr / self.gains[None, None, None, :] / hi
        self.vol = torch.tensor(np.ascontiguousarray(arr.transpose(3, 2, 1, 0)), device=self.dev)[:, None]   # (T,1,Z,Y,X)
        self.shape_zyx = tuple(self.vol.shape[2:])
        # phase-encode axis (physical, unit) from the sidecar and the direction matrix
        pe = sidecar.get("PhaseEncodingDirection", "j")
        ax = {"i": 0, "j": 1, "k": 2}[pe[0]]
        D = np.asarray(frame0.direction)[:3, :3]
        p = D[:, ax] * (-1.0 if pe.endswith("-") else 1.0)
        self.p_unit = torch.tensor(p / np.linalg.norm(p), dtype=torch.float32, device=self.dev)
        # slice-timing groups
        st = sidecar.get("SliceTiming")
        nz = frame0.shape[2]
        if st is not None and len(st) == nz and n_groups > 1:
            order = np.argsort(np.asarray(st, float), kind="stable")
            grp = np.zeros(nz, int)
            for gi, chunk in enumerate(np.array_split(order, n_groups)):
                grp[chunk] = gi
            self.n_groups = n_groups
        else:
            grp = np.zeros(nz, int)
            self.n_groups = 1
        self.slice_group = grp
        # common point set: foreground voxels, sampled with probability ~ gradient magnitude of the mean image
        gx, gy, gz = np.gradient(mean, *frame0.spacing[:3])
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) / hi
        self.gmag = gmag
        idx = np.argwhere(self.fg)
        prob = gmag[self.fg] + 0.1 * np.median(gmag[self.fg])
        prob = prob / prob.sum()
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(idx), size=min(n_points, len(idx)), replace=False, p=prob)
        self.vox = idx[pick]                                          # (S, 3) voxel indices x, y, z
        self.S = len(pick)
        phys_all = get_spatial_coordinate_grid(frame0, level=1, device="cpu")[0].cpu().numpy().reshape(nz, frame0.shape[1], frame0.shape[0], 3)
        # raster is (z, y, x) with x fastest -> index [z, y, x]
        self.P = torch.tensor(phys_all[self.vox[:, 2], self.vox[:, 1], self.vox[:, 0]], dtype=torch.float32, device=self.dev)   # (S,3) x,y,z mm
        self.P_all = torch.tensor(phys_all.reshape(-1, 3), dtype=torch.float32, device=self.dev)
        self.grp_pts = torch.tensor(grp[self.vox[:, 2]], device=self.dev)
        self.grp_all = torch.tensor(np.broadcast_to(grp[:, None, None], (nz, frame0.shape[1], frame0.shape[0])).reshape(-1), device=self.dev)
        cen = np.asarray([phys_all[tuple(v[::-1])] for v in idx]).mean(0)
        self.c0 = torch.tensor(cen, dtype=torch.float32, device=self.dev)
        self._to_norm = lambda phys: physical_to_normalized_torch(phys.flip(-1), self.shape_zyx, frame0.spacing[:3], frame0.origin[:3], np.asarray(frame0.direction)[:3, :3])

    # ---- forward -------------------------------------------------------------------------------------------------
    def warp_points(self, t, w, a, P, grp, frames=None):
        """Warped physical points (T', S, 3) for the given parameters."""
        R = _rotmat(w)
        y = (P[None] - self.c0) @ R.transpose(1, 2) + self.c0 + t[:, None, :]
        if a is not None:
            y = y + a[:, grp][:, :, None] * self.p_unit
        return y

    def sample(self, vol, y):
        """Trilinear samples (T, S) of vol (T,1,Z,Y,X) at physical points y (T,S,3)."""
        torch = self.torch
        g = self._to_norm(y)[:, None, None, :, :]                      # (T,1,1,S,3) in (x,y,z) normalised
        out = torch.nn.functional.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out[:, 0, 0, 0, :]

    # ---- objective ----------------------------------------------------------------------------------------------
    def solve(self, use_field=True, per_point_scale=True, rounds=(3, 5), iters=25, verbose=True):
        torch = self.torch
        T, G = self.T, self.n_groups
        t = torch.zeros((T, 3), device=self.dev, requires_grad=True)
        w = torch.zeros((T, 3), device=self.dev, requires_grad=True)
        a = torch.zeros((T, G), device=self.dev, requires_grad=use_field)
        params = [t, w] + ([a] if use_field else [])
        smooth = torch.nn.functional.avg_pool3d(self.vol, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        log = []
        for level, (vol, n_rounds) in enumerate(((smooth, rounds[0]), (self.vol, rounds[1]))):
            for rnd in range(n_rounds):
                with torch.no_grad():
                    E = self.sample(vol, self.warp_points(t, w, a if use_field else None, self.P, self.grp_pts))
                    c = E.median(dim=0).values
                    r0 = E - c
                    mad = 1.4826 * (r0 - r0.median(dim=0).values).abs().median(dim=0).values
                    glob = mad.median()
                    sigma = mad.clamp_min(0.5 * glob) if per_point_scale else torch.full_like(mad, float(glob))
                opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

                def closure():
                    opt.zero_grad()
                    E = self.sample(vol, self.warp_points(t, w, a if use_field else None, self.P, self.grp_pts))
                    u = (E - c) / sigma
                    au = u.abs()
                    data = torch.where(au < HUBER, 0.5 * u * u, HUBER * (au - 0.5 * HUBER)).sum()
                    d2t = t[2:] - 2 * t[1:-1] + t[:-2]
                    d2w = ROT_RADIUS * (w[2:] - 2 * w[1:-1] + w[:-2])
                    head = LAMBDA_HEAD * torch.sqrt(DELTA_HEAD ** 2 + (d2t * d2t).sum(1) + (d2w * d2w).sum(1)).sum()
                    field = 0.5 * (a * a).sum() / SIGMA_FIELD ** 2 if use_field else torch.zeros((), device=self.dev)
                    loss = data + head + field
                    loss.backward()
                    closure.parts = (float(data), float(head), float(field))
                    return loss

                loss = opt.step(closure)
                log.append(dict(level=level, round=rnd, loss=float(loss), data=closure.parts[0], head=closure.parts[1], field=closure.parts[2]))
                if verbose:
                    print("  level %d round %d loss %.1f data %.1f head %.2f field %.2f" % (level, rnd, float(loss), *closure.parts), flush=True)
        return dict(t=t.detach().cpu().numpy().astype(float), w=w.detach().cpu().numpy().astype(float),
                    a=a.detach().cpu().numpy().astype(float) if use_field else None, log=log,
                    sigma=sigma.cpu().numpy().astype(float))

    # ---- corrected series on the full grid -------------------------------------------------------------------------
    def corrected(self, t, w, a, chunk=8):
        torch = self.torch
        out = []
        tt = torch.tensor(t, dtype=torch.float32, device=self.dev)
        ww = torch.tensor(w, dtype=torch.float32, device=self.dev)
        aa = None if a is None else torch.tensor(a, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            for k in range(0, self.T, chunk):
                sl = slice(k, k + chunk)
                y = self.warp_points(tt[sl], ww[sl], None if aa is None else aa[sl], self.P_all, self.grp_all)
                out.append(self.sample(self.vol[sl], y).cpu().numpy())
        arr = np.concatenate(out, 0).reshape(self.T, *self.shape_zyx).transpose(3, 2, 1, 0)   # (X,Y,Z,T)
        return arr


def image_metrics(arr, fg, gmag, fd):
    """Temporal SD at tissue edges and in the foreground, DVARS and its correlation with FD."""
    edge = fg & (gmag >= np.percentile(gmag[fg], 80))
    m = arr.mean(-1)
    sd = arr.std(-1)
    rel_edge = float(np.median(sd[edge] / (m[edge] + 1e-6)))
    rel_fg = float(np.median(sd[fg] / (m[fg] + 1e-6)))
    d = np.diff(arr[fg], axis=-1)
    dvars = np.sqrt((d ** 2).mean(0))
    corr = float(np.corrcoef(dvars, fd[1:])[0, 1]) if np.std(fd[1:]) > 0 else float("nan")
    return dict(tsd_edge=rel_edge, tsd_fg=rel_fg, dvars_mean=float(dvars.mean()), dvars_fd_corr=corr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bold")
    ap.add_argument("out")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args(argv)
    import ants
    import torch

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    img = ants.image_read(args.bold)
    sidecar_path = args.bold.replace(".nii.gz", ".json")
    sidecar = json.load(open(sidecar_path)) if os.path.exists(sidecar_path) else {}
    pil = Pilot(img, sidecar, args.points, args.groups, device=args.device)
    tr = pil.tr
    band = _alternation_band(tr)
    print("T %d  TR %.2f  points %d  slice groups %d  band %s" % (pil.T, tr, pil.S, pil.n_groups, band), flush=True)
    raw = img.numpy().astype(np.float32) / pil.gains[None, None, None, :]
    results = {"meta": dict(T=pil.T, tr=tr, points=pil.S, groups=pil.n_groups, band=band,
                            lambda_head=LAMBDA_HEAD, delta_head=DELTA_HEAD, sigma_field=SIGMA_FIELD)}
    results["uncorrected"] = dict(image=image_metrics(raw, pil.fg, pil.gmag, np.zeros(pil.T)))

    if not args.skip_baseline:
        import syntx
        t0 = time.time()
        r = syntx.motion_correction(img, reference="mean", type_of_transform="Rigid", preset="bold", verbose=False)
        mp = r.motion_parameters
        tb = np.asarray(mp.translations, float)
        wb = np.radians(np.asarray(mp.rotations_deg, float))
        fdb = np.asarray(r.fd, float)
        corr = r["motion_corrected"].numpy().astype(np.float32) / pil.gains[None, None, None, :]
        results["baseline_unified"] = dict(seconds=time.time() - t0, fd_mean=float(fdb.mean()), fd_frac_gt_0p3=float(np.mean(fdb > 0.3)),
                                           fd_notch_mean=float(_fd_power(_bandstop(tb, tr, band), _bandstop(wb, tr, band)).mean()),
                                           image=image_metrics(corr, pil.fg, pil.gmag, fdb))
        np.save(os.path.join(args.out, "baseline_poses.npy"), np.c_[tb, wb])
        print("baseline", json.dumps({k: v for k, v in results["baseline_unified"].items() if k != "image"}), results["baseline_unified"]["image"], flush=True)

    variants = [("full", dict(use_field=True, per_point_scale=True)),
                ("rigid_only", dict(use_field=False, per_point_scale=True)),
                ("uniform_scale", dict(use_field=True, per_point_scale=False))]
    for name, kw in variants:
        t0 = time.time()
        print(name, flush=True)
        sol = pil.solve(**kw)
        fd = _fd_power(sol["t"], sol["w"])
        tn, wn = _bandstop(sol["t"], tr, band), _bandstop(sol["w"], tr, band)
        rec = dict(seconds=time.time() - t0, fd_mean=float(fd.mean()), fd_frac_gt_0p3=float(np.mean(fd > 0.3)),
                   fd_notch_mean=float(_fd_power(tn, wn).mean()), log=sol["log"])
        if sol["a"] is not None:
            a = sol["a"]
            rec["field_sd_mm"] = [float(x) for x in a.std(0)]
            rec["field_group_spread_sd_mm"] = float((a - a.mean(1, keepdims=True)).std()) if a.shape[1] > 1 else 0.0
            P = np.abs(np.fft.rfft(a.mean(1) - a.mean())) ** 2
            fr = np.fft.rfftfreq(pil.T, d=tr)
            rec["field_band_power_frac"] = float(P[(fr >= band[0]) & (fr <= band[1])].sum() / max(P[1:].sum(), 1e-12))
        corr = pil.corrected(sol["t"], sol["w"], sol["a"])
        rec["image"] = image_metrics(corr, pil.fg, pil.gmag, fd)
        if name == "full":
            # real-vs-pseudo check: keep the field, notch the rigid alternation, and see whether edges get worse or better
            corr_n = pil.corrected(tn, wn, sol["a"])
            rec["image_rigid_notched"] = image_metrics(corr_n, pil.fg, pil.gmag, _fd_power(tn, wn))
            corr_nf = pil.corrected(sol["t"], sol["w"], None)
            rec["image_field_dropped"] = image_metrics(corr_nf, pil.fg, pil.gmag, fd)
        np.savez(os.path.join(args.out, f"{name}_solution.npz"), t=sol["t"], w=sol["w"], a=sol["a"] if sol["a"] is not None else np.zeros(0), sigma=sol["sigma"])
        results[name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k not in ("image", "log", "image_rigid_notched", "image_field_dropped")}), rec["image"], flush=True)
        json.dump(results, open(os.path.join(args.out, "pilot.json"), "w"), indent=1)
    _figures(pil, results, args.out)
    print("DONE", args.out)


def _figures(pil, results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = pil.tr
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    tt = np.arange(pil.T) * tr
    if os.path.exists(os.path.join(out, "baseline_poses.npy")):
        b = np.load(os.path.join(out, "baseline_poses.npy"))
        axes[0].plot(tt, _fd_power(b[:, :3], b[:, 3:]), lw=0.8, label="baseline unified FD")
    for name in ("full", "rigid_only"):
        f = os.path.join(out, f"{name}_solution.npz")
        if os.path.exists(f):
            s = np.load(f)
            axes[0].plot(tt, _fd_power(s["t"], s["w"]), lw=0.8, label=f"pilot {name} rigid FD")
    axes[0].set_ylabel("FD (mm)"); axes[0].legend(fontsize=8)
    s = np.load(os.path.join(out, "full_solution.npz"))
    D = np.asarray(pil.frame0.direction)[:3, :3]
    for j, lab in ((1, "PE"), (2, "slice")):
        axes[1].plot(tt, s["t"] @ D[:, j], lw=0.8, label=f"pilot full translation {lab}")
    axes[1].set_ylabel("mm"); axes[1].legend(fontsize=8)
    if s["a"].size:
        for g in range(s["a"].shape[1]):
            axes[2].plot(tt, s["a"][:, g], lw=0.8, label=f"field shift group {g}")
        axes[2].legend(fontsize=8)
    axes[2].set_ylabel("PE field (mm)"); axes[2].set_xlabel("s")
    fig.tight_layout(); fig.savefig(os.path.join(out, "traces.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
