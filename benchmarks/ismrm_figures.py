"""Figures for the ISMRM abstract: the generative model and the prior (Figure 1), the evidence (Figure 2).

Figure 1: (A) the forward model of one frame, drawn with slices of the DWI simulator (head pose, eddy shear along phase
encode, shell contrast, gain); (B) the three surround penalties and their pull, with the knee; (C) what each penalty does to
a 1-D pose series made of drift + sub-knee jitter + one jump against a unit-curvature image term.
Figure 2: (A) the phantom gate (FD error against the known trajectory per setting); (B) the readout-floor rule on the
high-motion DWI series (B0-immune readout trace with no prior, cauchy 0.03, cauchy 0.05); (C) the cohort response (readout
step and FD change per series for the two weights).

Usage: python benchmarks/ismrm_figures.py <gate dir with phantoms.json and cohort.json> <out dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _denoise(y, w, d, form, curvature=1.0):
    import torch
    from syntx.motion_batched import surround_rho

    y_t = torch.as_tensor(y, dtype=torch.float64)
    x = y_t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        u = x[1:-1] - 0.5 * (x[:-2] + x[2:])
        loss = 0.5 * curvature * ((x - y_t) ** 2).sum() + surround_rho(u * u, w, d, form)[0].sum()
        loss.backward()
        return loss

    opt.step(closure)
    return x.detach().numpy()


def figure1(out):
    import ants
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from syntx.motion_batched import surround_rho
    from syntx.motion_simulation import simulate_dwi_series

    sim = simulate_dwi_series(16, seed=42, eddy_mm_per_unit=0.4)
    arr = sim.image.numpy()
    bv = np.asarray(sim.extra["bvals"])
    k0 = int(np.where(bv <= 50)[0][0]); kdw = int(np.argmax(np.abs(np.asarray(sim.eddy)[:, 0]))) if sim.eddy is not None else int(np.where(bv > 50)[0][0])
    z = arr.shape[2] // 2
    b0 = arr[:, :, z, k0]; dw = arr[:, :, z, kdw]
    diff = dw - b0

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.75, 0.9, 1.1], hspace=0.42, wspace=0.32)

    # ---------------- A: the forward model as a flow of boxes (row 0) with real slices (row 1)
    axA = fig.add_subplot(gs[0, :])
    axA.set_xlim(0, 100); axA.set_ylim(8, 30); axA.axis("off")
    axA.set_title("A. The forward model of frame k: one rigid head pose, the acquisition's own displacement and contrast, a per-frame gain", fontsize=12, loc="left")
    boxes = [
        (3, "reference of the\nframe's level $\\bar I_{\\ell(k)}(x)$\n(contrast lives here)", "#dfe9f3"),
        (21, "design-locked pose offset\n$D_k\\,\\beta$ (per level; $l=2$ terms)\nremoved from FD, never applied", "#e6f2df"),
        (39, "head pose $m_k$: $x \\mapsto R_k x + t_k$\n6 free parameters per frame;\nframe pose $= m_k + D_k\\beta$", "#fbe3d6"),
        (57, "eddy affine along $p$\n$y \\mapsto y + p\\,(a_{k0} + \\tilde a_k^{\\top}(y-c))$\n$(a_{k0}, \\tilde a_k) = C\\,g_k$, Jacobian $1 + \\tilde a_k^{\\top}p$", "#f3e3f3"),
        (75, "gain $\\gamma_k$\n(scanner drift,\nspatially uniform)", "#f6f0d6"),
        (91, "observed\nframe $I_k$", "#e8e8e8"),
    ]
    for x0, text, color in boxes:
        axA.add_patch(FancyBboxPatch((x0, 17), 13 if x0 < 91 else 8, 11, boxstyle="round,pad=0.4", fc=color, ec="#555", lw=1.0))
        axA.text(x0 + (6.5 if x0 < 91 else 4), 22.5, text, ha="center", va="center", fontsize=9)
    for x0 in (16.5, 34.5, 52.5, 70.5, 88.5):
        axA.add_patch(FancyArrowPatch((x0, 22.5), (x0 + 4.2, 22.5), arrowstyle="-|>", mutation_scale=14, color="#333", lw=1.2))
    axA.text(50, 13.6, "estimated jointly over all frames: the head poses $\\{R_k, t_k\\}$ carry a prior on their second differences; the design terms carry none",
             ha="center", fontsize=9.5, style="italic")
    axA.text(50, 10.6, "applied to the data: head pose, eddy affine, Jacobian.   removed from the reported pose, never applied: design-locked offsets.   FD: head pose only.",
             ha="center", fontsize=9.5)
    # slices of the simulator
    p = np.asarray(sim.pe_axis, float)
    for i, (img, title, cmap, vmin, vmax) in enumerate([
            (b0, "b = 0 frame (its level's reference)", "gray", None, None),
            (dw, f"b = {int(bv[kdw])} frame: shell contrast, a head pose, an eddy shear", "gray", None, None),
            (diff, f"difference to b0: contrast (tissue darkens) plus a displacement along phase encode p = ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})", "coolwarm", -float(np.abs(diff).max()), float(np.abs(diff).max()))]):
        sub = fig.add_subplot(gs[1, i])
        im = sub.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax); sub.axis("off"); sub.set_title(title, fontsize=8.5, wrap=True)
        if i == 2:
            cb = plt.colorbar(im, ax=sub, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7); cb.set_label("signal difference (a.u.)", fontsize=7)
        if i == 0:
            sub.text(0.0, -0.04, "simulated multi-shell DWI (syntx motion_simulation); eddy 0.4 mm per unit g; axial slice", transform=sub.transAxes, fontsize=7.5, va="top")

    # ---------------- B: the penalties
    axB = fig.add_subplot(gs[2, 0])
    d = 0.05; u = np.linspace(0, 0.4, 600); w = 1.0
    colors = {"charbonnier": "#1f77b4", "sparse": "#d62728", "cauchy": "#2ca02c"}
    axB2 = axB.twinx()
    for form in ("charbonnier", "sparse", "cauchy"):
        rho, om, _ = surround_rho(torch.as_tensor(u * u), w, d, form)
        axB.plot(u, rho.numpy(), color=colors[form], lw=1.8, label=f"{form}: $\\rho(u)$")
        axB2.plot(u, (om * torch.as_tensor(u)).numpy(), color=colors[form], lw=1.2, ls="--")
    axB.axvline(d, color="k", ls=":", lw=0.9); axB.text(d * 1.15, axB.get_ylim()[1] * 0.97, "knee $\\delta$ = 0.05 mm", fontsize=8, va="top")
    axB.set_xlabel("second difference of the head trajectory $|u_t|$ (mm; rotations at 50 mm)")
    axB.set_ylabel("penalty $\\rho(u)$ (solid)"); axB2.set_ylabel("pull $\\rho'(u)$ (dashed)")
    axB.set_title("B. Three surround penalties at weight 1 (curvature units)", fontsize=10, loc="left")
    axB.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62))
    axB.text(0.25, axB.get_ylim()[1] * 0.25, "cauchy: the pull peaks at the knee\nand re-descends; a real step is free", fontsize=8, color=colors["cauchy"])
    axB.text(0.25, axB.get_ylim()[1] * 0.08, "sparse: full pull already at $u \\to 0$", fontsize=8, color=colors["sparse"])

    # ---------------- C: the 1-D denoising
    axC = fig.add_subplot(gs[2, 1:])
    rng = np.random.default_rng(0); T = 120; tt = np.arange(T)
    drift = 0.6 * np.sin(2 * np.pi * tt / T) + 0.004 * tt; jitter = rng.normal(0.0, 0.02, T); jump = np.where(tt >= 80, 1.5, 0.0)
    y = drift + jitter + jump
    axC.plot(tt, y, color="#888", lw=0.8, label="observed pose: drift + sub-knee jitter + one jump")
    axC.plot(tt, drift + jump, "k--", lw=1.0, label="truth without jitter")
    for form in ("charbonnier", "sparse", "cauchy"):
        x = _denoise(y, 1.0, d, form)
        resid = x - (drift + jump); m = np.ones(T, bool); m[75:85] = False
        removed = 1 - np.var(resid[m]) / np.var(jitter[m])
        axC.plot(tt, x + 0.0, color=colors[form], lw=1.1, label=f"{form}, weight 1: {100 * removed:.0f} % of jitter energy removed")
    axC.set_xlabel("frame"); axC.set_ylabel("pose (mm)")
    axC.set_title("C. The same penalties against a unit-curvature image term on a synthetic trajectory: drift and the jump survive every form", fontsize=10, loc="left")
    axC.legend(fontsize=8, loc="upper left")
    ins = axC.inset_axes([0.62, 0.08, 0.36, 0.42])
    for form in ("charbonnier", "sparse", "cauchy"):
        x = _denoise(y, 1.0, d, form); ins.plot(tt[20:50], (x - drift)[20:50], color=colors[form], lw=1.0)
    ins.plot(tt[20:50], jitter[20:50], color="#888", lw=0.8); ins.axhline(0, color="k", lw=0.5)
    ins.set_title("residual jitter, frames 20-50 (mm)", fontsize=7.5); ins.tick_params(labelsize=7)
    f1 = os.path.join(out, "ismrm_fig1_model_and_prior.png"); fig.savefig(f1, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return f1


def figure2(gate_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = json.load(open(os.path.join(gate_dir, "phantoms.json")))
    co = json.load(open(os.path.join(gate_dir, "cohort.json")))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.1, 1.4, 1.0]})
    # A phantom gate
    ax = axes[0]
    order = ["none", "charbonnier:0.1", "sparse:0.1", "sparse:0.3", "cauchy:0.1", "cauchy:0.05", "cauchy:0.03"]
    order = [s for s in order if all(s in ph[n] for n in ph)]
    names = {"asl_drift": "ASL, drift + jitter", "dwi_eddy": "DWI, eddy + motion", "slowflow_trajectory": "slow flow, 4 states"}
    width = 0.8 / len(order)
    for i, s in enumerate(order):
        vals = [ph[n][s]["fd_error_mm"] for n in ph]
        ax.bar(np.arange(len(ph)) + i * width, vals, width, label=s, color=("#555" if s == "none" else None), edgecolor="k", lw=0.3)
    ax.set_xticks(np.arange(len(ph)) + 0.4 - width / 2); ax.set_xticklabels([names[n] for n in ph], fontsize=8)
    ax.set_ylabel("FD error against the known trajectory (mm)")
    ax.set_title("A. Phantom gate: a prior may lose at most 10 % on any phantom\nand must gain on DWI; only cauchy 0.03 and 0.05 pass", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, ncol=2)
    # B readout floor on BLAST031
    ax = axes[1]
    rec = co["dwi031"]; runs = rec["runs"]; x = np.arange(rec["T"])
    for s, c, lw in (("none", "#555", 1.0), ("cauchy:0.03", "#2ca02c", 1.2), ("cauchy:0.05", "#d62728", 1.0)):
        if s in runs:
            ax.plot(x, runs[s]["t_ro"], color=c, lw=lw, label=f"{s}: readout step MAD {runs[s]['step_mad_mm']['t_ro']:.3f} mm, FD {runs[s]['fd_mean']:.3f} mm")
    ax.set_xlabel("volume"); ax.set_ylabel("head translation along readout (mm)")
    ax.set_title(f"B. The readout-floor rule on {rec['label']}:\nthe B0-immune axis is real jitter; 0.05 smooths it, 0.03 does not", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    # C cohort response
    ax = axes[2]
    keys = list(co); labels = {"bold": "BOLD", "asl": "ASL 25", "asl60": "ASL 60", "dwi031": "DWI high motion", "dwi11": "DWI low SNR"}
    for s, c, mk in (("cauchy:0.03", "#2ca02c", "o"), ("cauchy:0.05", "#d62728", "s")):
        dro = [1000 * (co[k]["runs"][s]["step_mad_mm"]["t_ro"] - co[k]["runs"]["none"]["step_mad_mm"]["t_ro"]) for k in keys]
        dfd = [100 * (co[k]["runs"][s]["fd_mean"] / co[k]["runs"]["none"]["fd_mean"] - 1) for k in keys]
        ax.scatter(dro, dfd, color=c, marker=mk, s=55, label=s, zorder=3)
        for k, a, b in zip(keys, dro, dfd):
            ax.annotate(labels[k], (a, b), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvspan(-20, 20, color="#dfe9f3", alpha=0.6, lw=0); ax.axhspan(-100, 2, color="#e6f2df", alpha=0.5, lw=0)
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(-50, 15); ax.set_ylim(-9, 4)
    ax.set_xlabel("readout step change vs no prior (micrometres); blue band = allowed")
    ax.set_ylabel("FD mean change vs no prior (%); green = allowed")
    ax.set_title("C. Cohort gate, five real series: every point of cauchy 0.03\nlies in both bands; 0.05 leaves the readout band on one series", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    f2 = os.path.join(out, "ismrm_fig2_gate.png"); fig.tight_layout(); fig.savefig(f2, dpi=130, facecolor="white"); plt.close(fig)
    return f2


if __name__ == "__main__":
    gate_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    print(figure1(out)); print(figure2(gate_dir, out))
