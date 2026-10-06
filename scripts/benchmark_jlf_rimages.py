"""
benchmark_jlf_rimages.py - validate and time the new syntx joint label fusion.

Validation case: the 2-D ANTs sample images r16 r27 r30 r62 r64 r85 (256x256). Ground truth is the
3-class Otsu (primary) or k-means (sensitivity) segmentation of the *target* image; atlas labels are
the same segmentation of each other image, carried through registration. Leave-one-out: 6 targets
x 5 atlases. Two regimes: A = affine + SyN, B = affine only (large residual misregistration).

NOTE: truth is a function of the target intensity, so patch-similarity fusion has a built-in
advantage. This validates algorithmic behaviour, not anatomical accuracy.

Stages (separate processes; ``syntx.robust_affine`` pins ITK to one thread, which would bias ANTs):
  register : cache warped atlas images / labels / truth        (python ... register)
  fuse     : all arms on the cached atlases, both regimes      (python ... fuse)
  timing3d : synthetic 3-D phantom, old (HEAD) vs new, ANTs    (python ... timing3d)
  report   : self-contained HTML                               (python ... report)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import importlib.util
import io
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results", "jlf_rimages")
REF = os.path.join(OUT, "ref", "jlf_head.py")  # frozen copy of the committed module (git show HEAD:...)
IMAGES = ["r16", "r27", "r30", "r62", "r64", "r85"]
TRUTHS = ("otsu", "kmeans")


def _norm(img):
    a = img.numpy()
    fg = a[a > 0]
    lo, hi = np.percentile(fg, [2, 98])
    return img.new_image_like((np.clip((a - lo) / (hi - lo), 0, 1) * (a > 0)).astype(np.float32))


def _load(name):
    import ants
    return _norm(ants.image_read(ants.get_ants_data(name)))


def _segment(img, kind):
    """3-class segmentation inside the head mask; background = 0."""
    import ants
    mask = ants.threshold_image(img, 1e-6, 1e9)
    if kind == "otsu":
        seg = ants.otsu_segmentation(img, 3, mask)
    else:
        seg = ants.kmeans_segmentation(img, 3, mask, mrf=0.0)["segmentation"]
    return seg.new_image_like(seg.numpy().astype(np.uint32) * (mask.numpy() > 0))


def _head_module():
    spec = importlib.util.spec_from_file_location("jlf_head", REF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cache_path(t, a, regime, what):
    return os.path.join(OUT, "cache", f"{t}__{a}__{regime}__{what}.nii.gz")


def run_register(args):
    import ants
    import syntx
    import shutil
    tdir = os.path.join(OUT, "cache", "transforms")
    os.makedirs(tdir, exist_ok=True)
    imgs = {n: _load(n) for n in IMAGES}
    log = []
    for t in IMAGES:
        for a in IMAGES:
            if a == t:
                continue
            rec_path = os.path.join(OUT, "cache", f"{t}__{a}__registration.json")
            if os.path.exists(rec_path):  # registration results are on disk: do not redo
                log.append(json.load(open(rec_path)))
                continue
            fixed, mov = imgs[t], imgs[a]
            t0 = time.perf_counter()
            aff = syntx.robust_affine(fixed, mov, mode="auto", verbose=False)
            t_aff = time.perf_counter() - t0
            t0 = time.perf_counter()
            syn = syntx.syn(fixed, mov, initial_transform=aff["fwdtransforms"][0], verbose=False)
            t_syn = time.perf_counter() - t0
            saved = {}
            for regime, fwd in (("A", syn["fwdtransforms"]), ("B", aff["fwdtransforms"])):
                kept = []
                for i, src in enumerate(fwd):
                    ext = ".nii.gz" if src.endswith(".nii.gz") else os.path.splitext(src)[1]
                    dst = os.path.join(tdir, f"{t}__{a}__{regime}__{i}{ext}")
                    shutil.copy(src, dst)
                    kept.append(dst)
                saved[regime] = kept
                w = syntx.apply_transform_chain(moving=mov, fixed=fixed, transformlist=kept, interpolator="linear")
                ants.image_write(w.warped_image, cache_path(t, a, regime, "img"))
                for kind in TRUTHS:
                    lab = _segment(mov, kind)
                    wl = syntx.apply_transform_chain(moving=lab, fixed=fixed, transformlist=kept,
                                                     interpolator="nearestNeighbor")
                    ants.image_write(wl.warped_image, cache_path(t, a, regime, f"lab_{kind}"))
            rec = dict(target=t, atlas=a, t_affine_s=t_aff, t_syn_s=t_syn, transforms=saved,
                       device=str(syn.get("device", "?")) if hasattr(syn, "get") else "?")
            json.dump(rec, open(rec_path, "w"), indent=1)  # written last: marks the pair complete
            log.append(rec)
            print(json.dumps({k: rec[k] for k in ("target", "atlas", "t_affine_s", "t_syn_s")}), flush=True)
    for t in IMAGES:
        for kind in TRUTHS:
            ants.image_write(_segment(imgs[t], kind), os.path.join(OUT, "cache", f"{t}__truth_{kind}.nii.gz"))
    with open(os.path.join(OUT, "register_log.json"), "w") as f:
        json.dump(log, f, indent=1)


def _dice_table(truth, seg):
    import ants
    df = ants.label_overlap_measures(truth, seg)
    df = df[(df["Label"] != "All") & (df["Label"].astype(str) != "0")]
    return {str(int(float(l))): float(v) for l, v in zip(df["Label"], df["MeanOverlap"]) if 0.0 <= float(v) <= 1.0}


def _boundary_dice(truth, seg, width=2):
    """Mean Dice restricted to voxels within ``width`` voxels of a truth class boundary."""
    import scipy.ndimage as ndi
    t, s = truth.numpy().astype(int), seg.numpy().astype(int)
    edge = np.zeros(t.shape, dtype=bool)
    for ax in range(t.ndim):
        d = np.diff(t, axis=ax) != 0
        sl_a = [slice(None)] * t.ndim; sl_a[ax] = slice(0, -1)
        sl_b = [slice(None)] * t.ndim; sl_b[ax] = slice(1, None)
        edge[tuple(sl_a)] |= d; edge[tuple(sl_b)] |= d
    band = ndi.binary_dilation(edge, iterations=width)
    ds = []
    for c in np.unique(t[band]):
        if c == 0:
            continue
        a, b = (t[band] == c), (s[band] == c)
        den = a.sum() + b.sum()
        if den:
            ds.append(2.0 * (a & b).sum() / den)
    return float(np.mean(ds)) if ds else float("nan")


def _majority(labels):
    stack = np.stack([l.numpy().astype(np.int64) for l in labels])
    C = int(stack.max()) + 1
    counts = np.stack([(stack == c).sum(0) for c in range(C)])
    return labels[0].new_image_like(counts.argmax(0).astype(np.uint32))  # ties -> lowest label


def _arms():
    arms = [dict(name="majority_vote", engine="mv"), dict(name="atlas0_only", engine="a0")]
    for mode in ("joint", "patch"):  # frozen HEAD module, its own defaults of the time (beta 2, rho .05)
        arms.append(dict(name=f"head_{mode}", engine="head", mode=mode, beta=2.0, rho=0.05, rad=2))
    arms.append(dict(name="new_patch", engine="new", mode="patch", beta=2.0, rho=0.01, rad=2, r_search=0))
    for beta in (2.0, 4.0):
        arms.append(dict(name=f"new_joint_b{int(beta)}", engine="new", mode="joint", beta=beta, rho=0.01, rad=2, r_search=0))
    arms.append(dict(name="new_joint_b4_stdresid", engine="new", mode="joint", beta=4.0, rho=0.01, rad=2, r_search=0,
                     extra=dict(patch_norm=True)))
    for rs in (1, 2, 3):
        arms.append(dict(name=f"new_joint_b4_rs{rs}", engine="new", mode="joint", beta=4.0, rho=0.01, rad=2, r_search=rs))
    arms.append(dict(name="new_joint_b4_rs3_stdresid", engine="new", mode="joint", beta=4.0, rho=0.01, rad=2, r_search=3,
                     extra=dict(patch_norm=True)))
    arms.append(dict(name="new_patch_rs3", engine="new", mode="patch", beta=2.0, rho=0.01, rad=2, r_search=3))
    arms.append(dict(name="ants_rs0", engine="ants", rs=0))
    arms.append(dict(name="ants_rs3", engine="ants", rs=3))
    return arms


def _run_arm(arm, head, target, atlases, labels, mask, device):
    import ants
    import syntx
    t0 = time.perf_counter()
    if arm["engine"] == "mv":
        seg = _majority(labels)
    elif arm["engine"] == "a0":
        seg = labels[0]
    elif arm["engine"] == "head":
        r = head.joint_label_fusion(target, atlases, labels, mask=mask, rad=arm["rad"], beta=arm["beta"],
                                    rho=arm["rho"], mode=arm["mode"], device=None)
        seg = r["segmentation"]
    elif arm["engine"] == "new":
        r = syntx.joint_label_fusion(target, atlases, labels, mask=mask, rad=arm["rad"], beta=arm["beta"],
                                     rho=arm["rho"], mode=arm["mode"], r_search=arm["r_search"], device=device,
                                     **arm.get("extra", {}))
        seg = r["segmentation"]
    else:  # ANTs C++ reference; workaround for the ANTsPy no-background segmentation (see docs)
        r = ants.joint_label_fusion(target, mask, atlases, label_list=[l.clone("unsigned int") for l in labels],
                                    rad=[2] * target.dimension, beta=4, rho=0.01, r_search=arm["rs"],
                                    max_lab_plus_one=True)
        seg = r["segmentation"]
    sec = time.perf_counter() - t0
    seg = (seg * mask)
    return seg.new_image_like(seg.numpy().astype(np.uint32)), sec


def run_fuse(args):
    import ants
    import torch
    import syntx
    from syntx.deformation_metrics import compute_bidirectional_dice
    head = _head_module()
    acc = "mps" if torch.backends.mps.is_available() else "cpu"
    res = dict(meta=dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), host=platform.platform(),
                         torch=torch.__version__, accel=acc, argv=sys.argv, repeats=args.repeats,
                         itk_threads=os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "default")),
               cases={})
    arms = _arms()
    for kind in TRUTHS:
        for regime in ("A", "B"):
            for ti, t in enumerate(IMAGES):
                target = _load(t)
                mask = ants.threshold_image(target, 1e-6, 1e9)
                truth = ants.image_read(os.path.join(OUT, "cache", f"{t}__truth_{kind}.nii.gz"))
                names = [a for a in IMAGES if a != t]
                atlases = [ants.image_read(cache_path(t, a, regime, "img")) for a in names]
                labels = [ants.image_read(cache_path(t, a, regime, f"lab_{kind}")) for a in names]
                rec = {}
                segs = {}
                for arm in arms:
                    devs = [acc, "cpu"] if arm["engine"] == "new" else ["cpu"]
                    for dev in devs:
                        timing_case = (kind == "otsu" and regime == "A" and ti == 0)
                        nrep = args.repeats if timing_case else 1
                        _run_arm(arm, head, target, atlases, labels, mask, dev)  # warm-up, discarded
                        times = []
                        for _ in range(nrep):
                            seg, sec = _run_arm(arm, head, target, atlases, labels, mask, dev)
                            times.append(sec)
                        key = arm["name"] + ("" if dev == acc or arm["engine"] != "new" else "@cpu")
                        per = _dice_table(truth, seg)
                        d_eval = compute_bidirectional_dice(seg, truth, target, target, [], [], [])[0]
                        assert abs(d_eval - np.mean(list(per.values()))) < 1e-6, (key, d_eval)
                        rec[key] = dict(mean_dice=float(np.mean(list(per.values()))), per_label=per,
                                        boundary_dice=_boundary_dice(truth, seg), times_s=times, device=dev)
                        segs[key] = seg
                        print(f"[{kind}/{regime}/{t}] {key:22s} dice={rec[key]['mean_dice']:.4f} "
                              f"bnd={rec[key]['boundary_dice']:.4f} t={np.median(times) * 1000:.0f}ms", flush=True)
                # exactness: new patch == frozen HEAD patch (same model); new joint fast path == dense LU path
                h = head.joint_label_fusion(target, atlases, labels, mask=mask, rad=2, beta=2.0, rho=0.05,
                                            mode="patch", device=None)["segmentation"].numpy()
                n = syntx.joint_label_fusion(target, atlases, labels, mask=mask, rad=2, beta=2.0, rho=0.01,
                                             mode="patch", device="cpu", solver="lu")["segmentation"].numpy()
                rec["_new_vs_head_patch_label_diff_frac"] = float(np.mean(h != n))
                f = syntx.joint_label_fusion(target, atlases, labels, mask=mask, rad=2, beta=4.0, rho=0.01,
                                             mode="joint", device="cpu")["segmentation"].numpy()
                d = syntx.joint_label_fusion(target, atlases, labels, mask=mask, rad=2, beta=4.0, rho=0.01,
                                             mode="joint", device="cpu", solver="lu",
                                             skip_consensus=False)["segmentation"].numpy()
                rec["_fast_vs_dense_joint_label_diff_frac"] = float(np.mean(f != d))
                res["cases"][f"{kind}/{regime}/{t}"] = rec
                if kind == "otsu" and regime == "B" and ti == 0:
                    for k in ("majority_vote", "head_joint", "new_joint_b4", "new_joint_b4_rs3"):
                        ants.plot(target, overlay=segs[k], axis=0 if target.dimension == 3 else None,
                                  overlay_alpha=0.5, filename=os.path.join(OUT, f"fig_{k}.png")) \
                            if target.dimension == 3 else ants.plot(target, overlay=segs[k], overlay_alpha=0.5,
                                                                    filename=os.path.join(OUT, f"fig_{k}.png"))
                    ants.plot(target, overlay=truth, overlay_alpha=0.5, filename=os.path.join(OUT, "fig_truth.png"))
    res["meta"]["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


def _phantom3d(n, K, seed=0):
    """Textured 3-D head-like phantom: nested shells (labels 1..4) + K atlases with shifted, noisy labels."""
    import ants
    import scipy.ndimage as ndi
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n)] * 3, indexing="ij"))
    r = np.sqrt((g ** 2 * np.array([1.0, 1.2, 1.5]).reshape(3, 1, 1, 1)).sum(0))
    base = np.zeros((n, n, n), np.uint32)
    for lab, (lo, hi) in enumerate([(0.0, 0.3), (0.3, 0.55), (0.55, 0.75), (0.75, 0.9)], 1):
        base[(r >= lo) & (r < hi)] = lab
    tex = ndi.gaussian_filter(rng.normal(0, 1, (n, n, n)), 1.5)
    inten = (0.2 * (base > 0) + 0.15 * base + 0.05 * tex).astype(np.float32)
    inten = (inten - inten.min()) / (inten.max() - inten.min())
    target = ants.from_numpy(inten, spacing=(1.0, 1.0, 1.0))
    atlases, labels = [], []
    for i in range(K):
        disp = [ndi.gaussian_filter(rng.normal(0, 1, (n, n, n)), 6) * 12 for _ in range(3)]
        coords = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij")).astype(np.float32) + np.stack(disp)
        lab = ndi.map_coordinates(base, coords, order=0, mode="nearest").astype(np.uint32)
        img = ndi.map_coordinates(inten, coords, order=1, mode="nearest") + rng.normal(0, 0.02, (n, n, n))
        atlases.append(ants.from_numpy(img.astype(np.float32), spacing=(1.0, 1.0, 1.0)))
        labels.append(ants.from_numpy(lab, spacing=(1.0, 1.0, 1.0)))
    return target, atlases, labels, ants.from_numpy(base, spacing=(1.0, 1.0, 1.0))


def run_timing3d(args):
    import ants
    import torch
    import syntx
    head = _head_module()
    acc = "mps" if torch.backends.mps.is_available() else "cpu"
    out = dict(meta=dict(host=platform.platform(), accel=acc, torch=torch.__version__, repeats=args.repeats,
                         started_utc=dt.datetime.now(dt.timezone.utc).isoformat()), rows=[])
    K = 8
    for n in (96, 160):
        target, atlases, labels, truth = _phantom3d(n, K)
        mask = ants.threshold_image(target, 0.0, 1e9) * 0 + 1
        cases = []
        for rs in (0, 2):
            if rs == 0:
                cases.append(("head_patch", dict(engine="head", mode="patch", rs=0, dev="cpu")))
                cases.append(("head_joint(old surrogate)", dict(engine="head", mode="joint", rs=0, dev="cpu")))
                cases.append(("new_joint_dense_lu@cpu", dict(engine="new", mode="joint", rs=0, dev="cpu",
                                                              extra=dict(skip_consensus=False, solver="lu"))))
            for dev in (acc, "cpu"):
                cases.append((f"new_patch@{dev}", dict(engine="new", mode="patch", rs=rs, dev=dev)))
                cases.append((f"new_joint@{dev}", dict(engine="new", mode="joint", rs=rs, dev=dev)))
            cases.append((f"ants_rs{rs}", dict(engine="ants", rs=rs, dev="cpu")))
        for name, c in cases:
            times, seg, peak, extra = [], None, None, {}
            for i in range(args.repeats + 1):
                t0 = time.perf_counter()
                if c["engine"] == "head":
                    r = head.joint_label_fusion(target, atlases, labels, mask=mask, rad=2, beta=2.0, rho=0.05,
                                                mode=c["mode"], device=None)
                    seg = r["segmentation"]
                elif c["engine"] == "new":
                    r = syntx.joint_label_fusion(target, atlases, labels, mask=mask, rad=2,
                                                 beta=4.0 if c["mode"] == "joint" else 2.0, rho=0.01,
                                                 mode=c["mode"], r_search=c["rs"], device=c["dev"],
                                                 return_probabilities=False, **c.get("extra", {}))
                    seg = r["segmentation"]
                    extra = dict(disagree=r["disagreement_fraction"], stages=r["stage_timings_s"])
                else:
                    r = ants.joint_label_fusion(target, mask, atlases, label_list=[l.clone("unsigned int") for l in labels],
                                                rad=[2] * 3, beta=4, rho=0.01, r_search=c["rs"], max_lab_plus_one=True)
                    seg = r["segmentation"]
                dt_s = time.perf_counter() - t0
                if i > 0:  # first call is a warm-up
                    times.append(dt_s)
            d = _dice_table(truth, seg.new_image_like(seg.numpy().astype(np.uint32)))
            row = dict(n=n, K=K, name=name, r_search=c["rs"], times_s=times, mean_dice=float(np.mean(list(d.values()))), **extra)
            out["rows"].append(row)
            print(f"[{n}^3 rs={c['rs']}] {name:18s} median={np.median(times):.2f}s dice={row['mean_dice']:.4f}", flush=True)
    with open(os.path.join(OUT, "timing3d.json"), "w") as f:
        json.dump(out, f, indent=1)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    return base64.b64encode(buf.getvalue()).decode()


def run_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = json.load(open(os.path.join(OUT, "results.json")))
    T3 = json.load(open(os.path.join(OUT, "timing3d.json"))) if os.path.exists(os.path.join(OUT, "timing3d.json")) else None
    C = R["cases"]

    def avg(kind, regime, arm, field="mean_dice"):
        return float(np.mean([C[f"{kind}/{regime}/{t}"][arm][field] for t in IMAGES]))

    names = [k for k in C["otsu/A/r16"] if not k.startswith("_") and "@cpu" not in k]
    html_tabs = ""
    figs = ""
    for kind in TRUTHS:
        for regime in ("A", "B"):
            fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
            v = [avg(kind, regime, k) for k in names]
            ax.barh(names, v, color="#06b6d4")
            ax.set_xlim(min(v) - 0.02, 1.0)
            ax.invert_yaxis()
            ax.set_title(f"{kind} truth, regime {regime}: mean Dice over 6 targets", color="#1e293b")
            plt.tight_layout()
            figs += f"<img src='data:image/png;base64,{_png(fig)}' width=560>"
            plt.close(fig)
            rows = ""
            ref = avg(kind, regime, "head_joint")
            for k in names:
                per = " ".join(f"{C[f'{kind}/{regime}/{t}'][k]['mean_dice']:.3f}" for t in IMAGES)
                rows += (f"<tr><td>{k}</td><td>{avg(kind, regime, k):.4f}</td>"
                         f"<td>{avg(kind, regime, k) - ref:+.4f}</td><td>{avg(kind, regime, k, 'boundary_dice'):.4f}</td>"
                         f"<td>{per}</td></tr>")
            html_tabs += (f"<h3>{kind} / regime {regime}</h3><table><tr><th>arm</th><th>mean Dice</th><th>vs head_joint</th>"
                          f"<th>boundary Dice</th><th>per target ({' '.join(IMAGES)})</th></tr>{rows}</table>")
    # timing on r-images (target r16, otsu, A)
    base = C["otsu/A/r16"]
    trows = ""
    for k in base:
        if k.startswith("_"):
            continue
        ts = base[k]["times_s"]
        trows += f"<tr><td>{k}</td><td>{base[k]['device']}</td><td>{min(ts) * 1e3:.0f} / {np.median(ts) * 1e3:.0f} / {max(ts) * 1e3:.0f}</td></tr>"
    exact = ""
    for key, rec in C.items():
        exact += (f"<tr><td>{key}</td><td>{rec['_fast_vs_dense_joint_label_diff_frac']:.5f}</td>"
                  f"<td>{rec['_new_vs_head_patch_label_diff_frac']:.5f}</td></tr>")
    t3 = ""
    if T3:
        t3rows = ""
        for r in T3["rows"]:
            ts = r["times_s"]
            st = r.get("stages")
            t3rows += (f"<tr><td>{r['n']}^3</td><td>{r['r_search']}</td><td>{r['name']}</td>"
                       f"<td>{min(ts):.2f} / {np.median(ts):.2f} / {max(ts):.2f}</td><td>{r['mean_dice']:.4f}</td>"
                       f"<td>{'' if not r.get('disagree') else format(r['disagree'], '.3f')}</td>"
                       f"<td>{'' if not st else ' / '.join(f'{v:.2f}' for v in st.values())}</td></tr>")
        t3 = ("<h2>3-D timing (synthetic phantom, timing only)</h2><table><tr><th>size</th><th>r_search</th><th>arm</th>"
              "<th>s min/med/max</th><th>Dice vs phantom truth</th><th>disagree frac</th><th>search/weights/vote s</th></tr>"
              f"{t3rows}</table>")
    ims = "".join(f"<figure><img src='data:image/png;base64,{base64.b64encode(open(os.path.join(OUT, f'fig_{k}.png'), 'rb').read()).decode()}' width=300><figcaption>{k}</figcaption></figure>"
                  for k in ("truth", "majority_vote", "head_joint", "new_joint_b4", "new_joint_b4_rs3")
                  if os.path.exists(os.path.join(OUT, f"fig_{k}.png")))
    m = R["meta"]
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>syntx JLF: r-image validation</title>
<style>body{{background:#fff;color:#1e293b;font-family:-apple-system,Helvetica,sans-serif;max-width:1150px;margin:2em auto}}
table{{border-collapse:collapse;font-size:12.5px;margin-bottom:1em}}td,th{{border:1px solid #cbd5e1;padding:3px 7px}}th{{background:#f1f5f9}}
figure{{display:inline-block;margin:4px}}</style></head><body>
<h1>syntx joint label fusion: r-image validation</h1>
<p>Targets r16 r27 r30 r62 r64 r85, leave-one-out, 5 atlases. Truth = 3-class Otsu / k-means of the target (a function of target intensity: favours patch similarity; validates behaviour, not anatomy).
A = affine+SyN, B = affine only. Host {m['host']}; accelerator {m['accel']}; torch {m['torch']}; {m['started_utc']} to {m['finished_utc']}.
Timings n={m['repeats']} (+1 discarded warm-up) on r16/otsu/A, run alone.</p>
{figs}{html_tabs}
<h2>Timing, r16 target (ms, min / median / max)</h2><table><tr><th>arm</th><th>device</th><th>ms</th></tr>{trows}</table>
<h2>Exactness: fraction of voxels with a different label</h2>
<p>joint column: new fast path (consensus skipping, Cholesky) vs new dense LU path (same model). patch column: new vs frozen HEAD patch mode (same model).
<code>head_joint</code> is a different model (old LNCC surrogate), so it is a baseline for accuracy only, not for exactness.</p>
<table><tr><th>case</th><th>joint fast vs dense</th><th>patch new vs HEAD</th></tr>{exact}</table>
{t3}
<h2>Example segmentations (r16, otsu, regime B)</h2>{ims}
<p><b>Caveats.</b> ANTs arms use max_lab_plus_one=True on cloned labels (ANTsPy otherwise labels every mask voxel
foreground). Truth derives from target intensity, so patch-similarity methods have a built-in advantage; this validates behaviour, not anatomy.
Arms *_stdresid standardise each patch before forming residuals (patch_norm=True) (ablation).</p></body></html>"""
    out = os.path.join(ROOT, "docs", "reports", "jlf_rimages.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["register", "fuse", "timing3d", "report"])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"register": run_register, "fuse": run_fuse, "timing3d": run_timing3d, "report": run_report}[a.stage](a)
