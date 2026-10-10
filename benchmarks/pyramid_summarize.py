"""Tables and the mechanical application of the decision criteria of docs/PYRAMID_EVIDENCE.md to the JSON-lines written by
benchmarks/pyramid_compare.py.   python benchmarks/pyramid_summarize.py <results_dir>   (markdown to stdout)"""
import collections
import glob
import json
import os
import sys

import numpy as np

LABELS = ["DEFAULT", "P1", "P2", "P3", "P4", "P5", "P1+m", "P2+m", "P3+m", "P4+m", "P5+m"]
BASES = ["P1", "P2", "P3", "P4", "P5"]


def load(results_dir, name):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "**", name), recursive=True)):
        if "screen_invalid" in f:
            continue
        rows += [json.loads(l) for l in open(f)]
    return rows


def base(label):
    return label[:-2] if label.endswith("+m") else label


def tab(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(rd):
    verdict = {l: {} for l in LABELS if l != "DEFAULT"}
    crashes = collections.Counter()
    lines = []

    # ------------------------------------------------------------------ synthetic
    syn = load(rd, "synthetic.jsonl")
    g = collections.defaultdict(list)
    for r in syn:
        g[(r["ratio"], r["policy"])].append(r)
        if r["error"]:
            crashes[r["policy"]] += 1
    ratios = sorted({r["ratio"] for r in syn})

    def stats(ratio, lab):
        # margin variants are not run at 1:1 (no thick axis): they equal the unmargined policy there
        rs = g.get((ratio, lab)) or (g.get((ratio, base(lab))) if ratio == 1 else None) or []
        t = [r["tre"] for r in rs if r["tre"] is not None]
        return t, len(rs)

    lines.append("### Synthetic thin slab, known rigid transform: TRE in mm, mean / worst / trials under 1.5 mm\n")
    rows = []
    for ratio in ratios:
        row = [f"{ratio}:1"]
        for lab in LABELS:
            t, n = stats(ratio, lab)
            row.append(f"{np.mean(t):.2f} / {np.max(t):.2f} / {sum(x < 1.5 for x in t)} of {n}" if t else "-")
        rows.append(row)
    lines.append(tab(["ratio"] + LABELS, rows))

    for lab in verdict:
        t1, _ = stats(1, lab); d1, _ = stats(1, "DEFAULT")
        c1 = bool(t1 and d1) and np.mean(t1) <= np.mean(d1) + 0.05 and np.max(t1) <= np.max(d1) + 0.10
        verdict[lab]["C1_synthetic_1to1"] = c1
        ok2, why = True, []
        for ratio in [r for r in ratios if r > 1]:
            t, n = stats(ratio, lab); d, nd = stats(ratio, "DEFAULT")
            if not t or not d:
                ok2 = False; why.append(f"{ratio}:1 missing"); continue
            mt, md = np.mean(t), np.mean(d)
            better = (mt < md) if md >= 0.30 else (mt <= md + 0.05)
            few = sum(x < 1.5 for x in t) >= sum(x < 1.5 for x in d)
            if not (better and few):
                ok2 = False; why.append(f"{ratio}:1 {'not better' if not better else 'fewer passes'}")
            if ratio in (4, 10) and not all(x < 1.5 for x in t):
                ok2 = False; why.append(f"{ratio}:1 not all < 1.5 mm")
        verdict[lab]["C2_thin_slab"] = ok2
        verdict[lab]["C2_why"] = "; ".join(why)

    # ------------------------------------------------------------------ real
    real = load(rd, "real.jsonl")
    if real:
        lines.append("\n### Real T1-slab to NM pairs (ds001242 sub-01..10), pose error vs the ANTs rigid reference\n")
        gr = collections.defaultdict(list)
        for r in real:
            gr[r["policy"]].append(r)
            if r["error"]:
                crashes[r["policy"]] += 1
        rows = []
        for lab in LABELS:
            rs = [r for r in gr.get(lab, []) if not r["error"]]
            if not rs:
                continue
            pe = [r["pose_err_max"] for r in rs]
            lb = sum(1 for r in rs if r["loss_ants_pose"] is not None and r["loss_solved"] < r["loss_ants_pose"] - 1e-4)
            mb = sum(1 for r in rs if r["mi_solved"] < r["mi_ants"] - 1e-4)
            rows.append([lab, len(rs), f"{np.median(pe):.1f}", f"{np.mean(pe):.1f}", f"{np.max(pe):.1f}", sum(x < 3.0 for x in pe),
                         f"{sum(x < 2.0 for x in pe)}", f"{lb}", f"{mb}", f"{np.median([r['time'] for r in rs]):.1f}"])
        lines.append(tab(["policy", "pairs", "median err mm", "mean", "worst", "within 3 mm", "within 2 mm",
                          "solver loss better than at ANTs pose", "native MI better than ANTs", "median s"], rows))
        lines.append("\nPer subject pose error (mm):\n")
        subs = sorted({r["subject"] for r in real})
        rows = []
        for s in subs:
            row = [s]
            for lab in LABELS:
                r = next((x for x in gr.get(lab, []) if x["subject"] == s), None)
                row.append("-" if r is None else ("ERR" if r["error"] else f"{r['pose_err_max']:.1f}"))
            rows.append(row)
        lines.append(tab(["subject"] + LABELS, rows))
        dpe = [r["pose_err_max"] for r in gr.get("DEFAULT", []) if not r["error"]]
        for lab in verdict:
            pe = [r["pose_err_max"] for r in gr.get(lab, []) if not r["error"]]
            verdict[lab]["C4_real"] = bool(pe and dpe) and np.median(pe) <= np.median(dpe) and sum(x < 3 for x in pe) >= sum(x < 3 for x in dpe)

    # ------------------------------------------------------------------ isotropic timing / accuracy
    iso = load(rd, "isotropic.jsonl")
    if iso:
        lines.append("\n### Isotropic pairs: median wall time (s) over back-to-back repeats, and TRE (mm)\n")
        rows = []
        gi = collections.defaultdict(list)
        for r in iso:
            gi[(r["kind"], r["policy"])].append(r)
            if r["error"]:
                crashes[r["policy"]] += 1
        for kind in sorted({r["kind"] for r in iso}):
            dt = np.median([r["time"] for r in gi[(kind, "DEFAULT")]])
            dtre = [r["tre"] for r in gi[(kind, "DEFAULT")] if r["tre"] is not None]
            for lab in ["DEFAULT"] + BASES:
                rs = gi.get((kind, lab), [])
                t = np.median([r["time"] for r in rs]) if rs else float("nan")
                tr = [r["tre"] for r in rs if r["tre"] is not None]
                rows.append([kind, lab, len(rs), f"{t:.2f}", f"{t / dt:.2f}", f"{np.mean(tr):.3f}" if tr else "-", f"{np.max(tr):.3f}" if tr else "-"])
                if lab != "DEFAULT":
                    for l2 in (lab, lab + "+m"):
                        verdict[l2].setdefault("C3_time", True)
                        verdict[l2]["C3_time"] &= bool(rs) and t <= 1.10 * dt
                        verdict[l2].setdefault("C1_iso_pairs", True)
                        verdict[l2]["C1_iso_pairs"] &= bool(tr and dtre) and np.mean(tr) <= np.mean(dtre) + 0.05 and np.max(tr) <= np.max(dtre) + 0.10
        lines.append(tab(["pair", "policy", "runs", "median s", "x DEFAULT", "mean TRE", "worst TRE"], rows))

    # ------------------------------------------------------------------ motion
    mot = load(rd, "motion.jsonl")
    if mot:
        lines.append("\n### motion_batched, known transforms: TRE mm (mean over frames / worst frame), median seconds\n")
        gm = collections.defaultdict(list)
        for r in mot:
            gm[(r["case"], r["policy"])].append(r)
            if r["error"]:
                crashes[r["policy"]] += 1
        cases = sorted({r["case"] for r in mot})
        rows = []
        for c in cases:
            for lab in ["DEFAULT"] + BASES:
                rs = [r for r in gm.get((c, lab), []) if not r["error"]]
                nerr = len(gm.get((c, lab), [])) - len(rs)
                if rs:
                    rows.append([c, lab, f"{np.mean([r['tre_mean'] for r in rs]):.2f} / {np.max([r['tre_max'] for r in rs]):.2f}",
                                 f"{np.median([r['time'] for r in rs]):.1f}", nerr])
                else:
                    rows.append([c, lab, "-", "-", nerr])
        lines.append(tab(["case", "policy", "TRE mean / worst", "median s", "errors"], rows))
        for lab in verdict:
            ok = True
            for c in cases:
                d = [r for r in gm.get((c, "DEFAULT"), []) if not r["error"]]
                p = [r for r in gm.get((c, base(lab)), []) if not r["error"]]
                if not d or not p or np.mean([r["tre_mean"] for r in p]) > np.mean([r["tre_mean"] for r in d]) + 0.05:
                    ok = False
            verdict[lab]["C5_motion"] = ok

    # ------------------------------------------------------------------ properties
    prop = load(rd, "properties.jsonl")
    if prop:
        gp = collections.defaultdict(list)
        for r in prop:
            if "shape" in r and "out_shape" in r or r.get("ok") is False:
                gp[r["policy"]].append(r)
        lines.append("\n### Builder properties (shapes (3,64,64), (11,128,128), (11,256,256), (64,64,64), (7,33,65), (16,96,96), (2,40,40); levels 1,2,4,8)\n")
        rows = []
        for lab in BASES:
            rs = [r for r in prop if r.get("policy") == lab and "level" in r and "perf_shape" not in r]
            okr = [r for r in rs if r["ok"]]
            bad = [r for r in rs if not r["ok"]]
            thin_bad = sorted({tuple(r["shape"]) for r in bad})
            rows.append([lab, f"{len(okr)} of {len(rs)}", "; ".join(f"{s}" for s in thin_bad) or "none",
                         sum(1 for r in okr if r["min_axis"] < 2), sum(1 for r in okr if not r["finite"]),
                         sum(1 for r in okr if not r["const_preserved"]), f"{max([r['ramp_err_vox'] for r in okr] or [0]):.3f}",
                         sum(1 for r in okr if not r["deterministic"]), f"{max([r['cpu_vs_dev_maxabs'] for r in okr] or [0]):.1e}"])
            c6 = bool(okr) and all(r["finite"] and r["const_preserved"] and r["deterministic"] and r["ramp_err_vox"] <= 0.05 and r["cpu_vs_dev_maxabs"] <= 1e-4 and r["min_axis"] >= 2 for r in okr)
            for l2 in (lab, lab + "+m"):
                verdict[l2]["C6_properties"] = c6
        lines.append(tab(["policy", "built", "shapes that raise (no solver clamp)", "axis < 2", "NaN", "constant not preserved",
                          "max ramp error (working voxels, interior)", "non-deterministic", "max CPU-vs-MPS abs diff"], rows))
        lines.append("\nRaw shipping pyramids on a thin axis (no solver clamp):\n")
        rows = [[r["policy"], r["shape"], r["level"], r["out_shape"] if r["ok"] else r["error"][:80]] for r in prop if r.get("policy", "").startswith("build_")]
        lines.append(tab(["function", "shape", "level", "result"], rows))
        lines.append("\nTime and allocation, volume (176, 224, 192), MPS time / CPU total allocation:\n")
        rows = [[r["policy"], r["level"], f"{r['ms_median']:.2f}", r["out_voxels"], f"{r['cpu_alloc_total_mb']:.0f}"] for r in prop if "perf_shape" in r and r["ok"]]
        lines.append(tab(["policy", "level", "ms (median of 5)", "output voxels", "CPU MB allocated"], rows))


    # ------------------------------------------------------------------ deformable characterisation
    de = load(rd, "deformable.jsonl")
    if de:
        lines.append("\n### Deformable-model pyramids, characterised (isotropic volumes; RMSE of each level mapped back to the full grid, vs P1)\n")
        rows = []
        for vol in sorted({r["volume"] for r in de}):
            for lev in sorted({r["level"] for r in de if r["volume"] == vol}, reverse=True):
                for r in [x for x in de if x["volume"] == vol and x["level"] == lev and "policy" in x]:
                    d = next((x for x in de if x["volume"] == vol and x["level"] == lev and x.get("pair") == ["P1", r["policy"]]), None)
                    rows.append([vol, lev, r["policy"], r["out_shape"], f"{r['mean']:.3f}", f"{r['std']:.3f}", f"{r['vmax']:.3f}",
                                 "-" if d is None else f"{d['rmse_after_mapping_to_full_grid']:.4f}"])
        lines.append(tab(["volume", "level", "policy", "shape (z,y,x)", "mean", "std", "max", "RMSE vs P1"], rows))

    # ------------------------------------------------------------------ verdict
    lines.append("\n### Criteria applied mechanically\n")
    rows = []
    winners = []
    for lab in verdict:
        v = verdict[lab]
        v["C0_crashes"] = crashes[lab] == 0
        order = ["C0_crashes", "C1_synthetic_1to1", "C1_iso_pairs", "C2_thin_slab", "C3_time", "C4_real", "C5_motion", "C6_properties"]
        flags = [("pass" if v.get(k) else ("FAIL" if k in v else "n/a")) for k in order]
        allok = all(v.get(k) for k in order if k in v) and all(k in v for k in order)
        rows.append([lab] + flags + ["ALL PASS" if allok else (v.get("C2_why") or "")])
        if allok:
            winners.append(lab)
    lines.append(tab(["variant", "C0", "C1 synth 1:1", "C1 iso pairs", "C2 thin slab", "C3 time", "C4 real", "C5 motion", "C6 props", "result / C2 detail"], rows))
    lines.append("\nVariants passing all criteria: " + (", ".join(winners) if winners else "none"))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
