"""Markdown table for the ISMRM abstract from an ``ismrm_cases.py`` JSON (numbers come from the analysis output, not typed).

Usage: python benchmarks/ismrm_table.py <ismrm_cases.json>
"""
import json
import sys

LABELS = {"bold": "resting BOLD, TR 1.0 s", "asl": "ASL, control/label + M0, 25 vol", "asl60": "ASL, control/label, 60 vol",
          "dwi031": "DWI multi-shell, high motion", "dwi11": "DWI multi-shell, low SNR"}


def rows(d):
    out = []
    for key in ("bold", "asl", "asl60", "dwi031", "dwi11"):
        if key not in d:
            continue
        r = d[key]
        ax = r["pose"]["axes"]
        ro = ax["readout"]; pe = ax["phase_encode"]
        sh = lambda a: f"{100 * a['head']['share']:.0f} / {100 * a['applied_acquisition_term']['share']:.0f} / {100 * a['removed_design_locked_term']['share']:.0f}"
        def frac(img, name):
            for t in img["terms"]:
                if t["term"].startswith(name):
                    return 100 * t["fraction"]
            return float("nan")
        def design(img):
            return sum(frac(img, n) for n in ("level intercepts", "l=2", "l=4") if any(t["term"].startswith(n) for t in img["terms"]))
        raw, cor = r["image_raw"], r["image_corrected"]
        rcor = cor.get("residual_rms_fd_corr", float("nan"))
        cens = f"{100 * r['frac_gt_thr_raw']:.0f} % / {100 * r['frac_gt_thr_bandlimited']:.0f} %"
        out.append(f"| {LABELS[key]} | {r['T']} | {r['fd_mean']:.2f} / {r['fd_max']:.2f} | {cens} | {sh(ro)} | {sh(pe)} | "
                   f"{design(raw):.1f} / {design(cor):.1f} | {frac(raw, 'motion'):.2f} / {frac(cor, 'motion'):.2f} | "
                   f"{frac(raw, 'residual'):.1f} / {frac(cor, 'residual'):.1f} | {rcor:.2f} | {r['seconds']:.0f} s |")
    return out


def latex_rows(d):
    """The same rows for the LaTeX table (``&`` separated, ``\\\\`` terminated, percent signs escaped)."""
    out = []
    for r in rows(d):
        cells = [c.strip() for c in r.strip("|").split("|")]
        cells[3] = cells[3].replace(" %", "")          # the header carries the unit
        out.append(" & ".join(c.replace("%", "\\%") for c in cells) + " \\\\")
    return out


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    if "--latex" in sys.argv:
        print("\n".join(latex_rows(d)))
        raise SystemExit
    print("| case | T | FD mean / max (mm) | censored raw / band-limited | readout: head / eddy / design (%) | PE: head / eddy / design (%) | design (%) raw / corr | motion after design (%) raw / corr | residual (%) raw / corr | r(resid, FD) | time |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows(d)))
