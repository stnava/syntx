# syntx 6.8.1

Docs only: `docs/MOTION_BOLD.md` now reports the ASL check on five datasets (PTBP PCASL, breacher sub-9162 x2, SOCOM x2) and
supersedes the 6.8.0 statement that ASL should stay on the default backend. The `bold` preset beats the default adaptive
backend on injected known motion in all five (error 0.09-0.48 mm vs 0.24-0.77 mm; FD estimate +15-91% vs +50-115%) and
reads lower FD on ten real series; ASL remains less accurate than BOLD (+8%). No code change.
