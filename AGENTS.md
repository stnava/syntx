# Agent Operating Instructions — `syntx`

Read before touching any file:

1. `docs/antsx_implementation_standards.md` — the 5 cross-repo rules enforced by `tests/test_implementation_standards.py`.
2. `docs/contracts.md` — input/output, physical-space, and `antsxqc` boundary contracts.
3. `docs/tracking.md` — current phase, active work, and release history.
4. `GEMINI.md` — registration mathematical and visual reporting invariants.

## Non-Negotiable Rules

1. **Physical space first**: Never decouple pixel arrays from their `origin`, `spacing`, and `direction` cosine matrix.
2. **Single judge (`antsxqc`)**: Never hardcode QC thresholds or grade verdicts outside `antsxqc.grade()` / `antsxqc.hard_limit()` / `antsxqc.worst_status()`.
3. **Zero `.antsxqc-allow` exceptions**: `python -m antsxqc.lint src/syntx` must pass with 0 findings and no allowlist file.
4. **No silent fallbacks**: Never swallow failed warp loads or registration errors as synthetic identity pass metrics.
