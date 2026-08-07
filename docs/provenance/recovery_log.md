# TVF DSTI Recovery Log

| Phase | Date | Code State | Config | Dice_sym | Fold% | J_min | V_max | Time | Verdict |
|-------|------|------------|--------|----------|-------|-------|-------|------|---------|
| Prereq | 2026-08-06 | Stashed uncommitted diff (197 lines) | — | — | — | — | — | — | DONE |
| 1 | 2026-08-06 | HEAD (committed) | dsti→gaussian fallthrough | 0.6770 | 0.0000% | 0.9183 | 0.40 | 11.6s | PASS ✓ |
| 1 | 2026-08-06 | HEAD (committed) | gaussian explicit | 0.6770 | 0.0000% | 0.9183 | 0.40 | 13.4s | BASELINE_2D=0.6770 |
| 2 | 2026-08-06 | HEAD + momentum fix | new momentum, gs=0.45 | 0.6465 | 0.0000% | 0.8929 | 0.40 | 16.9s | -0.031 drop |
| 2 | 2026-08-06 | HEAD + momentum fix | new momentum, gs=0.90 | 0.6771 | 0.0000% | 0.9146 | 0.39 | 18.6s | **ACCEPT** gs=0.90 |
| 2 | 2026-08-06 | HEAD + momentum fix | new momentum, gs=3.60 | 0.6770 | 0.0000% | 0.9069 | 0.39 | 16.8s | matches |
| 2 | 2026-08-06 | HEAD + momentum fix | new momentum, gs=7.20 | 0.6770 | 0.0000% | 0.9184 | 0.39 | 15.7s | matches |
| 3 | 2026-08-06 | HEAD + mom + no cfl_max | no cfl_max, gs=0.90 | **0.8262** | 0.0000% | 0.0411 | 37.68 | 29.7s | **ACCEPT** +0.149 |
| 3 | 2026-08-06 | HEAD + mom + no cfl_max | no cfl_max, gs=0.45 | 0.8231 | 0.0000% | 0.0685 | 23.04 | 19.1s | +0.146 |
| 3 | 2026-08-06 | HEAD + mom + no cfl_max | no cfl_max, gs=0.25 | 0.8171 | 0.0000% | 0.0811 | 15.59 | 28.3s | +0.140 |
| 3 | 2026-08-06 | HEAD + mom + no cfl_max | no cfl_max, gs=0.15 | 0.7853 | 0.0000% | 0.2407 | 8.70 | 22.3s | +0.108 |
