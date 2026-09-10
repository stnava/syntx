# syntx v5.0.0 Diagnostic Benchmark Summary

- **Timestamp**: 2026-09-09 17:57:53
- **Device**: mps
- **Cohorts Tested**: 6 Canonical Pairs (Intra & Inter)
- **Win Rate vs ANTs C++**: 6/6 (100.0%)
- **Mean Sym DICE**: 0.6394 (v5.0.0) vs 0.5809 (ANTs C++) | Gain: +5.85%
- **Mean Affine Baseline**: 0.3079
- **Mean Compute Time**: 60.9s (v5.0.0) vs 304.4s (ANTs C++) | Speedup: 5.00x

## Detailed Pair Results

| Pair | Type | Fixed -> Moving | Affine DICE | ANTs DICE | v5.0.0 DICE | Gain vs ANTs | Folding % | Min det(J) | v5.0.0 Time | Outcome |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | intra | OASIS-TRT-20-17 -> OASIS-TRT-20-16 | 0.2839 | 0.5750 | **0.6339** | +5.89% | 0.0006% | 0.0000 | 77.9s | **WIN** |
| 1 | intra | OASIS-TRT-20-16 -> OASIS-TRT-20-17 | 0.2976 | 0.6050 | **0.6270** | +2.20% | 0.0055% | 0.0000 | 79.2s | **WIN** |
| 2 | intra | NKI-TRT-20-10 -> NKI-TRT-20-4 | 0.2940 | 0.6240 | **0.6722** | +4.82% | 0.0019% | 0.0000 | 58.6s | **WIN** |
| 41 | inter | MMRR-21-2 -> NKI-TRT-20-18 | 0.3312 | 0.5520 | **0.6559** | +10.39% | 0.0005% | 0.0000 | 45.7s | **WIN** |
| 44 | inter | NKI-TRT-20-2 -> MMRR-21-2 | 0.3464 | 0.5877 | **0.6277** | +4.00% | 0.0024% | 0.0000 | 57.6s | **WIN** |
| 45 | inter | MMRR-21-17 -> NKI-TRT-20-7 | 0.2944 | 0.5419 | **0.6200** | +7.81% | 0.0008% | 0.0000 | 46.4s | **WIN** |
