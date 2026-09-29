# Mindboggle 90-Pair 5-Arm Benchmark — 2026-09-20 Full Parameter Record

**Run date:** 2026-09-20  
**Commit:** `3e862c9` (tag: `v5.4.12`) → re-used affines at `c932cbc` (`v5.4.14`)  
**Hardware:** Apple Silicon GPU (MPS)  
**Cohort:** 90 Mindboggle-101 pairs (intra- and inter-scanner)  
**Affine backend key:** `pt7` (native PyTorch Mattes-MI solver, `mode='auto'`)  
**Source of truth:** `src/syntx/benchmark/config.py` `DEFAULT_BENCHMARK_CONFIG`, `src/syntx/benchmark/evaluate.py`

---

## Preprocessing (all arms)

| Step | Value |
|---|---|
| Intensity normalization | Foreground 2–98 % percentile → [0, 1] |
| N4 bias correction | `use_n4=True` (ANTsTorch N4, applied to raw images before normalization) |
| NLM denoising | `denoise=False` (skipped for this run) |
| Seed | `torch.manual_seed(seed + pair_idx)`, `np.random.seed(seed + pair_idx)` (default `seed=42`) |

---

## Arm 0 — Canonical Affine Seed (shared by all deformable arms)

```
syntx.robust_affine(fi, mi, mode='auto', verbose=False)
```

| Parameter | Value |
|---|---|
| `mode` | `'auto'` — native PyTorch Mattes-MI multi-start solver |
| Cache key | `pair_{idx:03d}_pt7_affine.mat` (reused if file exists and backend key matches) |
| **Mean Dice** | **0.3448** ± 0.0230 |
| **Mean runtime** | **5.3 s** |

---

## Arm 1 — ANTs C++ SyN (reference baseline)

```python
ants.registration(
    fixed=fi, moving=mi,
    type_of_transform="SyNOnly",
    initial_transform=aff_0,
    syn_metric="CC",
    syn_sampling=2,
    reg_iterations=(100, 100, 20),
    flow_sigma=3.0,
    total_sigma=0.0,
    grad_step=0.25,
)
```

| Parameter | Value |
|---|---|
| `type_of_transform` | `"SyNOnly"` (deformable only; shared affine seed applied externally) |
| `syn_metric` | `"CC"` |
| `syn_sampling` | 2 |
| `reg_iterations` | `(100, 100, 20)` |
| `flow_sigma` | 3.0 |
| `total_sigma` | 0.0 |
| `grad_step` | 0.25 |
| **Mean Dice** | **0.6097** ± 0.0220 |
| **Mean runtime** | **124.3 s** |
| **Mean folding %** | **0.0000 %** |

---

## Arm 2 — syntx.syn (Sobolev SyN)

```python
syntx.syn(
    fixed=fi, moving=mi,
    initial_transform=aff_0,
    backend="pytorch", device="mps",
    grad_step=0.25,
    flow_sigma=3.0,          # from evaluate.py default (model_cfg.get("fluid_sigma") → 2.5 overridden to 3.0)
    total_sigma=0.0,
    reg_iterations=[100, 100, 20],
    similarity_metric="cc2",
    formulation="eulerian",
    regularizer="sobolev",
    kernel_type="sobolev",
    sobolev_alpha=1.5,
    in_loop_inv_steps=10,    # from evaluate.py syn_config
    fast_smooth=True,        # syn_fast_smooth=True in syn_config
)
```

| Parameter | Value |
|---|---|
| `regularizer` | `"sobolev"` |
| `sobolev_alpha` | **1.5** |
| `grad_step` | 0.25 |
| `flow_sigma` | 3.0 |
| `total_sigma` | 0.0 |
| `reg_iterations` | `[100, 100, 20]` |
| `similarity_metric` | `"cc2"` |
| `formulation` | `"eulerian"` |
| `kernel_type` | `"sobolev"` |
| `in_loop_inv_steps` | 10 |
| `fast_smooth` | `True` |
| **Mean Dice** | **0.6350** ± 0.0215 |
| **Win rate vs ANTs** | **100 % (90/90)** |
| **Mean runtime** | **51.6 s** (2.4× speedup vs ANTs) |
| **Mean folding %** | **0.0066 %** |

---

## Arm 3 — syntx.tvf (DST-I / DSTI-1)

```python
syntx.tvf(
    fixed=fi, moving=mi,
    initial_transform=aff_0,
    device="mps",
    regularizer="dsti1",
    flow_sigma=3.0,          # spectral gate only (value inert; any positive value = enabled)
    total_sigma=0.035,       # tvf_total_sigma
    alpha=0.035,             # dsti_alpha — controls DST-I kernel regularization strength
    optimizer="reg_adam",
    optimizer_lr=1.2,
    max_step_norm=0.54,      # optimal CFL bound from 3D sweep
    n_time_steps=3,
    cfl_momentum=0.95,
    constant_speed=True,
    constant_speed_relaxation=0.10,
    antisymmetric=False,
    fast_smooth=False,
    reg_iterations=[100, 100, 20],
    similarity_metric="cc2",
    multipoint_loss=[0.0, 0.5, 1.0],
)
```

| Parameter | Value |
|---|---|
| `regularizer` | `"dsti1"` |
| `alpha` (`dsti_alpha`) | **0.035** |
| `flow_sigma` | 3.0 (gate only; kernel shape controlled by `alpha`) |
| `total_sigma` | 0.035 |
| `optimizer` | `"reg_adam"` |
| `optimizer_lr` | 1.2 |
| `max_step_norm` | **0.54** |
| `n_time_steps` | 3 |
| `cfl_momentum` | 0.95 |
| `constant_speed` | `True` |
| `constant_speed_relaxation` | 0.10 |
| `antisymmetric` | `False` |
| `reg_iterations` | `[100, 100, 20]` |
| `similarity_metric` | `"cc2"` |
| **Mean Dice** | **0.6387** ± 0.0219 |
| **Win rate vs ANTs** | **100 % (90/90)** |
| **Mean runtime** | **250.4 s** (0.5× vs ANTs) |
| **Mean folding %** | **0.00246 %** |

---

## Arm 4 — syntx.syngs (Geodesic Shooting)

```python
syntx.syngs(
    fixed=fi, moving=mi,
    initial_transform=aff_0,
    backend="pytorch", device="mps",
    flow_sigma=3.0,
    total_sigma=0.0,
    alpha=0.45,              # optimal Sobolev strength from 3D sweep
    regularizer="sobolev",
    transport_mode="transport",
    optimizer="reg_adam",
    optimizer_lr=1.0,
    max_step_norm=0.19,      # optimal CFL bound ensuring fold < 0.005%
    reg_iterations=[100, 100, 20],
    similarity_metric="cc2",
    bootstrap_mode="antithetic",
    bootstrap_orig_weight=0.50,
    bootstrap_jitter_scale=0.25,
    n_steps=8,
    solver="euler",
)
```

| Parameter | Value |
|---|---|
| `regularizer` | `"sobolev"` |
| `alpha` | **0.45** |
| `flow_sigma` | 3.0 |
| `total_sigma` | 0.0 |
| `transport_mode` | `"transport"` |
| `optimizer` | `"reg_adam"` |
| `optimizer_lr` | 1.0 |
| `max_step_norm` | **0.19** |
| `n_steps` | 8 |
| `solver` | `"euler"` |
| `bootstrap_mode` | `"antithetic"` |
| `bootstrap_orig_weight` | 0.50 |
| `bootstrap_jitter_scale` | 0.25 |
| `reg_iterations` | `[100, 100, 20]` |
| `similarity_metric` | `"cc2"` |
| **Mean Dice** | **0.6187** ± 0.0227 |
| **Win rate vs ANTs** | **86.7 % (78/90)** |
| **Mean runtime** | **115.9 s** (1.07× vs ANTs) |
| **Mean folding %** | **0.0019 %** |

---

## Arm 5 — syntx.greedy (Compositive Greedy)

```python
syntx.greedy(
    fixed=fi, moving=mi,
    initial_transform=aff_0,
    reg_iterations=[100, 100, 20],
    learning_rate=0.50,      # grad_step
    flow_sigma=1.8,
    total_sigma=0.28,
    optimizer="adam",
    regadam_sigma=0.8,
    anderson=False,
    anderson_steps=5,
    return_inverse=False,
    similarity_metric="cc2",
    device="mps",
)
```

| Parameter | Value |
|---|---|
| `learning_rate` (`grad_step`) | 0.50 |
| `flow_sigma` | 1.8 |
| `total_sigma` | 0.28 |
| `optimizer` | `"adam"` |
| `regadam_sigma` | 0.8 |
| `reg_iterations` | `[100, 100, 20]` |
| `similarity_metric` | `"cc2"` |
| **Mean Dice** | **0.6071** ± 0.0250 |
| **Win rate vs ANTs** | **36.7 % (33/90)** |
| **Mean runtime** | **12.9 s** (9.6× speedup vs ANTs) |
| **Mean folding %** | **0.0008 %** |

---

## Summary

| Arm | Method | Mean Dice | vs ANTs | Runtime | Folding % |
|---|---|---|---|---|---|
| 0 | Canonical Affine (`robust_affine`) | 0.3448 | — | 5.3 s | — |
| 1 | ANTs C++ SyN (reference) | 0.6097 | — | 124.3 s | 0.0000 % |
| 2 | syntx.syn (Sobolev, α=1.5) | **0.6350** | +0.0253, **90/90** | 51.6 s | 0.0066 % |
| 3 | syntx.tvf (DSTI-1, α=0.035) | **0.6387** | +0.0290, **90/90** | 250.4 s | 0.00246 % |
| 4 | syntx.syngs (Sobolev, α=0.45) | 0.6187 | +0.0090, 78/90 | 115.9 s | 0.0019 % |
| 5 | syntx.greedy | 0.6071 | +0.0000, 33/90 | **12.9 s** | 0.0008 % |
