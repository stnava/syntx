# syn.py Technical Review & Refactoring Plan

**File:** `src/syntx/syn.py` (5,416 lines, 261 KB)  
**Date:** 2026-08-15  
**Scope:** Deep correctness audit + architectural refactoring plan for regression prevention and code reuse across SyN, TVF, and SyNGS.

---

## Part 1: Correctness Issues & Bugs in syn.py

### 1.1 Function Definition Shadowing (Lines 5148–5412)

**Severity: Medium — silently loads dead code, then overwrites it.**

`plot_deformation_grid` (line 5148) and `extract_2d_slice` (line 5326) are defined as ~350 lines of code inside `syn.py`. Then at line 5406, they are immediately overwritten:

```python
from .viz import (extract_2d_slice, plot_deformation_grid, ...)
```

The 350 lines of visualization code are dead weight — imported by no one, overwritten at module load. They inflate the file, slow `import syntx`, and create a maintenance trap where a developer fixes the `syn.py` copy without realizing the `viz` copy is the one actually executed.

---

### 1.2 `syn_metric` Keyword Shadowing (Lines 4170, 4305)

**Severity: Medium — confusing precedence, potential silent override.**

`syn_metric` is a named parameter with default `'lncc'` (line 4170). But at line 4305:

```python
syn_metric = kwargs.pop('similarity_metric', syn_metric)
```

This means a user who passes both `syn_metric='lncc'` and `similarity_metric='mattes_mi'` will get `mattes_mi` silently. More importantly, `kwargs.pop` removes it from kwargs, so any downstream code checking `kwargs` for `similarity_metric` will not find it. This is an API ambiguity — two keys mapping to the same parameter with unclear precedence.

---

### 1.3 Named Parameters Re-read from `kwargs.get()` (Lines 4398–4404)

**Severity: Low — API confusion, no correctness impact.**

Six named parameters (`inverse_steps`, `inverse_method`, `vgg_layers`, `vgg_patch_size`, `vgg_num_patches`, `vgg_mode`, `vgg_lncc_window_size`) are declared in the function signature AND then unconditionally re-read via `kwargs.get()`. This means there are two ways to pass the same parameter:

```python
registration(fixed, moving, vgg_layers=[4])         # via named arg
registration(fixed, moving, **{'vgg_layers': [4]})   # via kwargs → same result
```

But because `kwargs.get()` defaults to the named arg value, both paths produce identical behavior. The duplication is harmless but adds confusion.

---

### 1.4 `device` Variable Overwritten (Lines 4425–4432)

**Severity: Low — correct behavior, misleading flow.**

```python
device = kwargs.get('device', None)          # line 4425
if device is None:
    if torch.cuda.is_available():
        device = 'cuda'                       # line 4428
    elif torch.backends.mps.is_available():
        device = 'mps'                        # line 4430
    else:
        device = 'cpu'                        # line 4432
```

The `device` from `kwargs` is immediately overwritten if `None`. This is correct but reads as if `device` is used before being set. Minor readability issue.

---

### 1.5 Provenance `levels` Records `None` Instead of Actual Levels (Line 4792)

**Severity: Low — provenance data inaccuracy.**

```python
provenance = build_engine_provenance(
    ...
    levels=levels,        # This is the user's original input, possibly None
    ...
)
```

If the user did not pass `levels`, this records `None` instead of the actual computed `levels_to_use` (line 4387). The provenance record is therefore incomplete for runs using default pyramid levels.

---

### 1.6 Inner `import math` (5 redundant occurrences)

**Severity: Low — negligible perf, code hygiene.**

`import math` appears at the top level (line 16) and then again at lines 2566, 2853, 3116, 4412, 4474 inside function bodies.

---

### 1.7 `normalize_tensor` (Lines 5064–5146) Is a General Utility in the Wrong Module

**Severity: Low — misplaced code.**

This is a pure tensor utility function with no registration-specific logic. It belongs in a `utils.py` or `spatial.py` module, not in the 5,400-line registration core.

---

## Part 2: Structural Problems (Shared with TVF & SyNGS)

### 2.1 Massive Code Duplication Across Registration Engines

The following blocks of code are copy-pasted across `syn.py`, `tvf.py`, and `syngs.py`:

| Duplicated Component | syn.py | tvf.py | syngs.py |
|---|---|---|---|
| `_apply_sobolev_green_operator` | Lines 2334–2382 | Lines 402–448 | Lines 329–347 |
| `_apply_dsti_green_operator` | Lines 2384–2468 | Lines 450–533 | — |
| `_apply_dsti1_green_operator` | Lines 2470–2553 | Lines 535–606 | — |
| `_create_boundary_mask` | Lines 681–692 (`get_boundary_mask`) | Lines 376–400 | Lines 252–278 |
| LARS optimizer | — | Lines 108–155 | Lines 34–85 |
| Registration entry-point boilerplate | Lines 4164–4832 | Lines 1514–1990 | Lines 919–1120 |
| Winsorize + normalize images | Lines 4340–4352 | Lines 1682–1694 | — |
| Parse ANTs affine | Imported from self | Lines 1673, 1679 | Line 1071 |
| Device auto-detection | Lines 4425–4432 | Lines 1709–1716 | — |
| Transform export (tempfile .nii.gz/.mat) | Lines 4558–4620 | Lines 1876–1893 | Lines 1081–1100 |
| Provenance building | Lines 4772–4820 | Lines 1942–1987 | — |
| GPU cleanup (gc + empty_cache) | Lines 4822–4829 | Lines 1913–1918 | — |

**Impact:** Every bug fix must be manually replicated across 3 files. The TVF review (§1.1 `cfl_max_val` shadowing, §2.1 XYZ/ZYX mismatch) demonstrates how SyNGS likely has the same bugs but has never been audited.

---

### 2.2 `SyNTo.fit()` Is 1,400 Lines of Monolithic Code

`SyNTo.fit()` (lines 2555–3927) is a single method containing:
- Center-of-mass initialization (lines 2630–2730)
- Affine optimization loop (lines 2730–2960)
- Multi-resolution warp upsampling (lines 2960–3000)
- Physical grid construction and caching (lines 3000–3050)
- Metric dispatch (lines 3050–3070)
- Deformable forward pass (lines 3070–3250)
- Gradient smoothing (lines 3250–3340)
- CFL step computation (lines 3330–3400)
- RProp optimizer (lines 3400–3500)
- Inverse field updates (lines 3500–3600)
- Convergence checking (lines 3540–3570)
- Retry logic (lines 3560–3580)
- Export and warp application (lines 3680–3760)
- Forward/inverse inference methods (lines 3756–3927)

This makes the function nearly impossible to unit test, review, or modify without introducing regressions.

---

## Part 3: Refactoring Plan

### Goal

Decompose the 5,416-line `syn.py` and eliminate cross-engine duplication to achieve:
1. **Single source of truth** for every algorithm (smoothing, CFL, inverse solvers, metrics).
2. **Unit-testable components** — each module <500 lines, each function <100 lines.
3. **Shared infrastructure** between SyN, TVF, and SyNGS without copy-paste.
4. **Regression prevention** — changes to CFL, LNCC, or smoothing are tested once, apply everywhere.

### Proposed Module Structure

```
src/syntx/
├── __init__.py                    # Public API (unchanged)
├── core/                          # NEW: shared infrastructure
│   ├── __init__.py
│   ├── affine.py                  # HierarchicalAffine, get_rotation_matrix,
│   │                              #   parse_ants_affine, grid_to_physical_affine,
│   │                              #   physical_to_grid_affine, export_ants_affine_transform
│   ├── grid.py                    # get_physical_grid_torch, physical_to_normalized_torch,
│   │                              #   physical_to_normalized_torch_cached,
│   │                              #   grid_sample_nd, grid_sample_bspline_torch,
│   │                              #   compose_grids, get_boundary_mask
│   ├── smoothing.py               # separable_gaussian_filter, get_cached_gaussian_kernel_1d,
│   │                              #   apply_sobolev_green_operator, apply_dsti_green_operator,
│   │                              #   apply_dsti1_green_operator, create_boundary_mask
│   ├── optimizers.py              # CFL step (extract from fit loop), LARS, TVFConjugateGradient,
│   │                              #   RProp, convergence checking
│   ├── losses.py                  # local_ncc_loss_nd, AnalyticalLNCC, ANTsPseudoLNCC,
│   │                              #   mattes_mi_loss_nd, mattes_mi_loss_core, b_spline_3
│   ├── jacobian.py                # _spatial_jacobian_nd, compute_jacobian_determinant_nd,
│   │                              #   compute_physical_jacobian_determinant
│   ├── inverse.py                 # update_inverse_field_nd, update_inverse_field_nd_anderson,
│   │                              #   update_inverse_field_nd_hybrid_lm,
│   │                              #   compute_inverse_identity_error_nd,
│   │                              #   calculate_inverse_identity_error
│   ├── pipeline.py                # Shared registration boilerplate:
│   │                              #   normalize_and_tensorize(fixed, moving, ...),
│   │                              #   auto_detect_device(), export_transforms(),
│   │                              #   build_return_dict(), cleanup_gpu()
│   └── utils.py                   # normalize_tensor
├── syn.py                         # SyNTo class + registration() entry point ONLY
│                                  #   (~1500 lines: class definition, fit loop, forward/inverse)
├── tvf.py                         # TVFModel class + tvf_registration() ONLY
│                                  #   (~1200 lines: class definition, ODE integration, fit)
├── syngs.py                       # GeodesicShootingModel + syngs_registration() ONLY
├── features.py                    # TriPlanarVGG3DLoss, FeatureSpaceLoss (move from syn.py)
├── viz/                           # Visualization (already exists, keep as-is)
├── benchmark/                     # Benchmarking (already exists)
└── ...                            # Other existing modules unchanged
```

### Migration Strategy

> [!IMPORTANT]
> All steps preserve backward compatibility. No public API changes. No import path changes for end users.

#### Phase 1: Extract Pure Utilities (Low Risk)

| Step | Source → Destination | Lines Moved | Risk |
|---|---|---|---|
| 1a | `syn.py` `get_rotation_matrix` → `core/affine.py` | 60 | None |
| 1b | `syn.py` `HierarchicalAffine` → `core/affine.py` | 100 | None |
| 1c | `syn.py` `parse_ants_affine`, `grid_to_physical_affine`, `physical_to_grid_affine` → `core/affine.py` | 150 | None |
| 1d | `syn.py` grid functions → `core/grid.py` | 200 | None |
| 1e | `syn.py` LNCC, MI, analytical loss classes → `core/losses.py` | 300 | None |
| 1f | `syn.py` Jacobian, inverse solvers → `core/jacobian.py`, `core/inverse.py` | 650 | Low |
| 1g | `syn.py` `normalize_tensor` → `core/utils.py` | 80 | None |
| 1h | Delete dead `plot_deformation_grid` and `extract_2d_slice` from `syn.py` bottom | 260 | None |

After Phase 1, `syn.py` drops from 5,416 to ~3,600 lines. All moved functions are re-exported from `syn.py` via `from .core.losses import local_ncc_loss_nd` etc., so no external imports break.

#### Phase 2: Extract Shared Smoothing & Optimizers (Medium Risk)

| Step | Source → Destination | Impact |
|---|---|---|
| 2a | `SyNTo._apply_sobolev_green_operator`, `_apply_dsti*` → `core/smoothing.py` as free functions | Eliminates 3-way duplication (syn, tvf, syngs) |
| 2b | `_create_boundary_mask` / `get_boundary_mask` → `core/smoothing.py` | Single implementation |
| 2c | `separable_gaussian_filter` → `core/smoothing.py` | Already a free function |
| 2d | LARS (from tvf.py, syngs.py) → `core/optimizers.py` | Single implementation |
| 2e | CFL step computation (extract from SyNTo.fit and TVFModel.fit) → `core/optimizers.py:cfl_step()` | Critical: this is where XYZ/ZYX bugs live. Fixing once fixes everywhere. |

After Phase 2, SyNTo, TVFModel, and GeodesicShootingModel all call `from .core.smoothing import apply_sobolev_green_operator` instead of each having their own copy.

#### Phase 3: Extract Pipeline Boilerplate (Medium Risk)

| Step | Source → Destination | Impact |
|---|---|---|
| 3a | Image normalization + winsorize → `core/pipeline.py:normalize_and_tensorize()` | Eliminates 2-way duplication (syn, tvf) |
| 3b | Device auto-detection → `core/pipeline.py:auto_detect_device()` | Eliminates 2-way duplication |
| 3c | Transform export (tempfile .nii.gz/.mat writing) → `core/pipeline.py:export_transforms()` | Eliminates 3-way duplication |
| 3d | Provenance building → keep in `reporting.py` (already extracted) | No change needed |
| 3e | GPU cleanup → `core/pipeline.py:cleanup_gpu(device, backend)` | Eliminates 3-way duplication |

After Phase 3, `registration()` and `tvf_registration()` each shrink by ~200 lines.

#### Phase 4: Decompose SyNTo.fit() (Higher Risk, Higher Reward)

| Step | Extract From SyNTo.fit() | New Location |
|---|---|---|
| 4a | CoM initialization (lines 2630–2730) → `core/affine.py:compute_com_initialization()` | Testable independently |
| 4b | Affine optimization loop (lines 2730–2960) → `SyNTo._fit_affine()` private method | Isolates affine from deformable |
| 4c | Deformable forward pass (lines 3070–3250) → `SyNTo._deformable_step()` | One epoch = one method call |
| 4d | Convergence + retry logic → `core/optimizers.py:check_convergence()` (already exists as free function) | Reused by TVF, SyNGS |

After Phase 4, `SyNTo.fit()` becomes a ~200-line orchestrator calling well-tested sub-methods.

### Testing Strategy for Regression Prevention

Each extracted module gets its own test file:

```
tests/
├── test_core_affine.py      # Round-trip: parse → export → re-parse = identity
├── test_core_grid.py        # Grid ↔ physical ↔ normalized round-trips
├── test_core_smoothing.py   # Gaussian filter symmetry, DST-I normalization, boundary mask shape
├── test_core_optimizers.py  # CFL step magnitude bounds, LARS trust ratio, convergence detection
├── test_core_losses.py      # LNCC variance floor, Cauchy-Schwarz clamping, MI gradient flow
├── test_core_jacobian.py    # det(J) = 1 for identity, known analytic cases
├── test_core_inverse.py     # Anderson vs fixed-point parity, inverse error < tolerance
├── test_core_pipeline.py    # Normalize round-trip, device detection, transform export I/O
```

### Priority Order

1. **Phase 1** (pure extraction, zero risk) — do first, immediately reduces cognitive load.
2. **Phase 2a–2c** (smoothing) — highest duplication-bug-density area.
3. **Phase 2d–2e** (optimizers) — CFL is where XYZ/ZYX and shrink_ratio bugs live.
4. **Phase 3** (pipeline) — reduces registration entry-point boilerplate.
5. **Phase 4** (fit decomposition) — highest reward but requires careful integration testing.

### Estimated Impact

| Metric | Before | After |
|---|---|---|
| `syn.py` line count | 5,416 | ~1,800 |
| `tvf.py` line count | 1,990 | ~1,200 |
| `syngs.py` line count | 1,391 | ~900 |
| Duplicated smoothing implementations | 3 | 1 |
| Duplicated optimizer implementations | 3 | 1 |
| Duplicated pipeline boilerplate | 3 | 1 |
| Files with independent unit tests | ~3 | ~11 |
| Average function length | ~120 lines | ~50 lines |
