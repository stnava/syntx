# Benchmark & Verification Report: ANTsTorch PR #96–#99 & `syntx.greedy` Port

- **Baseline Commit**: `syntx` `v6.39.0` (`2c15e8f6733477e4f7e19a0b37e139b10ff6859c`)
- **Environment**: `darwin` (Apple Silicon `mps` & `cpu`), Python `3.13.2`, PyTorch `2.13.0`
- **Upstream References**:
  - `ANTsX/ANTsTorch` PR `#96` (`0c83bcc`): Peak registration memory reductions (immediate post-`.backward()` autograd graph release and per-term detached-leaf backward for multi-channel/multi-metric `SyN`)
  - `ANTsX/ANTsTorch` PR `#97`: Shifted-slice separable 3-D Gaussian blur for `landmarks/blob.py` (`_separable_gaussian3d`) and `features.py` (`SwinViT` 32-multiple padding without 5-D `F.pad`)
  - `ANTsX/ANTsTorch` PR `#98` (`6a080e0`): Shifted-slice separable Gaussian blur in `core/smoothing.py` (`separable_gaussian_filter`)
  - `ANTsX/ANTsTorch` PR `#99` (`eea9c0b`): Strict `metric_chunks` validation, longest-axis slab slicing in `_ChunkedLocalNCC`, and `syn.py` provenance recording
  - `syntx.greedy` cross-repo audit: Closed-form affine coordinate evaluation (`_Conv1x1AffineGrid` with `[-1, 1]` clamp derivative mask), per-level cached fixed-image LNCC moments (`_is_default_greedy_lncc`), fused Adam update, batched `LossSlopeStop`/`loss_history` host transfers, on-device `normalized_to_physical_disp` export, and parallel `_write_ants_images_fast` + `_cache_disp_image_and_field`
- **Machine-Readable Provenance JSON**: [`docs/provenance/pr96_98_port_benchmark.json`](provenance/pr96_98_port_benchmark.json)
- **Benchmark Harness**: [`scripts/benchmark_pr96_98_port.py`](../scripts/benchmark_pr96_98_port.py)
- **Dedicated Unit Test Suite**: [`tests/test_pr96_98_port.py`](../tests/test_pr96_98_port.py)

---

## 1. What Was Ported vs. What Was Already Superseded in `syntx`

| Component | Upstream (`ANTsX/ANTsTorch`) Change | Ported to `syntx`? | Engineering Rationale & Verification |
| :--- | :--- | :--- | :--- |
| `landmarks/blob.py` (`_separable_gaussian3d`) | PR `#97`: Replace 5-D `F.conv3d` with per-axis shifted-slice `.add_` | **Yes (with commutative pair fix)** | Eliminates 5-D `F.conv3d` and 5-D `F.pad` on MPS (`4.081x` faster on MPS, `2.913x` faster on CPU). Uses `float64`-normalized tap weights (`-> tolist()`), commutative symmetric pair addition `(out[0:L-2d] + out[2d:L])` for exact axis-reversal symmetry (`0.00000000 mm` error on `flip_xy`), `mode="replicate"` (default) and `mode="zeros"`/`"constant"` support, and autograd-safe fallback when `requires_grad=True`. |
| `core/smoothing.py` (`separable_gaussian_filter`) | PR `#98`: Replace `F.conv1d` / `F.conv3d` with shifted-slice `.add_` | **Selective (CPU 3-D + MPS `klen >= 33` only)** | On MPS at normal registration sigmas (`sigma=3.0`, `klen=17 < 33`), `syntx`'s reshaped 3-D `F.conv1d` runs in `11.37 ms`. On CPU (`4.990x` faster on `192×256×256×3`, `3.387x` faster on `192×256×256×1`) and on MPS large kernels (`klen >= 33`), zero-pad-allocation `_shifted_slice_blur_1d_inplace` avoids `F.conv3d(..., groups=C)` overhead and reduces MPS large-kernel driver memory delta `5.0x` (`40.0 -> 8.0 MB`). Preserves `get_cached_gaussian_kernel_1d` (`float32`/`float64` tap weights) and `requires_grad=True` autograd safety. |
| `features.py` (`SwinUNETRExtractor`) | PR `#97`: Replace 5-D `F.pad` with per-axis `torch.cat` | **Yes** | On PyTorch 2.13.0 MPS, 5-D `F.pad(x, (0, 0, 0, 0, 0, pad_d))` when `H * W >= 65536` silently corrupts data (`mps_vs_cpu_max_diff = 4.685089`). Per-axis `torch.cat` in `_pad_to_multiple_3d` gives exact `0.0` diff vs CPU. |
| `syn.py`, `tvf.py`, `syngs.py`, `greedy.py` | PR `#96`: Immediate post-`.backward()` `del loss` | **Yes (all 4 solvers)** | Immediately detaches scalar loss values (`loss_t = loss.detach()`) and deletes live autograd loss / warped intermediate tensors right after `.backward()`, freeing similarity graphs before regularizer FFTs, `optimizer.step()`, warp composition, and inverse projection. |
| `syn.py` (`_backward_similarity_terms_per_term`) | PR `#96`: Per-term detached-leaf backward for multi-channel `SyN` | **Yes (with `grad is not None` & zero-grad guards)** | Backpropagates each active channel/metric term separately into detached `(I_mid_leaf, J_mid_leaf)` leaves so at most 1 metric's autograd graph resides in VRAM at a time (`-290.34 MB` peak allocated, `-1,024.00 MB` peak Metal driver memory on 3-channel `mbhard`), with explicit `grad is not None and t.requires_grad` filtering and `torch.zeros_like(warp_l2r)` fallback when all terms have `grad=None`. Single-channel (`len == 1`) path is untouched. |
| `core/losses/registry.py`, `cross_correlation.py`, `syn.py` | PR `#99`: `metric_chunks` validation, longest-axis slab slicing, provenance | **Yes** | Fixed `metric_chunks=0` silent fallback (`int(metric_chunks or 1)`) by strictly rejecting `bool`, non-`int`, and `< 1` with `ValueError`; updated `_ChunkedLocalNCC` to slice along `_longest_slab_axis(I.shape[2:])` with `.contiguous()` on strided views (`slab_axis > 2`); recorded `metric_chunks` in `syn.py` engine provenance and rejected `metric_chunks > 1` on linear-only transforms or non-CC metrics (including `'dt'`). |
| `greedy.py` (`syntx.greedy`) | Cross-repo audit Ranks 1–7 | **Yes** | Replaced `sample_field_cf(affine_grid_cf, ...)` with `_Conv1x1AffineGrid` (saving `in_bounds = (coords >= -1) & (coords <= 1)` for exact border-clamped backward; `-780.20 MB` to `-924.13 MB` peak allocated VRAM, `-1,024.00 MB` peak Metal driver memory); cached constant `fi_down` LNCC moments per pyramid level (`_is_default_greedy_lncc = True` on `cc2`/`lncc`, cutting deformable loop time `3.26 -> 2.69 s` on `greedy_default` and `8.62 -> 7.47 s` on `greedy_matched`); fused Adam step; batched `LossSlopeStop`/`loss_history` host syncs; added `require_finite_images` and `dsti_alpha` validation; replaced CPU float64 meshgrid + round-trip with GPU `normalized_to_physical_disp`; replaced `ants.image_write` with parallel `_write_ants_images_fast` + `_cache_disp_image_and_field` (`17.26 -> 14.24 s`, `1.212x` faster when `return_inverse=True`). |
| `cross_correlation.py` (`_DeterministicBoxMean`), `mps_kernels.py`, `pyramid.py`, `utils.py` | PR `#96` 3-D `F.avg_pool3d` LNCC & `_grid_sample_3d_metal` | **Already Superseded in `syntx`** | `syntx` already uses 15-tap separable `_DeterministicBoxMean` (zero saved pool activations, `8.3x` fewer taps than 5x 3-D `F.avg_pool3d`), sync-free `DeterministicGridSample` Metal `gs3d_bwd`, centralized `build_pyramid(..., 'deformable')`, and CT-safe `normalize_image(method='auto')`. |

---

## 2. Micro-Benchmark Results (Before vs. After)

All values below are reported directly from [`docs/provenance/pr96_98_port_benchmark.json`](provenance/pr96_98_port_benchmark.json).

### 2.1 `syntx.landmarks.blob._separable_gaussian3d` & `detect_sift3d` (`192×256×256` `mbhard`)

| Benchmark | Device | Before (`v6.39.0` 5-D `F.conv3d`) | After (Shifted-Slice + Commutative Pair) | Speedup | Interior / Full-Volume Max Diff |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `_separable_gaussian3d` (`sigma=2.0`) | CPU | `86.82 ms` (IQR `2.34 ms`) | **`29.80 ms`** (IQR `0.06 ms`) | **`2.913x`** | `2.98e-07` / `2.98e-07` |
| `_separable_gaussian3d` (`sigma=2.0`) | MPS | `49.62 ms` (IQR `0.04 ms`) | **`12.16 ms`** (IQR `0.02 ms`) | **`4.081x`** | `2.98e-07` / `2.98e-07` |
| `detect_sift3d` (`max_keypoints=200`) | CPU | `2.889 s` (`200` kps) | **`2.289 s`** (`200` kps) | **`1.262x`** | — |
| `detect_sift3d` (`max_keypoints=200`) | MPS | `0.528 s` (`200` kps) | **`0.230 s`** (`200` kps) | **`2.294x`** | — |

### 2.2 `syntx.core.smoothing.separable_gaussian_filter` (CPU & MPS)

| Case | Device | Shape | Sigma (`klen`) | Before (`v6.39.0`) | After | Speedup | Driver Delta Before / After (MB) | Max Abs Diff |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `vector_192x256x256x3_sigma3` | CPU | `(1, 192, 256, 256, 3)` | `3.0` (`17`) | `443.40 ms` | **`88.85 ms`** | **`4.990x`** | — | `1.19e-07` |
| `scalar_192x256x256x1_sigma2` | CPU | `(1, 192, 256, 256, 1)` | `2.0` (`13`) | `91.69 ms` | **`27.07 ms`** | **`3.387x`** | — | `1.49e-07` |
| `standard_kernel_192x256x256x3_sigma3` | MPS | `(1, 192, 256, 256, 3)` | `3.0` (`17 < 33`) | `11.35 ms` | **`11.37 ms`** (kept `F.conv1d`) | `0.998x` | `8.0 / 8.0` (`1,032.39 MB` peak) | `0.00e+00` |
| `large_kernel_128cubed_sigma20` | MPS | `(1, 128, 128, 128, 1)` | `20.0` (`69 >= 33`) | `1.28 ms` (warm `F.conv1d`) | `14.44 ms` (zero-pad `.add_`) | `0.088x` | **`40.0 / 8.0`** (`72.39 -> 40.39 MB` peak, **`1.792x` lower**) | `1.49e-07` |

### 2.3 `syntx.features.SwinUNETRExtractor._pad_to_multiple_3d` (MPS 5-D `F.pad` Bug Fix)

| Input Shape | Padded Shape | Before (`5-D F.pad` on MPS vs CPU Max / Mean Diff) | After (`_pad_to_multiple_3d` `torch.cat` on MPS vs CPU Max / Mean Diff) |
| :--- | :--- | ---: | ---: |
| `(1, 1, 18, 256, 256)` (`H * W = 65536`) | `(1, 1, 32, 256, 256)` | **`4.685089` / `0.449143`** (**SILENT MPS CORRUPTION**) | **`0.000000` / `0.000000`** (exact bitwise match) |

### 2.4 `syntx.core.losses._ChunkedLocalNCC` (`metric_chunks=1` vs `metric_chunks=4`, PR #99)

| Metric | `metric_chunks=1` (Unchunked) | `metric_chunks=4` (Longest-Axis Slab Slicing) | Delta / Diff |
| :--- | ---: | ---: | ---: |
| Warm Forward + Backward Time | `16.67 ms` | `27.39 ms` | `+10.72 ms` |
| In-Flight Peak MPS Allocated Memory | `336.01 MB` (delta `+304.00 MB`) | **`233.01 MB`** (delta **`+197.00 MB`**) | **`-107.00 MB` delta saving** (`-35.2%`) |
| Scalar Loss Value | `-0.0084497184` | `-0.0084497193` | **`9.31e-10`** abs diff |
| Max Abs Grad Diff (`grad_I` / `grad_J`) | — | — | **`1.71e-13` / `1.14e-13`** |

---

## 3. End-to-End Deformable Registration Benchmarks on Mindboggle `mbhard` (`192×256×256`)

Evaluated on Mindboggle Pair `44` (`mbhard`, `192×256×256` at `1.0 mm`, 62 DKT cortical labels, shared `syntx.robust_affine` `.mat`, isolated fresh subprocesses on MPS).

### 3.1 Runtime & MPS Memory (Before `v6.39.0` vs. After Port)

| Solver Configuration | Total Time Before / After (s) | Speedup | Deformable Loop Before / After (s) | Peak Allocated VRAM Before / After (MB) | Allocated Delta (MB) | Peak Metal Driver Mem Before / After (MB) | Driver Mem Delta (MB) | Post-Fit Driver Mem Before / After (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `syn_3channel` (`cc2+mse+mattes`, `[40,40,15]`) | `46.26 / 46.11` | `1.003x` | `46.26 / 46.11` | `4,189.40 / 3,899.06` | **`-290.34 MB`** | `16,295.94 / 15,271.94` | **`-1,024.00 MB`** | `5,161.62 / 3,113.62` (**`-2,048.0 MB`**) |
| `syn_beltrami` (`cc2`, `[100,100,20]`) | `51.25 / 53.32` | `0.961x` | `51.25 / 53.32` | `4,088.53 / 3,703.85` | **`-384.69 MB`** | `14,231.94 / 14,231.94` | `0.00 MB` | `5,153.62 / 4,129.62` (**`-1,024.0 MB`**) |
| `tvf` (`beltrami`, `[100,100,20]`) | `212.12 / 213.37` | `0.994x` | `212.12 / 213.37` | `41,601.26 / 41,607.02` | `+5.76 MB` | `49,189.77 / 49,633.77` | `+444.00 MB` | `6,915.62 / 6,027.62` (**`-888.0 MB`**) |
| `syngs` (`cc2`, `[100,100,20]`) | `62.14 / 64.23` | `0.967x` | `62.14 / 64.23` | `6,941.42 / 6,947.45` | `+6.03 MB` | `12,282.88 / 11,788.02` | **`-494.86 MB`** | `5,145.62 / 6,169.62` |
| `greedy_default` (`[100,100,20]`, `adam`) | `8.87 / 8.21` | **`1.080x`** | **`3.26 / 2.69`** (**`1.212x`**) | `2,559.13 / 1,778.93` | **`-780.20 MB`** | `5,290.14 / 4,266.14` | **`-1,024.00 MB`** | `3,141.62 / 3,141.62` |
| `greedy_matched` (`[100,100,50]`, `regadam`) | `13.81 / 12.62` | **`1.095x`** | **`8.62 / 7.47`** (**`1.154x`**) | `2,703.19 / 1,779.06` | **`-924.13 MB`** | `5,238.14 / 4,214.14` | **`-1,024.00 MB`** | `3,129.62 / 2,113.62` (**`-1,016.0 MB`**) |
| `greedy_with_inv` (`[100,100,20]`, `return_inverse=True`) | `17.26 / 14.24` | **`1.212x`** | **`3.23 / 2.60`** (**`1.242x`**) | `2,959.26 / 2,839.38` | **`-119.88 MB`** | `6,367.77 / 5,395.03` | **`-972.73 MB`** | `3,137.62 / 3,137.62` |

### 3.2 Anatomical Accuracy (62-Label DKT Sørensen–Dice) & Diffeomorphism QC (`mbhard`)

| Solver Configuration | Bidir Mean Dice Before | Bidir Mean Dice After | Dice Delta | Fwd / Inv Dice After | Liouville Folding % Before / After | Min Jacobian Before / After | Mean ICE (mm) Before / After |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `syn_3channel` (`cc2+mse+mattes`) | `0.600713` | `0.600713` | `-0.000000` | `0.602876 / 0.598549` | `0.0000% / 0.0000%` | `0.0688 / 0.0688` | `0.0095 / 0.0095` |
| `syn_beltrami` (`cc2`) | `0.603627` | `0.603654` | `+0.000027` | `0.606078 / 0.601230` | `0.0000% / 0.0000%` | `0.0555 / 0.0594` | `0.0108 / 0.0108` |
| `tvf` (`beltrami`) | `0.627078` | `0.627104` | `+0.000026` | `0.630079 / 0.624130` | `0.0000% / 0.0000%` | `0.0263 / 0.0263` | `0.0031 / 0.0030` |
| `syngs` (`cc2`) | `0.619110` | `0.618982` | `-0.000128` | `0.620489 / 0.617475` | `0.0000% / 0.0000%` | `0.0334 / 0.0334` | `0.0036 / 0.0039` |
| `greedy_default` (`[100,100,20]`, `adam`) | `0.558960` | `0.558881` | `-0.000079` | `0.558881` (unidir) | `0.0000% / 0.0000%` | `0.0536 / 0.0534` | `0.0000 / 0.0000` |
| `greedy_matched` (`[100,100,50]`, `regadam`) | `0.570402` | `0.570488` | `+0.000086` | `0.570488` (unidir) | `0.0000% / 0.0000%` | `0.0117 / 0.0110` | `0.0000 / 0.0000` |
| `greedy_with_inv` (`return_inverse=True`) | `0.554470` | `0.554380` | `-0.000090` | `0.558710 / 0.550050` | `0.0000% / 0.0000%` | `0.0554 / 0.0537` | `0.0020 / 0.0020` |

---

## 4. Verification Gates & Reproducibility Commands

```bash
# 1. Single-judge static lint gate
PYTHONPATH=src python3 -m antsxqc.lint src/syntx

# 2. Unit & regression test suites
PYTHONPATH=src pytest tests/test_pr96_98_port.py tests/test_implementation_standards.py tests/test_pyramid_guard.py tests/test_landmarks_simulated.py tests/test_tvf_bugs.py tests/test_library_bugs.py tests/test_greedy.py -q

# 3. Standalone benchmark smoke gate (< 5 s)
PYTHONPATH=src python3 scripts/benchmark_pr96_98_port.py --smoke

# 4. Full isolated subprocess benchmark reproduction
PYTHONPATH=src python3 scripts/benchmark_pr96_98_port.py --output docs/provenance/pr96_98_port_benchmark.json
```
