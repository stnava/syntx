# Environment and Codebase Exploration Report
*Date: 2026-07-14*
*Author: Codebase & Environment Explorer (explorer_1)*

## 1. Executive Summary
This report summarizes the codebase structure, execution environment, and dataset verification for the `syntx` registration benchmarking task. The testing suite is fully functional with 91/91 test cases passing. The PyTorch backend has MPS hardware acceleration enabled, while the JAX backend operates on CPU. The dataset consists of verified 2D phantoms and 3D native MRI volumes located in standard folders.

---

## 2. Environment Assessment

The installed Python packages, their versions, and hardware acceleration status are verified below:

| Package | Version | Import Path / Location |
|---|---|---|
| `ants-python` | `0.6.3` | `/Users/stnava/miniconda3/lib/python3.13/site-packages/ants` |
| `antspyt1w` | `1.1.3` | `/Users/stnava/miniconda3/lib/python3.13/site-packages/antspyt1w` |
| `antstorch` | `unknown` | `/Users/stnava/code/ANTsTorch/antstorch` |
| `torch` | `2.13.0` | `/Users/stnava/miniconda3/lib/python3.13/site-packages/torch` |
| `jax` | `0.10.2` | `/Users/stnava/miniconda3/lib/python3.13/site-packages/jax` |
| `jaxlib` | `0.10.2` | `/Users/stnava/miniconda3/lib/python3.13/site-packages/jaxlib` |

### Hardware Acceleration Status
* **PyTorch Acceleration**: **MPS (Metal Performance Shaders) is available** (`torch.backends.mps.is_available() == True`). CUDA is not available on this macOS system.
* **JAX Acceleration**: **CPU only**. The JAX backend resolves to `[CpuDevice(id=0)]`.

---

## 3. Image Dataset Verification

### 2D Phantoms
The 2D images `r16`, `r27`, and `r64` are located in the local package cache at `/Users/stnava/.antspy/`.
They were successfully loaded using `ants.get_data` and `ants.image_read`:

* **`r16`**: Shape `(256, 256)`, Voxel Spacing `(1.0, 1.0)`, path `/Users/stnava/.antspy/r16slice.jpg`
* **`r27`**: Shape `(256, 256)`, Voxel Spacing `(1.0, 1.0)`, path `/Users/stnava/.antspy/r27slice.jpg`
* **`r64`**: Shape `(256, 256)`, Voxel Spacing `(1.0, 1.0)`, path `/Users/stnava/.antspy/r64slice.jpg`

### 3D Scans & Target Template
The 3D MRI scans and the target template are located in `/Users/stnava/.antspyt1w/`.

| Image / Scan | File Path | Shape | Voxel Spacing | DKT Label Map Location |
|---|---|---|---|---|
| `28364-00000000-T1w-00` | `/Users/stnava/.antspyt1w/28364-00000000-T1w-00.nii.gz` | `(160, 256, 256)` | `(1.0, 1.0, 1.0)` | Needs generation via `antstorch` |
| `28386-00000000-T1w-01` | `/Users/stnava/.antspyt1w/28386-00000000-T1w-01.nii.gz` | `(160, 256, 256)` | `(1.0, 0.99999994, 1.0)` | Needs generation via `antstorch` |
| `28405-00000000-T1w-02` | `/Users/stnava/.antspyt1w/28405-00000000-T1w-02.nii.gz` | `(160, 256, 256)` | `(0.99999994, 1.0, 1.0)` | Needs generation via `antstorch` |
| `28478-00000000-T1w-03` | `/Users/stnava/.antspyt1w/28478-00000000-T1w-03.nii.gz` | `(160, 256, 256)` | `(1.0, 1.0, 1.0)` | Needs generation via `antstorch` |
| `28497-00000000-T1w-04` | `/Users/stnava/.antspyt1w/28497-00000000-T1w-04.nii.gz` | `(160, 256, 256)` | `(1.0, 1.0, 1.0)` | Cached: `cache/28497-00000000-T1w-04_dktseg.nii.gz` |
| `28523-00000000-T1w-05` | `/Users/stnava/.antspyt1w/28523-00000000-T1w-05.nii.gz` | `(160, 256, 256)` | `(0.99999994, 1.0, 1.0)` | Cached: `cache/28523-00000000-T1w-05_dktseg.nii.gz` |
| `28542-00000000-T1w-06` | `/Users/stnava/.antspyt1w/28542-00000000-T1w-06.nii.gz` | `(160, 256, 256)` | `(0.99999994, 1.0, 1.0)` | Needs generation via `antstorch` |
| `28575-00000000-T1w-07` | `/Users/stnava/.antspyt1w/28575-00000000-T1w-07.nii.gz` | `(160, 256, 256)` | `(0.99999994, 1.0, 1.0)` | Needs generation via `antstorch` |
| `T_template0` | `/Users/stnava/.antspyt1w/T_template0.nii.gz` | `(256, 256, 170)` | `(0.99325, 1.03279, 1.21504)` | Cached: `cache/T_template0_dktseg.nii.gz` |

#### Note on DKT Label Maps:
The target template (`T_template0`) and subjects `28497-00000000-T1w-04` and `28523-00000000-T1w-05` have pre-computed DKT label maps cached in the directory `/Users/stnava/code/syntx/cache/`. For any other scan, the segmentation can be generated dynamically using the `antstorch` interface:
```python
import antstorch
dktseg = antstorch.desikan_killiany_tourville_labeling(img, do_preprocessing=True, verbose=True)
```

---

## 4. Test Suite Execution & Coverage

We executed pytest with the `--runslow` flag to run all tests in the repository.

* **Command**: `pytest --runslow`
* **Execution Time**: 186.63 seconds (~3 mins 6 seconds)
* **Results**: **91 passed**, 0 failed, 0 skipped.
* **Warnings**: 6 minor warnings related to package deprecations and offline fallback weight warnings.

### Code Coverage Report
The project test coverage is exceptionally high, with an overall coverage rate of **93%**:

| File | Statements | Missed | Coverage |
|---|---|---|---|
| `src/syntx/__init__.py` | 7 | 0 | 100% |
| `src/syntx/features.py` | 319 | 19 | 94% |
| `src/syntx/resnet.py` | 75 | 0 | 100% |
| `src/syntx/syn.py` | 992 | 76 | 92% |
| `src/syntx/syn_jax.py` | 889 | 83 | 91% |
| `src/syntx/transform.py` | 96 | 0 | 100% |
| **TOTAL** | **2378** | **178** | **93%** |

---

## 5. Benchmark Configuration & Backends Analysis

We reviewed the benchmark templates and scripts in the `examples/` directory:
1. `examples/vgg_sweep_2d.py`: Sweeps configurations on 2D images (`r16`, `r64`) using VGG19 metrics.
2. `examples/compare_registration_backends_3d.py`: Benchmarks PyTorch and JAX backends against ANTs SyN on 3D/2D data.
3. `examples/evaluate_feature_metrics.py`: Evaluates modular perceptual similarity (resnet10, dinov2, combined LNCC-resnet) against cached segmentations.

### How Backends are Configured and Invoked
* **PyTorch Backend**:
  Invoked via `backend='pytorch'` in `syntx.registration(...)` or using `syntx.syn(...)` directly. Uses PyTorch optimizer and field compositions on GPU/MPS if available.
* **JAX Backend**:
  Invoked via `backend='jax'` in `syntx.registration(...)`. It utilizes the `SyNToJax` optimizer loop. It supports zero-copy autograd bridge (`DLPack` wrapper) to feed gradients back from PyTorch-based extractors to JAX registration updates.
* **Metric Customization**:
  The metrics are passed via the `similarity_metric` or `syn_metric` arguments. Multi-metric support allows combining intensity and deep feature metrics (e.g. `similarity_metric=['lncc', 'resnet10']` with `syn_metric_weights=[0.6, 0.4]`).

---

## 6. Registration Guardrails Compliance Check

We verified the codebase and scripts against the required guardrails in `GEMINI.md`:
1. **Single Interpolation Policy**: Affine and deformable warps must be composed and applied directly to native-space images in a single step (no intermediate resampling).
2. **VGG 3D Mode Requirement**: Standard brain label mappings must use **VGG 3D LNCC with Layer 4** (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`). 2D orthogonal slices or coarser layers degrade accuracy.
3. **Visualization Guidelines**: Performance comparison reports are required to display:
   * Edge/region overlaps between registered and target images.
   * Warped coordinate grids.
   * Jacobian determinant maps.
   * Side-by-side deformed and target images.
