# External Registration Benchmark Baselines (`VoxelMorph` & `FireANTs`)

This guide documents the production setup, physical-space adapters, `antsxdata` weight provenance, root-cause systems fixes, and replayable commands for benchmarking **VoxelMorph (`0.3.3`)** and **repaired FireANTs (`1.5.1`)** against `syntx` on Apple Silicon MPS and CPU.

For the causal remediation of `FireANTs` on MPS and bidirectional/inverse geometry, see [`docs/v_fire_ants/FIREANTS_UNIDIRECTIONAL_REMEDIATION.md`](v_fire_ants/FIREANTS_UNIDIRECTIONAL_REMEDIATION.md).

---

## 1. Environment Installation (`--no-deps` on Python 3.13 + Apple Silicon MPS)

To keep `torch`, `numpy`, `scipy`, `antspyx`, `jax`, `antsxqc`, and `syntx` pinned at their verified ecosystem versions without pulling TensorFlow or conflicting dependencies, install `pystrum`, `neurite`, `voxelmorph`, and the editable repaired `FireANTs` checkout strictly with `--no-deps`:

```bash
# 1. VoxelMorph 0.3.3 PyTorch stack (no TensorFlow, no dependency downgrades)
/Users/stnava/miniconda3/bin/python3 -m pip install --no-deps pystrum==0.4 neurite==0.3.4
/Users/stnava/miniconda3/bin/python3 -m pip install --no-deps git+https://github.com/voxelmorph/voxelmorph.git

# 2. Repaired FireANTs 1.5.1 editable checkout (see docs/v_fire_ants/FIREANTS_UNIDIRECTIONAL_REMEDIATION.md)
/Users/stnava/miniconda3/bin/python3 -m pip install --no-deps -e /Users/stnava/code/FireANTs

# 3. Verify imports & versions
/Users/stnava/miniconda3/bin/python3 -c "
import importlib.metadata, torch, voxelmorph as vxm, neurite as ne, pystrum, fireants, syntx, antsxdata
print('syntx:', syntx.__version__, 'torch:', torch.__version__, 'mps:', torch.backends.mps.is_available())
print('voxelmorph:', importlib.metadata.version('voxelmorph'), 'neurite:', importlib.metadata.version('neurite'), 'pystrum:', importlib.metadata.version('pystrum'))
print('fireants:', importlib.metadata.version('fireants'), fireants.__file__)
"
```

---

## 2. `antsxdata` Registry Specification (`syntx_features/voxelmorph_pairwise_3d_brain`)

Official DalcaLab PyTorch 3-D brain weights are registered in `/Users/stnava/code/antsxdata/src/antsxdata/registry/syntx_features.toml`:

- **Asset Reference**: `syntx_features/voxelmorph_pairwise_3d_brain`
- **Upstream URL**: `https://huggingface.co/DalcaLab/voxelmorph_pairwise_3d_tutorial/resolve/main/voxelmorph_pairwise_3d_model.pt`
- **License**: MIT (`DalcaLab/voxelmorph_pairwise_3d_tutorial`)
- **File Size**: `435,477` bytes
- **SHA-256**: `94c7094b79b54bb7d80c7e83103a11cc03d33cfcc779b13ba09ca60913d14b85`
- **Architecture**: `voxelmorph.nn.models.VxmPairwise(ndim=3, source_channels=1, target_channels=1, nb_features=(16, 16, 16, 16, 16), integration_steps=7)` (5-level U-Net requiring spatial dimensions divisible by $2^5 = 32$, handled via slice-copy symmetric padding `_pad_to_multiple(..., 32)` to avoid 5-D `F.pad` on MPS).

### Lossless Canonical Orientation Permutation (Zero Extra Interpolation)

Pretrained 3-D `3x3x3` convolutional kernels are not equivariant to storage orientation (e.g. `mbhard` Fixed `NKI-TRT-20-2` has ITK direction $\text{diag}(1, -1, 1)$). Instead of calling `reorient_image2` or resampling the input grid:

1. `syntx.benchmark.external.canonical_orientation_transform(fixed.direction, target_orientation="PIR")` computes the signed axis permutation $(\pi, s) \in S_3 \times \{\pm 1\}^3$ maximizing column alignment $D^\top C_{\text{target}}$ in ITK LPS physical space.
2. `apply_axis_permutation(tensor, perm, flips)` applies `torch.permute` + `torch.flip` losslessly to the affine-aligned `[0, 1]` volumes (`"PIR"` gave the highest empirical zero-shot DKT Dice `0.4757` NN / `0.4856` GL vs. `0.4670` NN in native `LAS` across all 48 signed permutations on `mbhard`).
3. `invert_displacement_permutation(disp_vox, perm, flips)` un-flips spatial axes, flips component signs $s_i$, and applies $\pi^{-1}$ to both channels and spatial dimensions before `voxel_disp_to_itk_lps` maps voxel shifts via $J_{\text{vox}\to\text{LPS}} = D \cdot \operatorname{diag}(\text{spacing})$ to ITK LPS physical `1Warp.nii.gz`.

---

## 3. Root-Cause Fixes for VoxelMorph on MPS & Physical Space

In the unoptimized single-scale pilot (`pilot_voxelmorph_fireants_mbhard.py`), single-scale `VxmPairwise` at `int_downsize=2` took `184.3 s` and reached only `0.4360` GL Dice. Profiling on Apple Silicon MPS identified four bottlenecks, all resolved in `src/syntx/benchmark/external.py`:

1. **Sync-free $5^3$ LNCC (`local_ncc_loss_nd`)**: `neurite.nn.modules.NCC(window_size=[9, 9, 9])` executes host-blocking `assert torch.all(valid)` synchronizations and dense $9^3$ convolutions on every step. Replacing it with `syntx.core.losses.cross_correlation.local_ncc_loss_nd(..., window_size=5, squared=True)` (`var_floor=1e-6`, foreground mask `tgt > 0.02`) eliminates host syncs and computes $5^3$ LNCC in `9 ms` at `192×256×256`.
2. **Centralized `[4, 2, 1]` Pyramid (`syntx.pyramid.build_pyramid`)**: Coarse-to-fine levels are built strictly via `syntx.pyramid.build_pyramid(..., levels=(4, 2, 1), policy="deformable")`, and velocity fields are upsampled across levels via `syntx.core.grid.resize_field` scaled by axis voxel ratios.
3. **MPS 3-D U-Net Backward Scaling**: On MPS, `BasicUNet` `Conv3d` backward takes `~6.6 s/step` at `192×256×256` (`L1`), `~0.82 s/step` at `96×128×128` (`L2`), and `~0.20 s/step` at `48×64×64` (`L4`), whereas `IntegrateVelocityField(7) + SpatialTransformer` backward at full `192×256×256` takes only `~0.06 s/step`. Accordingly:
   - `run_voxelmorph_multiscale` (`voxelmorph_svf`) trains `VxmPairwise` weights at coarse `L4` (`100` steps) and optimizes the upsampled SVF velocity directly at `L2` (`100` steps) and `L1` (`50` steps with cosine decay + `final_smooth_sigma=0.45`), achieving **`0.5186` NN / `0.5291` GL Dice** in **`34.58 s`** (`5.33×` faster and `+9.31 pp` higher GL Dice than the `184.3 s` pilot).
   - `run_voxelmorph_pretrained` with `polish_steps=30` (`voxelmorph_polished`) initializes the SVF velocity from the pretrained `VxmPairwise` forward pass (`0.51 s`) and runs 30 steps of full-resolution `L1` SVF optimization (`polish_lr=0.12`, `grad_weight=0.30`, `vel_smooth_sigma=0.40`), reaching **`0.5221` NN / `0.5335` GL Dice** with **`0.0000%` folding** in **`7.40 s`**.
4. **Single-Interpolation Physical Export & `antsxqc` / `syntx.provenance` Parity**: Every adapter composes `[1Warp.nii.gz, initial_transform]` once in `syntx.transforms.apply_transforms`, leaves `invtransforms=[]` when `return_inverse=False`, and is registered in `syntx.provenance._ENTRY_POINTS` so `capture_registration_calls()` + `assert_manifest_complete(..., require_calls=True)` record exact kwargs, versions, and `asset_sha256`.

---

## 4. Replay Commands (CLI & Python API)

### CLI Benchmark (`benchmarks/fast_syn_ants_benchmark.py`)

```bash
# Smoke gate (standalone)
/Users/stnava/miniconda3/bin/python3 benchmarks/fast_syn_ants_benchmark.py --smoke

# Smoke gate including all external/unidirectional arms
/Users/stnava/miniconda3/bin/python3 benchmarks/fast_syn_ants_benchmark.py --smoke --include-external

# Full mbhard (Pair 44) benchmark on isolated MPS (Beltrami + external baselines)
/Users/stnava/miniconda3/bin/python3 benchmarks/fast_syn_ants_benchmark.py \
    --pairs 44 \
    --best-models-only \
    --regularizers beltrami \
    --include-external \
    --device mps \
    --out-dir results/external_mbhard \
    --report-stem docs/reports/external_mbhard
```

### Python API (`syntx.benchmark.external`)

```python
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import normalize_intensity
from syntx.benchmark.external import (
    run_fireants_greedy,
    run_voxelmorph_multiscale,
    run_voxelmorph_pretrained,
)

pair = load_mindboggle_pair(pair_idx=44, use_n4=True)
fi = normalize_intensity(pair["fixed"])
mi = normalize_intensity(pair["moving"])
aff = syntx.robust_affine(fi, mi, mode="auto", device="mps", seed=42)["fwdtransforms"][0]

res_zero = run_voxelmorph_pretrained(fi, mi, initial_transform=aff, polish_steps=0, device="mps")
res_pol  = run_voxelmorph_pretrained(fi, mi, initial_transform=aff, polish_steps=30, device="mps")
res_ms   = run_voxelmorph_multiscale(fi, mi, initial_transform=aff, scales=(4, 2, 1), reg_iterations=(100, 100, 50), device="mps")
res_fa   = run_fireants_greedy(fi, mi, initial_transform=aff, scales=(4, 2, 1), reg_iterations=(100, 100, 50), device="mps")
```

---

## 5. Verified Shared-Affine `mbhard` (Pair 44) Results on Isolated Apple MPS

Measured sequentially on isolated Apple Silicon MPS (`Pair 44`: `NKI-TRT-20-2` `192×256×256` `RPI` $\leftarrow$ `MMRR-21-2` `170×256×256` `LAS`, N4 + `[2%, 98%]` foreground normalized, shared `syntx.robust_affine` `.mat`, single-interpolation label warping). Unidirectional methods (`invtransforms == []`) report Fixed-Space DKT Sørensen–Dice (`NN` and `genericLabel`).

| Arm | Directionality | Schedule / Variant | Fixed Dice (`NN`) | Fixed Dice (`genericLabel`) | Folding % | Min $\det J$ | MPS Runtime (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `syntx_robust_affine` | Invertible (`.mat`) | `mode='auto'` (`pt7`) | `0.3190` | `0.3226` | `0.0000%` | `+1.0000` | `9.14 s` |
| `voxelmorph_pretrained` | Unidirectional (`SVF`) | Zero-shot (`PIR` canonical, `polish_steps=0`) | `0.4757` | `0.4856` | `0.0594%` | `-5.4580` | `0.51 s` |
| `voxelmorph_polished` | Unidirectional (`SVF`) | Pretrained + `polish_steps=30` (`5^3` LNCC) | `0.5221` | `0.5335` | `0.0000%` | `+0.0340` | `7.40 s` |
| `voxelmorph_svf` | Unidirectional (`SVF`) | `[4, 2, 1]` `(100, 100, 50)` (`5^3` LNCC) | `0.5186` | `0.5291` | `0.0004%` | `-0.1866` | `34.58 s` |
| `syntx_greedy` (default) | Unidirectional (`Compositive`) | `[100, 100, 20]`, `adam` | `0.5639` | `0.5776` | `0.0000%` | `+0.0538` | `8.36 s` (`2.57 s` loop) |
| `syntx_greedy` (matched) | Unidirectional (`Compositive`) | `[100, 100, 50]`, `regadam` | `0.5716` | `0.5857` | `0.0000%` | `+0.0127` | `12.35 s` (`7.28 s` loop) |
| `fireants_greedy` (repaired) | Unidirectional (`Compositive`) | `[100, 100, 50]`, `regadam` | `0.5786` | `0.5932` | `0.0010%` | `-0.0058` | `27.37 s` |
| `syntx_syn_beltrami` | Symmetric Bidirectional | `[100, 100, 20]`, `beltrami` (`dil=0.1`) | **`0.6165`** | **`0.6338`** | **`0.0000%`** | **`+0.0592`** | `49.14 s` |

*(For `syntx_syn_beltrami`: Moving `NN` Dice = `0.6076`, Symmetric `NN` Dice = `0.6121`, Inverse Consistency Error mean / p95 = `0.0108 / 0.0726 mm`.)*
