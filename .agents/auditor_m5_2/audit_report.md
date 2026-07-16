## Forensic Audit Report

**Work Product**: final syntx codebase
**Profile**: General Project
**Verdict**: CLEAN

### Phase Results
- **Check 1: Cheating, facade code, hardcoded test results**: PASS — Checked all codebase files (`syn.py`, `syn_jax.py`, `features.py`, `resnet.py`, `transform.py`) and test files (`test_syn.py`, `test_syn_jax.py`, `test_challenger_verification.py`, etc.). No hardcoded test results, facade implementations, or cheating were found. All algorithms are authentic and run real optimization loops.
- **Check 2: Single Interpolation Policy**: PASS — Verified that `syn.py` and `syn_jax.py` compose initial transforms with learned affine and deformable grids entirely in coordinate space, executing a single `ants.apply_transforms` call at the end on the native-space images. No intermediate file-based or array-based pre-warping of input images or segmentations occurs prior to optimization.
- **Check 3: VGG Feature Space Guidelines**: PASS — Default parameters in `registration()` and `SyNTo.fit()` are correctly configured to `vgg_layers=[4]` and `vgg_mode='lncc_3d'`. VGG 2D mode is not recommended or defaulted for accuracy tasks.

### Evidence

#### Test Suite Output
All 92 tests passed with 92% statement coverage:
```
============================= test session starts ==============================
platform darwin -- Python 3.13.2, pytest-9.0.2, pluggy-1.5.0
rootdir: /Users/stnava/code/syntx
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0
collected 98 items

tests/test_challenger_verification.py ....                               [  4%]
tests/test_coverage_helpers.py ........................                  [ 28%]
tests/test_e2e_metrics.py ...........................                    [ 56%]
tests/test_feature_networks.py ...........                               [ 67%]
tests/test_swin_unetr_empirical.py .....                                 [ 72%]
tests/test_syn.py ..ss..ss...                                            [ 83%]
tests/test_syn_jax.py ..ss..........                                     [ 97%]
tests/test_transform.py ..                                               [100%]

================================ tests coverage ================================
_______________ coverage: platform darwin, python 3.13.2-final-0 _______________

Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
src/syntx/__init__.py        7      0   100%
src/syntx/features.py      319     19    94%   97, 129-130, 169, 368-369, 372-373, 383-384, 387-388, 398-399, 402-403, 431, 440, 449
src/syntx/resnet.py         75      0   100%
src/syntx/syn.py          1040     87    92%   185, 267, 501-503, 822-826, 849, 854, 894, 914-915, 919-920, 928-939, 949, 1006-1019, 1094-1095, 1104, 1148-1154, 1283-1290, 1298-1318, 1322-1324, 1333-1335, 1394, 1535, 1734-1735, 1749-1751, 1757-1759, 1762-1769, 1775-1777
src/syntx/syn_jax.py       926     82    91%   60-61, 79-80, 83, 105-106, 137-138, 157, 239, 599, 696, 722-729, 837-841, 986, 991, 1030, 1055-1073, 1078, 1082, 1085-1088, 1126, 1158-1166, 1238, 1285, 1301-1302, 1314-1316, 1333, 1403-1404, 1412, 1440, 1443-1446, 1454-1457
src/syntx/transform.py      96      0   100%
------------------------------------------------------
TOTAL                     2463    188    92%
============ 92 passed, 6 skipped, 6 warnings in 134.52s (0:02:14) =============
```

#### Code References

1. **Single Interpolation Policy check**:
   In `src/syntx/syn.py` (lines 1531-1536):
   ```python
   tx_list = []
   initial_grid = None
   if initial_transform is not None:
       tx_list = initial_transform if isinstance(initial_transform, list) else [initial_transform]
       initial_grid = compute_initial_grid(fixed, moving, tx_list)
       if affine_iterations is None:
           affine_iterations = [0]
   moving_reg = moving
   ```
   No pre-warping of the `moving` image is done.
   
   Final warped output images are computed in a single step (lines 1780-1781):
   ```python
   warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_transforms)
   warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=inv_transforms, whichtoinvert=whichtoinvert_inv)
   ```

2. **VGG parameters default settings**:
   In `src/syntx/syn.py` (lines 1473-1478):
   ```python
   def registration(
       fixed,
       moving,
       ...
       vgg_layers=[4],
       vgg_mode='lncc_3d',
       vgg_patch_size=32,
       vgg_num_patches=8,
       vgg_lncc_window_size=9,
       ...
   ):
   ```
   In `src/syntx/syn.py` (lines 832-833):
   ```python
   def fit(self, fixed_image, moving_image, ...,
           vgg_layers=[4], vgg_patch_size=32, vgg_num_patches=8, vgg_mode='lncc_3d',
           vgg_lncc_window_size=9, ...):
   ```
