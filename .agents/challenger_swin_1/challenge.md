# SwinUNETRExtractor Empirical Verification Report

## Challenge Summary

**Overall risk assessment**: CRITICAL

The current implementation of `SwinUNETRExtractor` in `src/syntx/features.py` is **completely broken** in practice due to an initialization argument error when using standard MONAI releases. Furthermore, even if initialization is bypassed or patched, the extractor contains a critical spatial downsampling calculation bug that corrupts the feature map dimensions when processing volumes whose shapes differ from the configured `img_size`. Finally, the existing test suite has broken imports that make tests fail to compile when MONAI is installed.

---

## Challenges

### [Critical] Challenge 1: Instantiation Failure via Invalid Keyword Argument `img_size`

- **Assumption challenged**: The MONAI `SwinUNETR` class constructor accepts an `img_size` parameter.
- **Attack scenario**: Attempting to initialize the `SwinUNETRExtractor` class using `extractor = SwinUNETRExtractor()` raises a `TypeError` and crashes.
- **Blast radius**: The `SwinUNETRExtractor` cannot be initialized or used in any registration pipeline, rendering it dead code.
- **Mitigation**: Remove `img_size=self.img_size` from the `SwinUNETR` initialization call in `src/syntx/features.py`. MONAI's `SwinUNETR` constructor does not require or accept `img_size`. Instead, `SwinUNETR` determines output spatial shapes dynamically based on input tensor size and downsampling steps.

---

### [High] Challenge 2: Spatial Misalignment in Interpolation Output due to Incorrect Downsampling Formula

- **Assumption challenged**: The feature map downsampling factor at a given layer of MONAI's SwinUNETR corresponds to $2^{\text{layer}}$.
- **Attack scenario**: Feeding input volumes with dimensions different from `img_size` (e.g., $64 \times 64 \times 64$ inputs with a configured `img_size` of $96 \times 96 \times 96$) triggers output feature interpolation back to the target scale. The code calculates the target scale using the formula `expected_shape = [max(1, s // (2 ** layer)) for s in spatial_shape]`.
- **Blast radius**: 
  - For MONAI's `SwinUNETR` backbone:
    - Layer 1 outputs are downsampled by a factor of 4 (shape $24^3$ for $96^3$ input). The formula $2^{\text{layer}} = 2^1 = 2$ scales it to $64 // 2 = 32^3$ instead of $64 // 4 = 16^3$.
    - Layer 2 outputs are downsampled by a factor of 8. The formula $2^2 = 4$ scales it to $64 // 4 = 16^3$ instead of $8^3$.
    - Layer 3 outputs are downsampled by a factor of 16. The formula $2^3 = 8$ scales it to $8^3$ instead of $4^3$.
    - Layer 4 outputs are downsampled by a factor of 32. The formula $2^4 = 16$ scales it to $4^3$ instead of $2^3$.
  - Consequently, all feature map sizes returned by `SwinUNETRExtractor.extract` under interpolation are scaled to twice their correct spatial dimensions. This mismatch breaks registration spatial alignment because feature maps are geometrically warped and computed with mismatched grids, degrading registration accuracy.
- **Mitigation**: Correct the downsampling factor formula to match the actual `SwinUNETR` backbone properties:
  `expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]`

---

### [High] Challenge 3: Broken Test Imports for weights loading and backbone tests

- **Assumption challenged**: `SwinViT` is directly importable from `monai.networks.nets`.
- **Attack scenario**: Running pytest when MONAI is installed fails on tests trying to import `SwinViT`.
- **Blast radius**: The entire suite of SwinUNETR tests in `tests/test_feature_networks.py` crashes due to `ImportError`.
- **Mitigation**: Update `tests/test_feature_networks.py` to import `SwinTransformer` from `monai.networks.nets.swin_unetr` (which is the actual underlying class instantiated in `SwinUNETR.swinViT`).

---

### [Medium] Challenge 4: High CPU Latency and Memory Footprint

- **Assumption challenged**: SwinUNETR features can be extracted efficiently during iteration cycles.
- **Attack scenario**: Running registration optimization loops on standard hardware where CPU is used (e.g., standard workstation or server with CPU only).
- **Blast radius**: Each forward pass of `SwinUNETRExtractor` takes approximately **0.8 seconds** on an Apple M-series CPU. An optimization loop of 100 iterations will require at least 80 seconds just for feature extraction, which is extremely slow and resource-intensive compared to standard intensity-based LNCC or lightweight ResNet-10 extractors.
- **Mitigation**: Documnet performance expectations and restrict or warn users when running SwinUNETR-based registration on CPU resources.

---

## Stress Test Results

| Scenario | Expected Behavior | Actual Behavior | Pass/Fail |
|---|---|---|---|
| Initialize `SwinUNETRExtractor` | Successful instance creation | Crashed with `TypeError: SwinUNETR got unexpected keyword argument 'img_size'` | **FAIL** |
| Extract features on $96^3$ input (native size) | Extracted shapes: [1, 96, 24, 24, 24] at L1, [1, 768, 3, 3, 3] at L4 | Same as expected | **PASS** (when constructor is monkeypatched) |
| Extract features on $64^3$ input (interpolated size) | Feature map sizes matching correct downsampling factors (e.g. L4 shape $2^3$) | Feature map sizes scaled using incorrect $2^{\text{layer}}$ factor (e.g. L4 shape $4^3$) | **FAIL** (when constructor is monkeypatched) |
| Run test suite with MONAI installed | Clean test execution (either passing or gracefully skipping) | Crashed with `ImportError` on `SwinViT` import | **FAIL** |
| Simulated offline environment | Extractor warns about download failure and initializes with random weights | Extractor issued warning and initialized successfully | **PASS** (when constructor is monkeypatched) |

---

## Unchallenged Areas

- **GPU Performance / VRAM utilization**: We did not benchmark GPU execution profile due to the verification sandbox being restricted to CPU-only execution. However, GPU memory pressure for $96^3$ SwinUNETR forward passes is expected to be high due to heavy self-attention maps.
