# BRIEFING — 2026-07-15T09:15:40-04:00

## Mission
Implement the comprehensive image comparison metrics suite in `src/syntx/image_compare.py` and expose it.

## 🔒 My Identity
- Archetype: implementer_qa_specialist
- Roles: implementer, qa, specialist
- Working directory: /Users/stnava/code/syntx/.agents/worker_m2
- Original parent: 090034f8-59e0-4293-872b-02443d4b77b8
- Milestone: image_compare_suite

## 🔒 Key Constraints
- Single Interpolation Policy: no intermediate file-based pre-warping.
- VGG 3D LNCC Layer 4 Requirement: Only VGG 3D LNCC with Layer 4 meets performance targets for VGG Layer 4 LNCC.
- Return standardized score: lower score indicates better similarity.
- Support both 2D and 3D images for all metrics.
- Support inputs `a` and `b` as ANTsImage, PyTorch tensors, JAX arrays, or NumPy arrays.
- Support at least 64 unique metric names.

## Current Parent
- Conversation ID: 090034f8-59e0-4293-872b-02443d4b77b8
- Updated: yes

## Task Summary
- **What to build**: Comprehensive image comparison metrics suite in `src/syntx/image_compare.py` with `image_compare` function supporting >= 64 configurations.
- **Success criteria**: All metrics correct, lower score is better, supports various formats, passes builds/tests, integrated in `src/syntx/__init__.py`.
- **Interface contracts**: `def image_compare(a, b, metricname: str, **kwargs) -> float:`
- **Code layout**: `src/syntx/image_compare.py`

## Key Decisions Made
- Implemented 88 metrics to comfortably exceed the minimum of 64 configurations, covering classical, spatial/gradient, and deep feature extractors (VGG19, DINOv2, ResNet10, SwinUNETR).
- Custom PyTorch-based SSIM and Multi-Scale SSIM implementation supporting both 2D and 3D images dynamically adjusting scales based on image size.
- Leveraged `FeatureSpaceLoss` for LNCC deep feature losses, complying with VGG 3D LNCC Layer 4 and Single Interpolation Policy.
- Integrated parent's JAX compile-time fix in `tests/test_syn_jax.py` to ensure build status is green.

## Change Tracker
- **Files modified**:
  - `src/syntx/image_compare.py` - Created suite implementation.
  - `src/syntx/__init__.py` - Exported `image_compare`.
  - `tests/test_syn_jax.py` - Fixed JAX helper test call.
  - `tests/test_image_compare.py` - Added comprehensive test suite.
- **Build status**: PASS
- **Pending issues**: None

## Quality Status
- **Build/test result**: PASS (113 passed, 6 skipped)
- **Lint status**: PASS
- **Tests added/modified**: `tests/test_image_compare.py` (8 new test cases testing 88 configurations)

## Loaded Skills
- **Source**: none
- **Local copy**: none
- **Core methodology**: none

## Artifact Index
- `/Users/stnava/code/syntx/.agents/worker_m2/ORIGINAL_REQUEST.md` — Original request copy.
- `/Users/stnava/code/syntx/.agents/worker_m2/BRIEFING.md` — This briefing document.
