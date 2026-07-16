# BRIEFING — 2026-07-14T18:16:21Z

## Mission
Analyze the integration of SwinUNETRExtractor in syntx features and propose tests.

## 🔒 My Identity
- Archetype: explorer
- Roles: Teamwork explorer, Investigation
- Working directory: /Users/stnava/code/syntx/.agents/explorer_swin_2
- Original parent: 48bf2a28-238f-4da4-9d3d-4b00215bbd4c
- Milestone: SwinUNETR Integration Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Single Interpolation Policy: No intermediate file-based pre-warping; compose transforms.
- Similarity Metric & VGG: Must adhere to GEMINI.md constraints (e.g. DICE, VGG 3D mode).

## Current Parent
- Conversation ID: 48bf2a28-238f-4da4-9d3d-4b00215bbd4c
- Updated: 2026-07-14T18:17:45Z

## Investigation State
- **Explored paths**:
  - `src/syntx/features.py`: Base classes, resnet, VGG, DINO extractors, and FeatureSpaceLoss structures.
  - `src/syntx/syn.py`: Target location for adding the metric selection routing.
  - `src/syntx/syn_jax.py`: Explored for metric routing (only standard similarity metrics in native JAX).
  - `tests/test_feature_networks.py`: Existing tests for shapes, DINOv2 pruning, feature loss, and resnet.
  - `tests/test_syn_jax.py`: Checked JAX registration tests and patterns.
- **Key findings**:
  - Designed the lazy loading logic for `SwinUNETRExtractor` to gracefully support environments without MONAI.
  - Determined that positional embeddings in MONAI's SwinViT require input volumes to match its constructed size. Designed dynamic interpolation in `extract` to resolve this and allow arbitrary image shapes.
  - Proposed clean-up logic for weight loading to handle prefixes (`module.` and `swinViT.`).
  - Proposed 4 mock-based unit tests to cover error paths, shape matching, interpolation, and prefix handling in `tests/test_feature_networks.py`.
- **Unexplored areas**:
  - Full E2E testing of SwinUNETR with actual weights, which is a downstream milestone.

## Key Decisions Made
- Opted for mock-based testing of optional `monai` dependencies to guarantee execution in a variety of testing/CI environments.
- Designed dual interpolation (interpolation to target size for backbone, then interpolation of feature maps back to expected scale factor `2**(layer+1)`) to maintain dimensional consistency.

## Artifact Index
- /Users/stnava/code/syntx/.agents/explorer_swin_2/ORIGINAL_REQUEST.md — Original task description
- /Users/stnava/code/syntx/.agents/explorer_swin_2/analysis.md — Complete SwinUNETR integration design and unit tests plan
