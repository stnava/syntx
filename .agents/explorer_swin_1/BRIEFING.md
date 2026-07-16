# BRIEFING — 2026-07-14T14:16:21-04:00

## Mission
Explore integrating SwinUNETRExtractor into src/syntx/features.py and design tests for it.

## 🔒 My Identity
- Archetype: explorer_swin_1
- Roles: Teamwork explorer, Read-only investigation
- Working directory: /Users/stnava/code/syntx/.agents/explorer_swin_1
- Original parent: 48bf2a28-238f-4da4-9d3d-4b00215bbd4c
- Milestone: SwinUNETR integration investigation

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Code-only network mode (no external network access, etc.)

## Current Parent
- Conversation ID: 48bf2a28-238f-4da4-9d3d-4b00215bbd4c
- Updated: 2026-07-14T14:16:21-04:00

## Investigation State
- **Explored paths**:
  - `src/syntx/features.py`
  - `src/syntx/__init__.py`
  - `src/syntx/syn.py`
  - `tests/test_feature_networks.py`
  - `.agents/explorer_swin_2/analysis.md`
  - `.agents/explorer_swin_3/analysis.md`
  - `.agents/explorer_swin_3/proposed_changes.patch`
- **Key findings**:
  - Consensus: Dynamic lazy importing of MONAI to avoid global dependencies; download pre-trained weights from MONAI Zoo into `~/.syntx_cache/model_swinvit.pt`; clean state dict keys sequentially to strip `module.` and `swinViT.` prefixes.
  - Conflict/Resolution: SwinUNETR's positional embeddings crash if input size does not match initialized `img_size` (e.g. at coarse levels of the multi-resolution pyramid). Adopted `explorer_swin_2`'s dynamic input/output shape interpolation to resolve this.
- **Unexplored areas**:
  - Integration of PyTorch feature losses in JAX SyN loops (Milestone 4).

## Key Decisions Made
- Concluded that dynamic input interpolation is crucial for multi-resolution SyN registration.
- Selected sequential key cleansing (`module.` then `swinViT.`) as the most robust way to parse checkpoint weights.
- Designed comprehensive mocked unit tests for offline-safe environments.

## Artifact Index
- `/Users/stnava/code/syntx/.agents/explorer_swin_1/analysis.md` — Final synthesis and plan for SwinUNETRExtractor.
