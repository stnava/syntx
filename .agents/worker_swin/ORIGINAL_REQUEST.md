## 2026-07-14T18:18:17Z

You are worker_swin, a worker agent.
Your working directory for metadata files is `/Users/stnava/code/syntx/.agents/worker_swin`. Please create this directory if it doesn't exist and write all coordination/metadata files there.
Your task is to implement Milestone 1: Swin UNETR 3D Encoder in the codebase.
Specifically, apply the proposed changes in `/Users/stnava/code/syntx/.agents/explorer_swin_1/analysis.md` to:
1. `src/syntx/features.py`: Implement `SwinUNETRExtractor` inheriting from `FeatureExtractor` with lazy MONAI loading, cached/downloaded weights from MONAI zoo at `~/.syntx_cache/model_swinvit.pt`, prefix stripping (`module.` and `swinViT.`), and input/output shape interpolation.
2. `src/syntx/__init__.py`: Import and expose `SwinUNETRExtractor`.
3. `src/syntx/syn.py`: Register metric name keys `'swinunetr'` and `'swin_unetr'` to instantiate `SwinUNETRExtractor`.
4. `tests/test_feature_networks.py`: Add the proposed unit tests (`test_swin_unetr_extractor_lazy_import`, `test_swin_unetr_extractor_shapes`, `test_swin_unetr_extractor_interpolation`, `test_swin_unetr_weights_download_and_key_cleaning`).
Run the test suite using pytest to verify that all existing and new unit tests pass and that coverage remains high.
Write a report of the implemented changes and test results to `/Users/stnava/code/syntx/.agents/worker_swin/handoff.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
