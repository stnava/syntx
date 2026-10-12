# syntx 6.5.0

Central compute-device resolver; MPS-first everywhere (design: `docs/DEVICE_POLICY.md`). Correctness over backward compatibility.

## Added
* `syntx.device`: `resolve_device(requested=None, task=None)`, `resolve_device_with_source`, `auto_device`, `available_devices`, `is_available`, `DeviceUnavailableError`, `REGISTRATION_TASKS`. Precedence explicit > env (`ANTSXMM_REG_DEVICE` for registration tasks, `ANTSXMM_DEVICE`) > cuda > mps > cpu; an unavailable explicit/env device raises; `PYTORCH_ENABLE_MPS_FALLBACK=1` is set when mps is chosen.
* `ComputeContext.model_device_source`.
* `syntx.motion_correction(..., device=)`: the PyTorch backends now take a device (it was never forwarded; the batched passes used a private `auto`), recorded in `summary['device']` / `summary['device_source']`. `backend='ants'` is CPU-only and rejects other devices.
* Tests: `tests/test_device_resolver.py`, `tests/test_device_guard.py`.

## Changed (breaking on purpose)
* `ComputeContext.model_device` / `reg_device` default to `"auto"` and are resolved at construction (were `"cpu"`: every standalone pipeline that built `ComputeContext()` ran on the CPU). `reg_device_source` is now a free label (`explicit`, `env:<NAME>`, `auto:<device>`, caller tiers) instead of `cache|benchmark|override|default`.
* All local `cuda > mps > cpu` copies in syntx (motion_batched, core.pipeline.auto_detect_device, syn, syngs, robust_affine, slab_capture, cli, landmarks, benchmark.evaluate/worker/data, viz.reports, classifier, diagnose) use the central resolver; the landmarks modules were mps > cuda > cpu and are now cuda > mps > cpu. An explicit unavailable device now raises instead of failing later or degrading.
* `predict_diagnosis_deep(device=None)` and the benchmark N4 step resolve a device instead of defaulting to `"cpu"`.

## Evidence
Rigid motion, 40 SOCOM BOLD volumes: `pytorch_batched` mps 19.7-20.4 s vs cpu 40.5-51.6 s; `pytorch_batched_adaptive` mps 0.8-1.0 s vs cpu 1.0-1.1 s. T1->BOLD SyN: mps 89 s vs cpu 236 s.
