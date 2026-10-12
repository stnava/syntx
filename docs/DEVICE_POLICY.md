# Compute device policy (syntx >= 6.5.0)

One resolver, `syntx.device.resolve_device(requested=None, task=None) -> str`, decides the compute device for the whole
antsx ecosystem. Apple Silicon first: MPS is preferred over the CPU everywhere a torch device is chosen.

## Precedence

1. `requested`, when it is not `None` / `"auto"`.
2. Environment. Registration tasks (`reg`, `registration`, `rigid_motion`, `rigid`, `affine`, `syn`): `ANTSXMM_REG_DEVICE`, then `ANTSXMM_DEVICE`. Other tasks: `ANTSXMM_DEVICE`. `auto` or empty falls through.
3. `cuda` > `mps` > `cpu`.

An explicit or environment device that is not available raises `syntx.device.DeviceUnavailableError`; an unknown type raises `ValueError`. Only the automatic path falls through. When `mps` is chosen `PYTORCH_ENABLE_MPS_FALLBACK=1` is set unless the user already set it. `resolve_device_with_source` also returns the label (`explicit`, `env:<NAME>`, `auto:<device>`).

## ComputeContext

`ComputeContext.model_device` / `reg_device` default to `"auto"` and are resolved in `__post_init__`; `reg_device_source` / `model_device_source` record how. A context is never `"auto"` and never silently `"cpu"` after construction. A caller that layers its own tier (antsxmm's per-task policy) passes the policy device as explicit and sets `reg_device_source="policy"`.

## Where CPU is correct

Allowed, with a reason in each repo's `tests/test_device_guard.py`: ANTs/ITK/numpy steps; float64 numerics (MPS has no float64: ROI correlation matrices, CompCor/SVD, the nuisance regression); JAX paths (cpu only); `torch.Generator(device="cpu")` seeding; `torch.load(map_location="cpu")`; low-level tensor helpers whose device follows the caller; explicit CPU baselines of a benchmark; ops with measured MPS penalties (`spatial.py` chunked op, about 150x slower on MPS; `features.py` model copied to CPU). To force CPU for a task: `ANTSXMM_REG_DEVICE=cpu`, `ANTSXMM_DEVICE=cpu`, or `device="cpu"`.

## Measured (Apple Silicon, SOCOM sub-Blast-01 rest BOLD, 40 volumes, batched rigid motion pass)

| backend | mps | cpu |
|---|---|---|
| `pytorch_batched_adaptive` | 0.79-1.03 s | 1.02-1.06 s |
| `pytorch_batched` | 19.7, 20.4 s | 51.6, 40.5 s |

So rigid motion is not faster on the CPU; the default is MPS for every task. The same SOCOM T1->BOLD SyN: 89 s on MPS, 236 s on CPU.

## Guard

`tests/test_device_guard.py` forbids new local `cuda.is_available()` / `mps.is_available()` probes and new `device="cpu"` defaults in `src/syntx` outside the allow-list, and forbids `reg_device="cpu"` / `model_device="cpu"` entirely.
