# Benchmark Configuration Schema (`run_config.json`)

This file defines the complete parameter set for reproducible syntx registration benchmarks.

## Schema

### `syn_config` — SyN Registration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grad_step` | float | 0.25 | CFL gradient step size (voxel units). Controls max displacement per iteration. Use 0.05 for 0% folding parity with ANTs C++. |
| `fluid_sigma` | float | 3.0 | ITK variance convention for fluid regularization (σ² = 3.0, actual σ = √3 ≈ 1.732). |
| `elastic_sigma` | float | 0.0 | ITK variance for total/elastic field smoothing. 0.0 = pure fluid deformation. |
| `lncc_radius` | int | 2 | LNCC window half-size. Window = 2*radius+1 (radius=2 → 5×5×5). |
| `inverse_steps` | int | 10 | Fixed-point iterations for computing the inverse warp at each optimization step. |
| `syn_regularizer` | str | "gaussian" | Regularization method: "gaussian" (ITK-compatible), "sobolev", "dsti" (spectral), "dsti1" (separable 1D DST-I, MPS-safe). |
| `syn_fast_smooth` | bool | false | If true, use approximate fast smoothing. |
| `syn_use_analytical_gradients` | bool | false | If true, use analytical chain-rule gradients instead of PyTorch autograd. |
| `syn_inverse_method` | str | "anderson" | Inverse computation method: "anderson" or "fixed_point". |
| `reg_iterations` | list[int] | [100, 100, 20] | Multi-resolution iteration schedule [coarse, medium, fine]. |

### `tvf_config` — Time-Varying Velocity Field Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tvf_grad_step` | float | 0.211 | CFL gradient step size (voxel units). |
| `tvf_flow_sigma` | float | 0.0 | Smoothing applied to the flow field update at each step. |
| `tvf_total_sigma` | float | 0.2 | Smoothing applied to the total velocity field. |
| `tvf_cfl_momentum` | float | 0.9 | Momentum used for optimization acceleration. |
| `tvf_n_time_steps` | int | 3 | Number of integration time steps along the velocity field. |
| `tvf_regularizer` | str | "gaussian" | Regularization method. "gaussian", "sobolev", "dsti", "dsti1". |
| `tvf_fast_smooth` | bool | false | If true, use approximate fast smoothing. |
| `tvf_use_analytical_gradients` | bool | false | If true, use analytical gradients instead of autograd. |
| `tvf_antisymmetric` | bool | true | Enforce antisymmetric property of the velocity field. |
| `tvf_constant_speed` | bool | true | Enforce constant speed property. |
| `tvf_constant_speed_relaxation` | float | 0.10 | Relaxation parameter for the constant speed constraint. |
| `reg_iterations` | list[int] | [80, 80, 20] | Multi-resolution iteration schedule [coarse, medium, fine]. |

## Example

```json
{
    "_metadata": {
        "version": "1.0",
        "description": "Peak benchmark configuration for syntx v3.0.18+",
        "created": "2026-08-14",
        "notes": "dsti1 regularizer requires the MPS F.conv3d fix (v3.0.18+)"
    },
    "syn_config": {
        "grad_step": 0.25,
        "fluid_sigma": 3.0,
        "elastic_sigma": 0.0,
        "lncc_radius": 2,
        "inverse_steps": 10,
        "syn_regularizer": "dsti1",
        "syn_fast_smooth": false,
        "syn_use_analytical_gradients": false,
        "syn_inverse_method": "anderson",
        "reg_iterations": [100, 100, 20]
    },
    "tvf_config": {
        "tvf_grad_step": 0.211,
        "tvf_flow_sigma": 0.0,
        "tvf_total_sigma": 0.2,
        "tvf_cfl_momentum": 0.9,
        "tvf_n_time_steps": 3,
        "tvf_regularizer": "gaussian",
        "tvf_fast_smooth": false,
        "tvf_use_analytical_gradients": false,
        "tvf_antisymmetric": true,
        "tvf_constant_speed": true,
        "tvf_constant_speed_relaxation": 0.1,
        "reg_iterations": [80, 80, 20]
    }
}
```

## Provenance Notes
- SyN peak parameters from GEMINI.md Section 3
- TVF peak parameters from GEMINI.md Section 3
