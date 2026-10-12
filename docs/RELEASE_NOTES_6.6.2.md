# syntx 6.6.2

Fix: standardize `syntx.scattered.syn_scattered` on the canonical `cfl` optimizer by default, resolving premature optimization stall on large displacement fields.

Details:
- Fix CFL step scaling in `SyNScattered.fit` by removing the erroneous `* optimizer_lr` multiplier that throttled step sizes down to 0.0125 voxels/iteration.
- Set `optimizer_type='cfl'` as default in `ScatteredRegistrationConfig`, with automatic step resolution (`1.0` for CFL, `0.5` for RegAdam, `0.05` for Rprop).
- Upgrade `reg_adam` to Riemannian Adam: accumulate moments on raw gradients, spatially smooth the Adam quotient through the chosen spectral Green operator, and scale by adaptive CFL bounds matching `syntx.syn`.
- Retain full backward compatibility for explicit `optimizer_type='rprop'` calls.
