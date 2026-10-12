# syntx 6.6.3

Enhancement & Fix: Expose total-field smoothing (`total_sigma`), fix Dirichlet boundary decay, introduce local CFL pseudo-time stepping, and support automated multi-resolution pyramids in `syntx.scattered.syn_scattered`.

Details:
- **Expose Total-Field Smoothing (`total_sigma`)**:
  - Add standard `total_sigma` (with bidirectional alias `elastic_sigma`) to `ScatteredRegistrationConfig` and `SyNScattered.__init__`.
  - Enables Gaussian diffusion of accumulated displacement fields after each step, carrying motion from well-driven regions into weakly-driven neighbors.
- **Fix Dirichlet Boundary Decay**:
  - Eliminate the artificial `* b_mask` zeroing after Gaussian smoothing in `solver.py` (line 1568), which previously acted as a Dirichlet zero-boundary condition diffusing zeros inward and collapsing interior displacement.
  - Adopt natural Neumann boundary conditions (`mode='replicate'`) without perimeter zeroing, strictly matching `syntx.syn`.
- **Local CFL Pseudo-Time Stepping (`cfl_mode='local'`)**:
  - Add `cfl_mode: {'global', 'local'}` to `ScatteredRegistrationConfig`.
  - In `'local'` mode, per-cell voxel step bounds (`level_cfl`) are applied elementwise, ensuring weakly-driven cells far from optimum are not throttled by peak gradients elsewhere in the domain.
- **Automated Multi-Resolution Pyramids**:
  - Support `levels='auto'`, constructing a 3-level schedule (`[4, 2, 1]`) when `min(grid_res) >= 64`.
