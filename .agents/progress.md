# Progress Report

Last visited: 2026-07-15T16:20:10Z

## Current Tasks
- **Fixing 3D Registration Parity and Folding (Milestone 5 Parity Sweep)**
  - Reverted inconsistent/duplicate coordinate reversals from `src/syntx/syn.py` and `src/syntx/syn_jax.py`.
  - Re-aligned the spatial Jacobian component ordering in both PyTorch (`_spatial_jacobian_nd`) and JAX (`_spatial_jacobian_nd_jax`) backends to match internal tensor indexing, resolving the shearing/folding issue.
  - Standardized coordinate reversal handling: low-level functions accept raw inputs directly, and caller sites pass the reversed metadata.
  - Fixed learning rate parameter passing to model.fit in registration.
  - Corrected physical spacing conversion for grid mapping.
  - Fixed NIfTI displacement field vector component order to match ANTs Python wrapper expectations.
  - Successfully verified all tests: 122 tests passed.

Status: Completed. Last visited: 2026-07-15T17:13:00Z.
