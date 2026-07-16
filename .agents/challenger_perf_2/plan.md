# Verification Plan

This plan outlines the steps taken by the Empirical Challenger to verify the optimizer behavior and parity.

## Steps
1. **Run Optimizer Unit Tests**: Execute `pytest tests/test_optimizers.py` to confirm basic optimizer execution under both PyTorch and JAX backends.
   - *Verification*: Confirm 8/8 tests pass.
2. **Run Full Test Suite with Coverage**: Execute `pytest --cov=src` to check complete test correctness and overall code coverage.
   - *Verification*: Confirm all tests pass and coverage metrics are generated.
3. **Execute Optimizer Sweep Script**: Run `python examples/run_optimizer_sweeps.py` to perform the 2D and 3D parameter sweeps.
   - *Verification*:
     - Verify completion without errors.
     - Inspect the generated CSV at `outputs_comparison/optimizer_sweep_results.csv`.
     - Confirm that ANTs registration parity matches within 1%.
4. **Identify Instabilities and Failure Modes**:
   - Check L-BFGS, SGD, Adam, and CFL for numerical stability, NaN values, grid folding, and optimization progress.
5. **Inspect Compliance with GEMINI.md**:
   - Ensure the generated HTML report includes the required diagnostic visualizations (edge overlap, region overlap, deformed grids, Jacobian maps, and side-by-side images).
6. **Compile Verification Report**:
   - Save findings to `.agents/challenger_perf_2/handoff.md` and report back to the parent agent.
