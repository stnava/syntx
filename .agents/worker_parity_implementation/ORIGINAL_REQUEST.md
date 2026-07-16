## 2026-07-15T15:36:49Z

You are a teamwork_preview_worker.
Your task is to implement the native physical space optimization and correct affine coordinate composition in PyTorch and JAX to achieve 3D registration parity.
Specifically:
1. Read the explorer's handoff report at /Users/stnava/code/syntx/.agents/explorer_investigation/handoff.md.
2. Read the project scope and constraints in /Users/stnava/code/syntx/.agents/orchestrator_3d_parity_1/PROJECT.md and GEMINI.md.
3. Create `scratch/test_internal_dice.py` based on the explorer's draft/specification to evaluate physical coordinate-mapping accuracy. Run it to see the initial status (or verify your new code).
4. Implement native physical space optimization and correct affine coordinate composition in `src/syntx/syn.py` and `src/syntx/syn_jax.py`. Displacement fields must operate natively in physical mm coordinates, and composition mapping to moving space must be strictly composed as y = A(phi_2_inv(phi_1(x))).
5. Ensure you strictly adhere to the single interpolation policy: do NOT pre-warp images or labels prior to optimization. They must be composed and applied in a single step.
6. Verify that your implementation passes `scratch/test_internal_dice.py` with a DICE score >= 0.999.
7. Run `pytest` to verify all unit tests in the repository pass successfully.
8. Verify that existing 2D parity is preserved and 3D registration runs successfully.
9. Write a detailed report of your changes, build/test results, and verification findings in handoff.md in your working directory.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your working directory is: /Users/stnava/code/syntx/.agents/worker_parity_implementation
Please message the parent when you are done.
