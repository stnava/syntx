## 2026-07-15T15:35:15Z

You are a teamwork_preview_explorer.
Your task is to investigate and plan the implementation of 3D registration parity between `syntx` and `ants.registration`.
Specifically:
1. Read and analyze the PyTorch and JAX SyN registration optimization loops in `src/syntx/syn.py` and `src/syntx/syn_jax.py`.
2. Analyze the current coordinate-mapping logic (grid space to physical space and vice versa).
3. Investigate how displacement fields (phi_1, phi_2) are updated, scaled, and regularized (fluid_sigma, elastic_sigma).
4. Propose how to rewrite these loops so displacement fields natively operate in physical mm coordinates in the fixed image domain (like ITK SyN).
5. Analyze the affine coordinate composition. How to strictly implement the mapping as y = A(phi_2_inv(phi_1(x))) while respecting the single interpolation policy (no pre-warping of input images/labels)?
6. Recommend a draft/specification for `scratch/test_internal_dice.py` to evaluate coordinate-mapping accuracy internally.
7. Write your findings to handoff.md in your working directory.

Your working directory is: /Users/stnava/code/syntx/.agents/explorer_investigation
Please make sure to write a detailed report of your findings in your working directory and message the parent when done.
