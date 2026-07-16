## 2026-07-15T13:13:32-04:00

You are a teamwork_preview_reviewer.
Your task is to review the physical space optimization and affine coordinate composition implementation in src/syntx/syn.py and src/syntx/syn_jax.py.
Please:
1. Examine code correctness, completeness, robustness, and interface conformance.
2. Verify that the single interpolation policy is strictly respected (no pre-warping of input images/labels).
3. Check the correctness of the composed coordinate mapping logic (y = A(phi_2_inv(phi_1(x)))).
4. Run/examine the build and tests to ensure no regressions are introduced.
5. Write your review verdict and findings in handoff.md in your working directory.

Your working directory is: /Users/stnava/code/syntx/.agents/reviewer_parity_check
Please message the parent when you are done.
