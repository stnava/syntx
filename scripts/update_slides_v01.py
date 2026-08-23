import re

V01_IMAGES = {
    1: "docs/presentation/figures/fig_syn_manifold_conceptual_v01.jpg",
    2: "docs/presentation/figures/diag_spatial_inverse_problem_v01.png",
    3: "docs/presentation/figures/diag_topology_preservation_v01.png",
    4: "docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg",
    5: "docs/presentation/figures/diag_lncc_function_space_v01.png",
    6: "docs/presentation/figures/diag_variance_floor_proof_v01.png",
    7: "docs/presentation/figures/diag_so3_lie_algebra_v01.png",
    8: "docs/presentation/figures/diag_18cone_multistart_v01.png",
    9: "docs/presentation/figures/diag_single_interpolation_v01.png",
    10: "docs/presentation/figures/diag_syn_frechet_midpoint_v01.png",
    11: "docs/presentation/figures/diag_eulerian_vs_lagrangian_v01.png",
    12: "docs/presentation/figures/diag_anderson_acceleration_v01.png",
    13: "docs/presentation/figures/diag_antithetic_bootstrapping_v01.png",
    14: "docs/presentation/figures/fig_lddmm_kinetic_action_v01.jpg",
    15: "docs/presentation/figures/diag_tvf_spline_trajectory_v01.png",
    16: "docs/presentation/figures/diag_sobolev_adam_comparison_v01.png",
    17: "docs/presentation/figures/diag_sobolev_adam_comparison_v01.png",
    18: "docs/presentation/figures/diag_dsti_boundary_operators_v01.png",
    19: "docs/presentation/figures/diag_cohort90_metrology_v01.png",
    20: "docs/presentation/figures/fig_diffeomorphic_ai_future_v01.jpg",
}

with open("scripts/generate_presentation.py", "r") as f:
    text = f.read()

# Replace each image entry in SLIDES_DATA
for num, img_path in V01_IMAGES.items():
    pattern = rf'("num":\s*{num},.*?"image":\s*")[^"]+(")'
    text = re.sub(pattern, rf'\g<1>{img_path}\g<2>', text, flags=re.DOTALL)

with open("scripts/generate_presentation.py", "w") as f:
    f.write(text)

print("Updated scripts/generate_presentation.py with _v01 figures!")
