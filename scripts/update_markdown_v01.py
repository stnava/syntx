import re

V01_MAP = {
    r"docs/manuscript/figures/fig_syn_manifold_conceptual\.jpg": "docs/presentation/figures/fig_syn_manifold_conceptual_v01.jpg",
    r"docs/presentation/figures/diag_spatial_inverse_problem\.png": "docs/presentation/figures/diag_spatial_inverse_problem_v01.png",
    r"docs/presentation/figures/diag_topology_preservation\.png": "docs/presentation/figures/diag_topology_preservation_v01.png",
    r"docs/manuscript/figures/fig_tvf_manifold_conceptual\.jpg": "docs/presentation/figures/fig_tvf_manifold_conceptual_v01.jpg",
    r"docs/manuscript/figures/fig1_architecture_flow\.jpg": "docs/presentation/figures/diag_lncc_function_space_v01.png",
    r"docs/presentation/figures/diag_variance_floor_proof\.png": "docs/presentation/figures/diag_variance_floor_proof_v01.png",
    r"docs/manuscript/figures/fig_visual_story1_robust_affine\.png": "docs/presentation/figures/diag_so3_lie_algebra_v01.png",
    r"docs/manuscript/figures/fig_algo1_robust_affine\.png": "docs/presentation/figures/diag_18cone_multistart_v01.png",
    r"docs/manuscript/figures/fig_syn_standard_report_flow\.png": "docs/presentation/figures/diag_single_interpolation_v01.png",
    r"docs/manuscript/figures/fig_visual_story2_syn_geodesic\.png": "docs/presentation/figures/diag_syn_frechet_midpoint_v01.png",
    r"docs/manuscript/figures/fig_algo2_syn_architecture\.png": "docs/presentation/figures/diag_eulerian_vs_lagrangian_v01.png",
    r"docs/manuscript/figures/fig_algo3_antithetic_bootstrapping\.png": "docs/presentation/figures/diag_antithetic_bootstrapping_v01.png",
    r"docs/manuscript/figures/fig_visual_story4_tvf_trajectory_flow\.png": "docs/presentation/figures/diag_tvf_spline_trajectory_v01.png",
    r"docs/presentation/figures/diag_sobolev_adam_comparison\.png": "docs/presentation/figures/diag_sobolev_adam_comparison_v01.png",
    r"docs/manuscript/figures/fig_algo5_sobolev_adam_cfl\.png": "docs/presentation/figures/diag_sobolev_adam_comparison_v01.png",
    r"docs/manuscript/figures/fig_visual_story6_dsti_multiscale_suite\.png": "docs/presentation/figures/diag_dsti_boundary_operators_v01.png",
    r"docs/manuscript/figures/fig4_cohort90_statistical_distributions\.png": "docs/presentation/figures/diag_cohort90_metrology_v01.png",
    r"docs/manuscript/figures/fig_tvf_standard_report_flow\.png": "docs/presentation/figures/fig_diffeomorphic_ai_future_v01.jpg",
}

for md_path in [
    "docs/presentation/syntx_phd_masterclass_presentation.md",
    "/Users/stnava/.gemini/antigravity-cli/brain/c4defdc2-4f56-4c75-a05c-afbe553de3de/syntx_phd_masterclass_presentation.md"
]:
    if os.path.exists(md_path):
        with open(md_path, "r") as f:
            content = f.read()
        for old_p, new_p in V01_MAP.items():
            content = re.sub(old_p, new_p, content)
        with open(md_path, "w") as f:
            f.write(content)
        print(f"Updated: {md_path}")
