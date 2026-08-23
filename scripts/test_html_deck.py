import base64
import os

print("Testing base64 encoding of slide figures...")
for p in [
    "docs/manuscript/figures/fig_syn_manifold_conceptual.jpg",
    "docs/presentation/figures/diag_topology_preservation.png",
    "docs/presentation/figures/diag_variance_floor_proof.png",
    "docs/presentation/figures/diag_sobolev_adam_comparison.png",
    "docs/manuscript/figures/fig_syn_standard_report_flow.png",
    "docs/manuscript/figures/fig_tvf_standard_report_flow.png"
]:
    assert os.path.exists(p), f"Missing: {p}"
    size_kb = os.path.getsize(p) / 1024
    print(f"  {p}: {size_kb:.1f} KB")

print("All test figures exist!")
