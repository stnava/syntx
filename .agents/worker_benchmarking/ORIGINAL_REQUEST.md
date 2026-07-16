## 2026-07-14T19:40:11Z
You are the Performance & Benchmarking Worker (teamwork_preview_worker).
Your working directory is /Users/stnava/code/syntx/.agents/worker_benchmarking/.
Your objective is to implement and execute the 2D and 3D registration benchmarks and export the visual dashboard report as specified in the original request.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Key Requirements:
1. Inspect the 'antspyt1w.resnet_grader' API using a Python command to find out how to grade scan quality of T1w images.
2. Implement a comprehensive benchmark runner script 'examples/run_benchmarks.py' that addresses R1, R2, and R3.
3. R1: Run 2D registrations between 'r16', 'r27', and 'r64' phantoms using VGG19, DINOv2, ResNet-10 feature metrics, and standard 'ants.registration(..., "SyN")'. Record and analyze optimization parameters (learning rates, iterations, convergence).
4. R2: Grade the 3D T1w scans '28364-00000000-T1w-00' through '28575-00000000-T1w-07' (8 scans total) in /Users/stnava/.antspyt1w/ using 'antspyt1w.resnet_grader'. Conduct 3D deformable registrations to the fixed template 'T_template0' at NATIVE resolution using VGG19 (vgg_mode='lncc_3d', layers=[4]), DINOv2, ResNet-10, Swin UNETR, and baseline 'ants.registration'. Evaluate cortical label map overlap (DICE score on DKT labels) against the template's labels.
5. Generate DKT label maps dynamically for scans that do not have cached segmentations in 'cache/' using 'antstorch.desikan_killiany_tourville_labeling(img)'.
6. Make sure to adhere to the Single Interpolation Policy (no intermediate resampling/pre-warping) and VGG 3D Mode Requirement (only vgg_mode='lncc_3d' with layers=[4] is acceptable).
7. R3: Compile a detailed benchmark comparison report (HTML dashboard) saved to 'docs/benchmarks.html' containing:
   - Summary tables of DICE scores, runtimes, and folding rates.
   - Correlation plots between scan quality (resnet_grader score) and registration DICE.
   - Grid warp, edge overlap, Jacobian determinant maps, and side-by-side warped vs target visual maps.
   - Save the visual maps for top-performing configurations as images in the output folder 'outputs_comparison/'.
8. Execute the script to generate all benchmark data and the HTML report.
9. Report back to parent with a handoff report and paths of created files.
