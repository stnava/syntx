# Progress Tracker

Last visited: 2026-07-14T16:58:55-04:00

## Tasks
- [x] Inspect the `antspyt1w.resnet_grader` API (Numeric grader scores retrieved).
- [x] Verify existing code structure and imports in `syntx` (All 85 pytest unit tests pass).
- [x] Draft/implement `examples/run_benchmarks.py`.
- [x] Implement R1: 2D registrations between phantoms (Completed and saved to `outputs_comparison/r1_2d_results.csv`).
- [/] Implement R2: 3D registration benchmark on 8 T1w scans, grading them, dynamically generating label maps, performing deformable registration with VGG19 (lncc_3d, layers=[4]), DINOv2, ResNet-10, Swin UNETR, and ants baseline. (Running; scans 1, 2, 3, and 4 completed, scan 5 is currently running).
- [/] Implement R3: Generate the detailed HTML report `docs/benchmarks.html` and output visual comparison maps in `outputs_comparison/`.
- [ ] Run the benchmark script and verify successful completion of all steps.
- [ ] Write handoff.md and send final message.
