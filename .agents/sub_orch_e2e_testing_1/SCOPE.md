# Scope: E2E Testing Track

## Architecture
The E2E Testing Track designs and implements a requirement-driven, opaque-box test suite for verifying (1) DLPack-based PyTorch Feature-Space Loss integration in JAX SyN loops, and (2) MONAI Swin UNETR 3D feature extractor.
The test harness runs the tests, verifies outputs and constraints, and measures coverage.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Design Test Cases | Design 27 test cases across Tiers 1-4 covering all features, boundary conditions, combinations, and application workloads. | none | PLANNED |
| 2 | Write TEST_INFRA.md | Document the test architecture, features inventory, test cases, and thresholds in PROJECT_ROOT/TEST_INFRA.md. | M1 | PLANNED |
| 3 | Implement Test Harness & Cases | Write test implementation files under `tests/` and a robust test runner. | M2 | PLANNED |
| 4 | Verify Coverage & Execution | Execute the test suite, ensure all tests run, and check that test infra coverage is >= 90%. | M3 | PLANNED |
| 5 | Publish TEST_READY.md | Write PROJECT_ROOT/TEST_READY.md confirming test suite completeness, with coverage report. | M4 | PLANNED |

## Interface Contracts
### E2E Test Runner
- Execution Command: `pytest tests/test_e2e_metrics.py` (or similar command to run E2E test cases).
- Pass criteria: Zero errors, zero failures, all 27 tests executed.
- Code coverage check: Coverage of the test runner/infrastructure scripts >= 90%.
