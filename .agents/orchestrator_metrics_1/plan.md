# Scope: Image Comparison Metrics Suite

## Architecture
- Module/package: `syntx.image_compare`
- API Design: All metrics accessible programmatically via metric names.
- Dimensions: Support both 2D and 3D images (3D can extend 2D).
- Generative Space: 2D generative pipeline producing a cross-product of intensity and shape changes.
- Visualization: Edge/region overlap, deformed grids, Jacobian maps, side-by-side images.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Exploration & Design | Examine existing codebase, PyTorch/JAX setup, ANTs bindings, and design 64+ metrics. | None | DONE |
| M2 | Metric Suite Implementation | Implement at least 64 image comparison metrics (supporting 2D & 3D). | M1 | DONE |
| M3 | Generative Space Generation | Develop 2D generative cross-product space with intensity & shape changes (Grenander's metric deformation as GT). | M1 | DONE |
| M4 | Testing & Verification | Implement unit tests for metric API, zero-distance/divergence properties, and generative space constraints. | M2, M3 | DONE |
| M5 | Evaluation & Report | Run subset of metrics against generative space and generate HTML report meeting GEMINI.md guidelines. | M4 | DONE |
| M6 | Documentation & Cleanup | Write user documentation and runnable example script. | M5 | DONE |

## Interface Contracts
### `syntx.image_compare`
- Input images: 2D or 3D tensors/arrays.
- Function: `syntx.image_compare.compare(img1, img2, metricname: str, **kwargs) -> float`
- Metrics: at least 64 unique metric names.
- Identical images result in 0 or minimum possible value.
- Diverging images result in strictly increasing values.
