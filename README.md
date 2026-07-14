# syntx

`syntx` is a high-performance Python package focusing on symmetric diffeomorphic (`SyN`) and affine image registration methods, built on top of **PyTorch** and **JAX** for GPU/MPS acceleration and auto-differentiation capabilities.

Ported from the registration modules of the `sulceye` package, `syntx` is designed for distribution on PyPI and works seamlessly with standard medical image types, particularly `ANTsImage` from the `antspyx` library.

---

## Key Features
- **Auto-Differentiation Backends:** Choose between `'pytorch'` and `'jax'` for core computations.
- **Symmetric Normalization (SyN):** Fully symmetric greedy optimization matching classic ITK/ANTs SyN implementations.
- **Interoperability:** Seamless conversions between PyTorch/JAX coordinate spaces and ITK physical coordinate matrices.
- **Direct PyPy/PyPI Packaging:** Implemented cleanly with minimum external dependencies.

---

## Installation

To install `syntx` locally from the repository:
```bash
pip install -e .
```

### Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `antspyx`
- `torch`
- `jax`
- `jaxlib`

---

## Usage Example

`syntx` exposes a high-level `syn` and `registration` API that mirrors `ants.registration`:

```python
import ants
import syntx

# Load ANTs images
fixed = ants.image_read( ants.get_data('r16') )
moving = ants.image_read( ants.get_data('r64')  )

# Run registration using PyTorch (default)
result = syntx.syn(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyNTo',
    backend='pytorch',
    reg_iterations=[100, 100, 50],
    affine_iterations=[100, 50, 20],
)

# Access the warped moving output image
warped_moving = result['warpedmovout']

# Access transform files (saved to temporary paths for ANTs compatibility)
forward_transforms = result['fwdtransforms']
inverse_transforms = result['invtransforms']
```

For JAX backend acceleration:
```python
result = syntx.syn(
    fixed=fixed,
    moving=moving,
    type_of_transform='SyNTo',
    backend='jax',
    reg_iterations=[100, 100, 50],
    affine_iterations=[100, 50, 20],
)
```

---

## Running the Examples and Generating Reports

An example comparing classic ANTs, PyTorch, and JAX registration is included under `examples/`. It generates a comparison report summarizing Mutual Information, Jacobian Determinants (topological safety), and Execution Speed.

To run the comparison:
```bash
python examples/generate_ants_2d_comparison_report.py
```

This generates an HTML report under `reports/ants_2d_syn_comparison.html`.

---

## Running Tests

Tests can be executed via `pytest`:
```bash
pytest
```

---

## Makefile Automation

A `Makefile` is included to automate standard development tasks:

*   **Install** (install package in editable mode):
    ```bash
    make install
    ```
*   **Test** (run test suite in Fast mode, skipping slow 3D registrations, and printing a code coverage table):
    ```bash
    make test
    ```
*   **Test All** (run the full test suite including slow 3D registrations, with coverage):
    ```bash
    make test-all
    ```
*   **Clean** (remove build artifacts, cached directories, and temporary files):
    ```bash
    make clean
    ```
*   **Release** (clean, build sdist and wheel packages, and upload to PyPI using twine):
    ```bash
    make release
    ```

It automatically detects and prioritizes the active python virtual environment (`VIRTUAL_ENV`).
