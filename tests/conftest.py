import functools
import numpy as np
import pytest
import ants

import syntx.motion_simulation as _ms

_orig_make_head_phantom = _ms.make_head_phantom
_HEAD_PHANTOM_CACHE = {}


@functools.wraps(_orig_make_head_phantom)
def _cached_make_head_phantom(shape=(36, 40, 32), spacing=(2.5, 2.5, 2.5), seed: int = 0, direction=None):
    dir_key = None if direction is None else tuple(np.asarray(direction, dtype=np.float64).ravel().tolist())
    key = (tuple(int(x) for x in shape), tuple(float(s) for s in spacing), int(seed), dir_key)
    ph = _HEAD_PHANTOM_CACHE.get(key)
    if ph is None:
        ph = _orig_make_head_phantom(shape=shape, spacing=spacing, seed=seed, direction=direction)
        _HEAD_PHANTOM_CACHE[key] = ph
    return _ms.HeadPhantom(
        reference=ph.reference.clone(),
        gm=ph.gm.copy(),
        wm=ph.wm.copy(),
        csf=ph.csf.copy(),
        texture=ph.texture.copy(),
        fibre=ph.fibre.copy(),
        p0=ph.p0.copy(),
    )


_ms.make_head_phantom = _cached_make_head_phantom

_PROCEDURAL_BRAIN_CACHE = {}


def get_cached_procedural_brain_array(shape=(96, 104, 88), seed: int = 1234, sigma: float = 1.0) -> np.ndarray:
    """Cached procedural 3D brain array for synthetic affine and landmark tests."""
    key = (tuple(shape), int(seed), float(sigma))
    arr = _PROCEDURAL_BRAIN_CACHE.get(key)
    if arr is None:
        import siq
        from scipy.ndimage import gaussian_filter
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            raw = siq.simulate_brain_procedural(shape).numpy().astype(np.float32)
        finally:
            np.random.set_state(state)
        arr = gaussian_filter(raw, sigma).astype(np.float32)
        _PROCEDURAL_BRAIN_CACHE[key] = arr
    return arr.copy()

@pytest.fixture
def synthetic_ants_volume():
    """Creates a 32x32x32 antsImage with two distinct ROI blobs."""
    vol = np.zeros((32, 32, 32), dtype=float)
    vol[8:16, 8:16, 8:16] = 10
    vol[20:28, 20:28, 20:28] = 20
    # Add origin/spacing to test physical space
    return ants.from_numpy(vol, origin=(10, 20, 30), spacing=(1, 1, 1))

@pytest.fixture
def synthetic_ants_intensity():
    """Creates a 32x32x32 antsImage intensity gradient."""
    x, y, z = np.ogrid[:32, :32, :32]
    data = (x + y + z).astype(float)
    return ants.from_numpy(data, origin=(10, 20, 30), spacing=(1, 1, 1))

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
