import numpy as np
import pytest
import ants

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
