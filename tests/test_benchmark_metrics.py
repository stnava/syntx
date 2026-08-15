import pytest
import numpy as np
import ants
import tempfile
import os

from syntx.benchmark.metrics import compute_pair_metrics

def test_compute_pair_metrics_affine_only():
    """Test compute_pair_metrics when only an affine transform is provided."""
    # Create simple 2D synthetic images
    fi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    fl = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    ml = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tx_path = tmp.name
        
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=2)
    ants.write_transform(tx, tx_path)
    
    res = compute_pair_metrics(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        fwdtransforms=[tx_path],
        invtransforms=[tx_path],
        whichtoinvert_inv=[True],
        runtime_seconds=1.5
    )
    
    # Affine only -> energies and foldings should be default
    assert res['runtime_seconds'] == 1.5
    assert res['folding_pct'] == 0.0
    assert res['min_jacobian'] == 1.0
    assert res['harmonic_energy'] == 0.0
    assert res['bending_energy'] == 0.0
    
    assert not np.isnan(res['mattes_mi'])
    assert not np.isnan(res['lncc'])
    assert not np.isnan(res['dice_fixed'])
    
    os.remove(tx_path)

def test_compute_pair_metrics_nonlinear():
    """Test compute_pair_metrics when a nonlinear warp is provided."""
    fi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    fl = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    ml = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        warp_path = tmp.name
        
    # Create dummy 2D warp field
    warp_arr = np.zeros((20, 20, 2), dtype=np.float32)
    warp_arr[10, 10, 0] = 0.5  # add slight deformation to trigger energy
    warp_img = ants.from_numpy(warp_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0), has_components=True)
    ants.image_write(warp_img, warp_path)
    
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tx_path = tmp.name
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=2)
    ants.write_transform(tx, tx_path)
    
    res = compute_pair_metrics(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        fwdtransforms=[warp_path, tx_path],
        invtransforms=[tx_path, warp_path]
    )
    
    # Should have computed energy and jacobian
    assert not np.isnan(res['folding_pct'])
    assert not np.isnan(res['min_jacobian'])
    assert not np.isnan(res['harmonic_energy'])
    assert not np.isnan(res['bending_energy'])
    
    assert res['harmonic_energy'] > 0.0
    
    os.remove(warp_path)
    os.remove(tx_path)

def test_compute_pair_metrics_exceptions():
    """Test compute_pair_metrics graceful exception handling."""
    fi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    fl = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    ml = ants.from_numpy(np.ones((20, 20), dtype=np.uint32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    
    # 1. Provide an invalid warp path
    res_bad_warp = compute_pair_metrics(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        fwdtransforms=['nonexistent_warp.nii.gz'],
        invtransforms=['nonexistent_warp.nii.gz']
    )
    assert np.isnan(res_bad_warp['harmonic_energy'])
    assert np.isnan(res_bad_warp['min_jacobian'])
    assert np.isnan(res_bad_warp['mattes_mi']) # apply_transforms will fail

    # 2. Provide invalid labels (e.g. None)
    res_bad_labels = compute_pair_metrics(
        fixed=fi,
        moving=mi,
        fixed_label=None, # will cause exception in Dice computation
        moving_label=None,
        fwdtransforms=[],
        invtransforms=[]
    )
    assert np.isnan(res_bad_labels['dice_fixed'])
