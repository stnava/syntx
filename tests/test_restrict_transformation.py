"""Mechanism-level verification of syn()'s restrict_transformation option.

Deliberately does not test by comparing to an external reference implementation's
aggregate output -- it constructs a case with a *known* physical-space answer (a
pure shift along one physical axis) and asserts the per-axis structure of the
resulting deformation field directly, the same way ants.registration's own
restrict_transformation is documented to behave. This is what actually proves the
axis convention (physical XYZ order, weight 1=free/0=restricted) is wired correctly,
as opposed to a test that would pass even with the axes transposed or the weights
inverted.
"""
import numpy as np
import pytest

ants = pytest.importorskip("ants")
torch = pytest.importorskip("torch")

import syntx


def _asymmetric_blob(shape=(28, 28, 28)):
    """A volume with no axis-swap symmetry, so a transposed-axis bug cannot hide."""
    vol = np.zeros(shape, dtype=np.float32)
    vol[6:10, 8:20, 10:14] = 1.0
    vol[14:22, 6:9, 16:24] = 1.0
    return vol


def _shifted_pair(shift_axis: int, shift_voxels: int = 3, spacing=(1.0, 1.5, 2.0)):
    """fixed/moving pair differing by a pure integer-voxel shift along one array axis.

    With identity direction and ants.from_numpy, array axis i corresponds to physical
    axis i (X=0, Y=1, Z=2), matching `spacing`'s order -- confirmed by reading syntx's
    own `curr_spacing_fixed_xyz` usage in syn.py, which divides displacement-field
    components by `spacing` directly.
    """
    base = _asymmetric_blob()
    fixed = ants.from_numpy(base, spacing=spacing)
    moving_arr = np.roll(base, shift=shift_voxels, axis=shift_axis)
    moving = ants.from_numpy(moving_arr, spacing=spacing)
    return fixed, moving


def _field_component_magnitudes(reg, fixed):
    """Mean |displacement| per physical axis, read directly from the fwd warp field."""
    warp_path = [p for p in reg["fwdtransforms"] if str(p).endswith((".nii.gz", ".nii"))]
    assert warp_path, f"expected a displacement field among fwdtransforms, got {reg['fwdtransforms']}"
    field = ants.image_read(warp_path[0]).numpy()  # (X, Y, Z, 1, 3) or (X, Y, Z, 3)
    field = field.reshape(field.shape[0], field.shape[1], field.shape[2], -1)[..., :3]
    return np.abs(field).reshape(-1, 3).mean(axis=0)  # (3,) mean |disp| for X, Y, Z


REG_KWARGS = dict(
    type_of_transform="SyNOnly",
    syn_metric="cc2",
    syn_sampling=2,
    reg_iterations=[30, 20],
    verbose=False,
    seed=42,
)


def test_restriction_confines_deformation_to_the_allowed_physical_axis():
    """Shift along Y (axis=1); allow only Y (0,1,0): should correct well, and the X/Z
    field components should stay near zero while Y captures the correction."""
    fixed, moving = _shifted_pair(shift_axis=1)
    reg = syntx.syn(fixed, moving, restrict_transformation=(0.0, 1.0, 0.0), **REG_KWARGS)
    mag = _field_component_magnitudes(reg, fixed)
    # Calibrated empirically: this synthetic case converges to ~0.1-0.2 mean |disp| at
    # these (deliberately fast, low-iteration) settings. The precise magnitude is not
    # the point -- the point is that it is clearly nonzero on the allowed axis and
    # *exactly* zero (not merely small) on the restricted axes, which is what the mask
    # multiply guarantees structurally.
    assert mag[1] > 0.05, f"Y (allowed axis) should show real deformation, got {mag}"
    assert mag[0] == 0.0, f"X (restricted) must be exactly 0, got {mag}"
    assert mag[2] == 0.0, f"Z (restricted) must be exactly 0, got {mag}"


def test_restriction_forbidding_the_needed_axis_fails_to_correct():
    """Same shift along Y, but restrict_transformation=(1,0,1) forbids Y: the
    registration should NOT be able to produce the needed correction there."""
    fixed, moving = _shifted_pair(shift_axis=1)
    reg = syntx.syn(fixed, moving, restrict_transformation=(1.0, 0.0, 1.0), **REG_KWARGS)
    mag = _field_component_magnitudes(reg, fixed)
    assert mag[1] < 1e-3, f"Y (restricted) should stay ~0 even though it's needed, got {mag}"


def test_no_restriction_is_unaffected_backward_compatible():
    """restrict_transformation=None (default) must reproduce prior (unrestricted)
    behaviour: real deformation along the shifted axis, not suppressed."""
    fixed, moving = _shifted_pair(shift_axis=1)
    reg = syntx.syn(fixed, moving, restrict_transformation=None, **REG_KWARGS)
    mag = _field_component_magnitudes(reg, fixed)
    assert mag[1] > 0.05, f"unrestricted registration should correct the Y shift, got {mag}"
    assert mag[1] > mag[0] and mag[1] > mag[2], (
        f"unrestricted correction should be dominated by the true (Y) shift axis, got {mag}"
    )


def test_invalid_length_raises_at_construction():
    fixed, moving = _shifted_pair(shift_axis=1)
    with pytest.raises(ValueError, match="length"):
        syntx.syn(fixed, moving, restrict_transformation=(0.0, 1.0), **REG_KWARGS)


def test_out_of_range_weight_raises():
    fixed, moving = _shifted_pair(shift_axis=1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        syntx.syn(fixed, moving, restrict_transformation=(0.0, 1.5, 0.0), **REG_KWARGS)


def test_jax_backend_raises_rather_than_silently_ignoring():
    fixed, moving = _shifted_pair(shift_axis=1)
    with pytest.raises(NotImplementedError):
        syntx.syn(fixed, moving, restrict_transformation=(0.0, 1.0, 0.0), backend="jax", **REG_KWARGS)


def test_restriction_applies_under_the_adam_optimizer_branch_too():
    """The Adam/RegAdam optimizer path (u_reg_l/u_reg_r) is a separate injection point
    from the default 'cfl' path and is not exercised by the tests above; verify it
    independently so a regression there is not silently uncovered."""
    fixed, moving = _shifted_pair(shift_axis=1)
    reg = syntx.syn(
        fixed, moving, restrict_transformation=(0.0, 1.0, 0.0),
        optimizer_type="reg_adam", regularizer="sobolev",
        **REG_KWARGS,
    )
    mag = _field_component_magnitudes(reg, fixed)
    assert mag[0] == 0.0 and mag[2] == 0.0, f"X/Z (restricted) must be exactly 0 under reg_adam too, got {mag}"
