"""Parameter scales and the metric-manifold probe (syntx.core.parameter_scales)."""
import numpy as np
import pytest
import torch

from syntx.core.parameter_scales import ParamAxis, matrix_element_scale, probe_manifold, rigid_axes, rotation_axis_scale


def _points(n=4000, spacing_mm=1.0, extent_vox=40.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=g) - 0.5) * 2.0 * extent_vox * spacing_mm


class TestScales:
    def test_rotation_scale_is_the_rms_distance_from_the_axis_and_matches_a_real_rotation(self):
        pts = _points()
        c = torch.zeros(3)
        for axis in range(3):
            s = rotation_axis_scale(pts, c, axis)
            ang = 1e-3
            w = torch.zeros(3); w[axis] = ang
            moved = pts + torch.linalg.cross(w.expand_as(pts), pts)          # small rotation: x + w x x
            real = float((moved - pts).pow(2).sum(-1).mean().sqrt()) / ang
            assert abs(s - real) / real < 1e-3

    def test_scales_grow_with_the_object_in_mm_but_not_in_voxels(self):
        for sp in (0.1, 2.0, 20.0):                                    # mouse ... human ... a coarse 20 mm grid
            pts = _points(spacing_mm=sp)
            assert rotation_axis_scale(pts, torch.zeros(3), 2) / sp == pytest.approx(
                rotation_axis_scale(_points(spacing_mm=1.0), torch.zeros(3), 2), rel=1e-3)

    def test_matrix_element_scale_is_the_rms_coordinate(self):
        pts = _points()
        assert matrix_element_scale(pts, torch.zeros(3), 1) == pytest.approx(float(pts[:, 1].pow(2).mean().sqrt()), rel=1e-6)


def _quadratic(axes, optimum, k):
    """Loss in units of voxels^2 of displacement: sum_a 0.5 k_a ((theta_a - opt_a) * scale_a / voxel)^2, per frame."""
    def fn(voxel_mm):
        def loss():
            total = 0
            for a, ax in enumerate(axes):
                th = ax.tensor[ax.index]
                total = total + 0.5 * k[a] * ((th - optimum[a]) * ax.scale_mm / voxel_mm) ** 2
            return total
        return loss
    return fn


class TestProbe:
    @pytest.mark.parametrize("voxel_mm,radius_mm", [(2.0, 75.0), (0.2, 7.5), (20.0, 750.0), (1.5, 30.0)])
    def test_recovers_curvature_and_remaining_distance_in_voxels_at_any_spacing(self, voxel_mm, radius_mm):
        B = 3
        t, w = torch.zeros(B, 3), torch.zeros(B, 3)
        axes = [ParamAxis(t, (slice(None), j), 1.0) for j in range(3)] + [ParamAxis(w, (slice(None), j), radius_mm) for j in range(3)]
        k = [4.0, 2.0, 1.0, 8.0, 0.5, 3.0]                                   # per voxel^2, the "truth"
        opt = [torch.tensor([0.7, -0.3, 1.1]) * voxel_mm / ax.scale_mm for ax in axes]   # optimum 0.7/-0.3/1.1 voxels away on each axis, per frame b
        opt = [o[: B] if o.numel() >= B else o for o in opt]
        loss = _quadratic(axes, opt, k)(voxel_mm)
        pr = probe_manifold(loss, axes, voxel_mm, probe_vox=0.5)
        assert np.allclose(pr.curv.numpy(), np.array(k)[None, :].repeat(B, 0), rtol=1e-3, atol=1e-3)
        # frame b sits |opt_b| = (0.7, 0.3, 1.1) voxels from the optimum on every axis
        expected = np.stack([np.abs(np.array([0.7, -0.3, 1.1]))] * 6, axis=1)
        assert np.allclose(pr.newton_vox.numpy(), expected, rtol=1e-3, atol=1e-3)
        assert np.allclose(pr.remaining_vox.numpy(), [0.7, 0.3, 1.1], rtol=1e-3)

    def test_parameters_are_restored_exactly(self):
        t = torch.tensor([[0.1, 0.2, 0.3]]); w = torch.tensor([[0.01, 0.02, 0.03]])
        t0, w0 = t.clone(), w.clone()
        axes = [ParamAxis(t, (slice(None), j), 1.0) for j in range(3)] + [ParamAxis(w, (slice(None), j), 50.0) for j in range(3)]
        probe_manifold(lambda: (t ** 2).sum(-1) + (w ** 2).sum(-1), axes, voxel_mm=2.0)
        assert torch.equal(t, t0) and torch.equal(w, w0)

    def test_flat_or_concave_axes_are_not_reported_as_converged(self):
        t = torch.zeros(1, 3)
        axes = [ParamAxis(t, (slice(None), j), 1.0) for j in range(3)]
        flat = probe_manifold(lambda: torch.zeros(1), axes, voxel_mm=1.0)
        assert torch.isinf(flat.remaining_vox).all()
        concave = probe_manifold(lambda: -(t ** 2).sum(-1), axes, voxel_mm=1.0)
        assert torch.isinf(concave.remaining_vox).all()

    def test_single_model_scalar_loss(self):
        p = torch.tensor([0.0, 0.0, 0.0])
        axes = [ParamAxis(p, (j,), 1.0) for j in range(3)]
        pr = probe_manifold(lambda: ((p - torch.tensor([1.0, 0.0, 0.0])) ** 2).sum() * 0.5, axes, voxel_mm=1.0)
        assert pr.remaining_vox.shape == (1,) and pr.remaining_vox.item() == pytest.approx(1.0, rel=1e-3)
