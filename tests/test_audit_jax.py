"""Fast checks for the JAX backends' audit fixes (docs/DOCSTRING_AUDIT.md, JAX sections)."""
import numpy as np
import pytest

ants = pytest.importorskip("ants")
pytest.importorskip("jax")


def _pair(n=32, shift=(2, 1)):
    yy, xx = np.mgrid[:n, :n]
    f = np.exp(-((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (2 * (n / 6) ** 2)).astype("float32")
    m = np.roll(f, shift, axis=(0, 1))
    return ants.from_numpy(f), ants.from_numpy(m)


def _linear_tx(paths):
    out = []
    for p in paths:
        if str(p).endswith(".mat"):
            out.append(ants.read_transform(p))
    return out


def test_syngs_jax_backend_runs():
    import syntx
    f, m = _pair()
    r = syntx.syngs(f, m, backend="jax", optimizer="cfl", reg_iterations=[3, 2])
    assert np.isfinite(r["warpedmovout"].numpy()).all()


def test_syn_jax_synonly_does_not_optimise_an_affine():
    """SyNOnly: the JAX fit must not run its own affine stage, so its linear transform is the
    same (centre-of-mass) one PyTorch exports."""
    import syntx
    f, m = _pair()
    lin = {be: _linear_tx(syntx.syn(f, m, backend=be, type_of_transform="SyNOnly",
                                    reg_iterations=[3, 2])["fwdtransforms"])
           for be in ("pytorch", "jax")}
    assert len(lin["jax"]) == len(lin["pytorch"]) == 1
    np.testing.assert_allclose(lin["jax"][0].parameters, lin["pytorch"][0].parameters, atol=1e-4)  # float32 export paths


def test_get_affine_matrix_jax_respects_transform_type():
    import jax.numpy as jnp
    from syntx.syn_jax import get_affine_matrix_jax
    p = {'translation': jnp.array([0.1, -0.2]), 'omega': jnp.array([0.3]), 'scale': jnp.array([1.5]),
         'anisotropic_scale': jnp.array([1.2, 0.8]), 'shear': jnp.array([0.1])}
    c, s_ = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s_], [s_, c]])
    T = np.asarray(get_affine_matrix_jax(p, 2, 'Translation'))
    np.testing.assert_allclose(T[:2, :2], np.eye(2), atol=1e-6)
    np.testing.assert_allclose(T[:2, 2], [0.1, -0.2], atol=1e-6)
    Rr = np.asarray(get_affine_matrix_jax(p, 2, 'Rigid'))[:2, :2]
    assert abs(np.linalg.det(Rr) - 1.0) < 1e-5                      # no scale in a rigid map
    Sm = np.asarray(get_affine_matrix_jax(p, 2, 'Similarity'))[:2, :2]
    np.testing.assert_allclose(Sm, 1.5 * Rr, atol=1e-5)
    with pytest.raises(ValueError, match="transform_type"):
        get_affine_matrix_jax(p, 2, 'SyN')


@pytest.mark.parametrize("weights", [[0.5, 0.5], [[1.0, 0.0], [0.5, 0.5]]])
def test_syn_jax_multichannel_metrics(weights):
    """Two channels / two metrics (flat or per-level weights) run on the JAX backend (crashed in
    the centre-of-mass init and in the metric loop)."""
    import syntx
    f, m = _pair()
    r = syntx.syn([f, f], [m, m], backend="jax", syn_metric=["lncc", "mattes_mi"],
                  syn_metric_weights=weights, reg_iterations=[3, 2])
    w = r["warpedmovout"].numpy()
    assert np.isfinite(w).all()
    assert np.corrcoef(f.numpy().ravel(), w.ravel())[0, 1] > np.corrcoef(f.numpy().ravel(), m.numpy().ravel())[0, 1]


def test_syn_jax_analytic_gradients_need_one_channel():
    import syntx
    f, m = _pair()
    with pytest.raises(ValueError, match="single-channel"):
        syntx.syn([f, f], [m, m], backend="jax", syn_metric=["lncc", "lncc"], reg_iterations=[2],
                  use_analytical_gradients=True)


def test_syn_jax_divergence_retry_reruns_with_halved_step(capsys):
    """A metric whose value is MSE but whose gradient ascends it makes every level diverge: the
    level is rerun up to twice, each time from the checkpoint with half the CFL step."""
    import jax
    import jax.numpy as jnp
    from syntx.syn_jax import SyNTo
    f, m = _pair()
    I = jnp.array(f.numpy().T)[None, None]
    J = jnp.array(m.numpy().T)[None, None]

    def ascend(x, y, mask=None):
        mse = jnp.mean((x - y) ** 2)
        return jax.lax.stop_gradient(2.0 * mse) - mse

    mdl = SyNTo(dim=2, grid_shape=(32, 32))
    mdl.fit(I, J, levels=[1], epochs_per_level=[30], affine_epochs=0, cfl_voxels=0.15,
            similarity_metric=ascend, verbose=1)
    out = capsys.readouterr().out
    assert "Retry 1/2 with CFL=0.0750" in out and "Retry 2/2 with CFL=0.0375" in out
    assert len(mdl.syn_losses) == 90                    # three attempts of 30 epochs


@pytest.mark.parametrize("pseudo", [False, True])
@pytest.mark.parametrize("squared", [False, True])
def test_jax_lncc_matches_torch_semantics(pseudo, squared):
    """CC vs CC^2 must not depend on the gradient flavour (pseudo-gradient CC returned CC^2)."""
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import local_ncc_loss_nd_jax
    from syntx.core.losses import local_ncc_loss_nd
    rng = np.random.default_rng(0)
    a = rng.random((1, 1, 20, 20)).astype('float32')
    b = -(0.5 * a + 0.5 * rng.random((1, 1, 20, 20))).astype('float32')      # anti-correlated
    vj = float(local_ncc_loss_nd_jax(jnp.array(a), jnp.array(b), window_size=5,
                                     use_ants_pseudo_gradient=pseudo, squared=squared))
    vt = float(local_ncc_loss_nd(torch.tensor(a), torch.tensor(b), window_size=5,
                                 use_ants_pseudo_gradient=pseudo, squared=squared))
    assert abs(vj - vt) < 1e-4


def test_jax_inverse_methods_and_options():
    import jax.numpy as jnp
    from syntx.syn_jax import update_inverse_field_nd_jax
    rng = np.random.default_rng(0)
    w = jnp.array(0.3 * rng.standard_normal((1, 16, 16, 2)).astype('float32'))
    geo = dict(spacing=(1.0, 1.0), origin=(0.0, 0.0), direction=((1.0, 0.0), (0.0, 1.0)))
    for meth in ('fixed_point', 'anderson', 'hybrid_lm'):
        r = update_inverse_field_nd_jax(w, None, steps=10, method=meth, **geo)   # no warm start
        assert r.shape == w.shape and bool(jnp.isfinite(r).all())
    with pytest.raises(ValueError, match="unknown inverse method"):
        update_inverse_field_nd_jax(w, None, steps=2, method='fixed-point', **geo)
    with pytest.raises(ValueError, match="relaxation"):
        update_inverse_field_nd_jax(w, None, steps=2, method='anderson', relaxation=0.5, **geo)
    # relaxation scales the fixed-point step: 0 leaves the (rim-masked) initial guess
    r0 = update_inverse_field_nd_jax(w, None, steps=5, method='fixed_point', relaxation=0.0, **geo)
    r1 = update_inverse_field_nd_jax(w, None, steps=5, method='fixed_point', relaxation=1.0, **geo)
    assert float(jnp.abs(r0[:, 2:-2, 2:-2] + w[:, 2:-2, 2:-2]).max()) < 1e-6
    assert float(jnp.abs(r1 - r0).max()) > 1e-3


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("solver", ["rk4", "midpoint", "euler"])
def test_jax_velocity_integration_matches_torch(physical, solver):
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import integrate_time_varying_velocity_field_jax
    from syntx.core.inverse import integrate_time_varying_velocity_field
    rng = np.random.default_rng(1)
    v = [(0.05 if not physical else 0.8) * rng.standard_normal((1, 12, 14, 2)).astype('float32') for _ in range(3)]
    geo = dict(spacing=(1.2, 0.9), origin=(1.0, -2.0), direction=np.eye(2)) if physical else {}
    for mode in ("forward", "backward"):
        pj = np.asarray(integrate_time_varying_velocity_field_jax([jnp.array(x) for x in v], mode=mode, solver=solver, **geo))
        pt = integrate_time_varying_velocity_field([torch.tensor(x) for x in v], mode=mode, solver=solver, **geo).numpy()
        np.testing.assert_allclose(pj, pt, atol=2e-5)
    with pytest.raises(ValueError):
        integrate_time_varying_velocity_field_jax([jnp.array(x) for x in v], solver='heun', **geo)
    with pytest.raises(ValueError):
        integrate_time_varying_velocity_field_jax([jnp.array(x) for x in v], mode='bwd', **geo)


@pytest.mark.parametrize("opt", ["cfl", "sgd", "adam", "rprop"])
def test_syn_jax_optimizers_run(opt):
    import jax.numpy as jnp
    from syntx.syn_jax import SyNTo
    f, m = _pair()
    I = jnp.array(f.numpy().T)[None, None]
    J = jnp.array(m.numpy().T)[None, None]
    mdl = SyNTo(dim=2, grid_shape=(32, 32), elastic_sigma=0.5)
    mdl.fit(I, J, levels=[2, 1], epochs_per_level=[3, 2], affine_epochs=0, optimizer_type=opt,
            optimizer_lr=0.05)
    assert np.isfinite(np.asarray(mdl.warp_l2r)).all() and len(mdl.syn_losses) == 5


def test_syn_jax_constructor_fails_loudly():
    from syntx.syn_jax import SyNTo
    with pytest.raises(NotImplementedError, match="boundary_suppression_thresh"):
        SyNTo(dim=2, grid_shape=(8, 8), boundary_suppression_thresh=0.05)
    with pytest.raises(TypeError):
        SyNTo(dim=2, grid_shape=(8, 8), velocity_clamp=1.0)
    with pytest.raises(ValueError, match="transform_type"):
        SyNTo(dim=2, grid_shape=(8, 8), transform_type='SyN')


def test_syn_jax_lbfgs_inverts_on_the_fixed_grid(monkeypatch):
    """Both half fields live on the fixed (midpoint) grid: the L-BFGS objective inverted the r2l
    field with the MOVING geometry."""
    import jax.numpy as jnp
    import syntx.syn_jax as sj
    seen = []
    real = sj.update_inverse_field_nd_jax

    def spy(*a, **k):
        seen.append(tuple(k.get('spacing') or ()))
        return real(*a, **k)

    monkeypatch.setattr(sj, "update_inverse_field_nd_jax", spy)
    f, m = _pair(16)
    I = jnp.array(f.numpy().T)[None, None]
    J = jnp.array(m.numpy().T)[None, None]
    mdl = sj.SyNTo(dim=2, grid_shape=(16, 16))
    mdl.fit(I, J, levels=[1], epochs_per_level=[1], affine_epochs=0, optimizer_type='lbfgs',
            optimizer_lr=1.0, fixed_spacing=[1.0, 1.0], moving_spacing=[2.0, 3.0])
    assert seen and all(sp in ((1.0, 1.0), ()) for sp in seen), sorted(set(seen))


def test_jax_soft_dice_matches_torch_and_registers():
    import torch
    import jax
    import jax.numpy as jnp
    import syntx
    from syntx.syn_jax import soft_dice_loss_nd_jax
    from syntx.core.losses import soft_dice_loss_nd
    rng = np.random.default_rng(0)
    a, b = (rng.random((1, 2, 9, 11)).astype('float32') for _ in range(2))
    msk = (rng.random((1, 1, 9, 11)) > 0.3).astype('float32')
    vj = float(jax.jit(soft_dice_loss_nd_jax)(jnp.array(a), jnp.array(b), jnp.array(msk)))
    vt = float(soft_dice_loss_nd(torch.tensor(a), torch.tensor(b), mask=torch.tensor(msk)))
    assert abs(vj - vt) < 1e-6
    f, m = _pair()
    r = syntx.syn([f, f], [m, m], backend="jax", syn_metric=["cc2", "dice"], reg_iterations=[3, 2])
    assert np.isfinite(r["warpedmovout"].numpy()).all()


def test_syn_jax_forward_applies_the_affine_in_the_right_axes():
    """forward() with a zero warp and a known physical affine must match ants.apply_transforms
    (M_phys, already in tensor order, was re-permuted back to (x, y, z))."""
    import jax.numpy as jnp
    from syntx.syn_jax import SyNTo
    nx, ny = 24, 20                                            # ANTs (x, y) sizes
    yy, xx = np.mgrid[:ny, :nx]
    arr = np.exp(-((xx - 9.0) ** 2 / 18.0 + (yy - 11.0) ** 2 / 8.0)).astype('float32').T   # (x, y)
    img = ants.from_numpy(arr)
    M = np.array([[1.1, 0.25], [-0.1, 0.9]])
    t = np.array([1.5, -2.0])
    tx = ants.create_ants_transform(transform_type='AffineTransform', dimension=2, matrix=M, translation=t)
    ref = ants.apply_ants_transform_to_image(tx, img, img, interpolation='linear').numpy()

    def H(n):                                                  # normalised -> physical, (x, y)
        h = np.eye(3)
        h[:2, :2] = np.diag((np.array(n) - 1) / 2.0)
        h[:2, 2] = (np.array(n) - 1) / 2.0
        return h

    T_phys = np.eye(3)
    T_phys[:2, :2], T_phys[:2, 2] = M, t
    mdl = SyNTo(dim=2, grid_shape=(ny, nx), spacing=[1.0, 1.0], origin=[0.0, 0.0])
    mdl.warp_l2r = np.zeros((1, ny, nx, 2), dtype='float32')
    mdl.affine_params['T_init'] = np.linalg.inv(H((nx, ny))) @ T_phys @ H((nx, ny))
    out = np.asarray(mdl.forward(jnp.array(arr)[None, None])).squeeze()
    inner = (slice(3, -3), slice(3, -3))
    np.testing.assert_allclose(out[inner], ref[inner], atol=2e-3)
    # forward_inverse with a zero warp_r2l applies the inverse affine
    mdl.warp_r2l = np.zeros((1, ny, nx, 2), dtype='float32')
    ref_inv = ants.apply_ants_transform_to_image(tx.invert(), img, img, interpolation='linear').numpy()
    out_inv = np.asarray(mdl.forward_inverse(jnp.array(arr)[None, None])).squeeze()
    np.testing.assert_allclose(out_inv[inner], ref_inv[inner], atol=1e-2)   # edge padding differs; an axis error is ~1


def _geom_t(shape, spacing, origin, direction):
    import jax.numpy as jnp
    return (jnp.array(list(shape)), jnp.array(list(reversed(spacing))), jnp.array(list(reversed(origin))),
            jnp.array(np.asarray(direction, dtype=float)[::-1, ::-1].copy()))


def test_syn_jax_initial_grid_uses_fixed_geometry_in_both_paths():
    """The initial grid lives on the fixed grid: the autograd path (warp_images_jax) indexed it
    with the MOVING geometry, the analytic path with the fixed one."""
    import jax.numpy as jnp
    from syntx.syn_jax import warp_images_jax, prepare_mid_images_and_gradients_jax, get_physical_grid_jax
    shp, mshp = (12, 14), (10, 9)
    fsp, msp = [1.0, 1.0], [2.0, 1.5]
    X = get_physical_grid_jax(shp, fsp, [0.0, 0.0], np.eye(2))
    rng = np.random.default_rng(0)
    I = jnp.array(rng.random((1, 1) + shp).astype('float32'))
    J = jnp.array(rng.random((1, 1) + mshp).astype('float32'))
    yy, xx = np.meshgrid(np.linspace(-1, 1, shp[0]), np.linspace(-1, 1, shp[1]), indexing='ij')
    init = jnp.array(np.stack([0.8 * xx + 0.1 * yy, 0.7 * yy - 0.05], -1)[None].astype('float32'))
    z = jnp.zeros((1,) + shp + (2,))
    ft = _geom_t(shp, fsp, [0.0, 0.0], np.eye(2))
    mt = _geom_t(mshp, msp, [3.0, -1.0], np.eye(2))
    M, t = jnp.eye(2), jnp.zeros(2)
    _, jm_auto = warp_images_jax(z, z, z, z, I, J, X, *ft, *mt, M, t, init)
    out = prepare_mid_images_and_gradients_jax(z, z, z, z, I, J, X, *ft, *mt, tuple(fsp), tuple(msp), M, t, init)
    np.testing.assert_allclose(np.asarray(jm_auto), np.asarray(out[1]), atol=1e-6)


@pytest.mark.parametrize("physical", [False, True])
def test_jax_anderson_inverse_matches_torch(physical):
    """Same field, same steps: the JAX Anderson inverse must follow the PyTorch one (its
    safeguard compared the candidate's scaled composition error with the unscaled step)."""
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import update_inverse_field_nd_jax_anderson
    from syntx.core.inverse import update_inverse_field_nd_anderson
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:20, :18]
    amp = 2.5 if physical else 0.12
    w = np.stack([amp * np.sin(xx / 3.0) * np.cos(yy / 4.0), amp * np.cos(xx / 5.0)], -1)[None].astype('float32')
    geo = dict(spacing=(1.3, 0.8), origin=(0.0, 0.0), direction=np.eye(2)) if physical else {}
    for steps in (3, 8, 20):
        vj = np.asarray(update_inverse_field_nd_jax_anderson(jnp.array(w), None, steps=steps, **geo))
        vt = update_inverse_field_nd_anderson(torch.tensor(w), None, steps=steps, **geo).numpy()
        np.testing.assert_allclose(vj, vt, atol=1e-4)


@pytest.mark.parametrize("physical", [False, True])
def test_jax_hybrid_lm_inverse_matches_torch(physical):
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import update_inverse_field_jax_hybrid_lm
    from syntx.core.inverse import update_inverse_field_nd_hybrid_lm
    yy, xx = np.mgrid[:20, :18]
    amp = 2.5 if physical else 0.12
    w = np.stack([amp * np.sin(xx / 3.0) * np.cos(yy / 4.0), amp * np.cos(xx / 5.0)], -1)[None].astype('float32')
    geo = dict(spacing=(1.3, 0.8), origin=(0.0, 0.0), direction=np.eye(2)) if physical else {}
    vj = np.asarray(update_inverse_field_jax_hybrid_lm(jnp.array(w), None, steps=10, **geo))
    vt = update_inverse_field_nd_hybrid_lm(torch.tensor(w), None, steps=10, **geo).numpy()
    np.testing.assert_allclose(vj, vt, atol=1e-4)


@pytest.mark.parametrize("op", ["sobolev", "dsti"])
def test_jax_green_operators_match_torch_with_anisotropic_spacing(op):
    import torch
    import jax.numpy as jnp
    from syntx.syn_jax import SyNTo
    from syntx.core.smoothing import apply_sobolev_green_operator, apply_dsti_green_operator
    rng = np.random.default_rng(5)
    m = rng.standard_normal((1, 12, 17, 2)).astype('float32')
    sp = [2.0, 0.7]                                            # ITK (x, y)
    mdl = SyNTo(dim=2, grid_shape=(12, 17))
    if op == "sobolev":
        vj = mdl._apply_sobolev_green_operator(jnp.array(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
        vt = apply_sobolev_green_operator(torch.tensor(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
    else:
        vj = mdl._apply_dsti_green_operator(jnp.array(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
        vt = apply_dsti_green_operator(torch.tensor(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
    np.testing.assert_allclose(np.asarray(vj), vt.numpy(), atol=1e-4)



def test_syngs_jax_fails_loudly_for_what_it_does_not_implement():
    import syntx
    f, m = _pair()
    with pytest.raises(ValueError, match="reg_adam"):            # syngs' default optimizer
        syntx.syngs(f, m, backend="jax", reg_iterations=[2])
    with pytest.raises(TypeError, match="transport_mode"):
        syntx.syngs(f, m, backend="jax", optimizer="cfl", reg_iterations=[2], transport_mode='transport')
    with pytest.raises(ValueError, match="similarity_metric"):
        syntx.syngs(f, m, backend="jax", optimizer="cfl", reg_iterations=[2], syn_metric='box_lncc')
    m2 = ants.from_numpy(m.numpy(), spacing=(1.5, 1.0))
    with pytest.raises(ValueError, match="fixed grid"):
        syntx.syngs(f, m2, backend="jax", optimizer="cfl", reg_iterations=[2])
    from syntx.syngs_jax import GeodesicShootingModelJAX
    with pytest.raises(ValueError, match="solver"):
        GeodesicShootingModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8), solver='heun')
    with pytest.raises(TypeError):
        GeodesicShootingModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8), cfl_max=0.4)
    mdl = GeodesicShootingModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8))
    with pytest.raises(TypeError, match="multipoint_loss"):
        mdl.fit(np.zeros((1, 1, 8, 8)), np.zeros((1, 1, 8, 8)), levels=[1], epochs_per_level=[1],
                affine_epochs=0, multipoint_loss=[0.5])


def test_syngs_jax_metric_and_solvers():
    import torch
    import jax.numpy as jnp
    from syntx.syngs_jax import GeodesicShootingModelJAX
    from syntx.core.losses import local_ncc_loss_nd
    rng = np.random.default_rng(0)
    a, b = (rng.random((1, 1, 16, 16)).astype('float32') for _ in range(2))
    for metric, sq in (('lncc', False), ('cc2', True)):
        mdl = GeodesicShootingModelJAX(dim=2, image_shape=(16, 16), velocity_shape=(16, 16), similarity_metric=metric)
        vj = float(mdl._eval_similarity(jnp.array(a), jnp.array(b), 5))
        vt = float(local_ncc_loss_nd(torch.tensor(a), torch.tensor(b), window_size=5, squared=sq))
        assert abs(vj - vt) < 1e-4
    yy, xx = np.mgrid[:16, :16]
    v0 = jnp.array(np.stack([0.8 * np.sin(xx / 3.0), 0.6 * np.cos(yy / 4.0)], -1)[None].astype('float32'))
    out = {}
    for solver in ('euler', 'midpoint', 'rk4'):
        mdl = GeodesicShootingModelJAX(dim=2, image_shape=(16, 16), velocity_shape=(16, 16), solver=solver, n_steps=4)
        out[solver] = np.asarray(mdl.shoot(v0, 4))
    assert np.abs(out['midpoint'] - out['euler']).max() > 1e-4            # midpoint is not Euler
    assert np.abs(out['midpoint'] - out['rk4']).max() < np.abs(out['euler'] - out['rk4']).max()


def test_level_spacing_itk_pairs_reversed_shapes():
    from syntx.syn_jax import level_spacing_itk
    # ITK spacing (x=1, y=2) on a tensor (y=11, x=21) grid resampled to (6, 11)
    assert level_spacing_itk([1.0, 2.0], (11, 21), (6, 11)) == [1.0 * 20 / 10, 2.0 * 10 / 5]


def test_syngs_jax_nonsymmetric_inverse_is_minus_v0(monkeypatch):
    """Not symmetric: the loss must shoot the inverse from -v0 (fit passed +v0)."""
    import jax
    import jax.numpy as jnp
    from syntx.syngs_jax import GeodesicShootingModelJAX
    f, m = _pair(16)
    mdl = GeodesicShootingModelJAX(dim=2, image_shape=(16, 16), velocity_shape=(16, 16), symmetric=False)
    mdl.velocity_0_fwd = jnp.full((1, 16, 16, 2), 0.3, dtype=jnp.float32)      # non-zero start
    seen = []
    real = mdl.forward

    def spy(fi, mo, velocity_0_fwd=None, velocity_0_inv=None, **k):
        jax.debug.callback(lambda r: seen.append(float(r)), jnp.max(jnp.abs(velocity_0_inv + velocity_0_fwd)))
        return real(fi, mo, velocity_0_fwd=velocity_0_fwd, velocity_0_inv=velocity_0_inv, **k)

    monkeypatch.setattr(mdl, "forward", spy)
    mdl.fit(jnp.array(f.numpy().T)[None, None], jnp.array(m.numpy().T)[None, None], levels=[1],
            epochs_per_level=[2], affine_epochs=0)
    assert seen and max(seen) < 1e-6, seen


def test_syngs_jax_init_velocities_resize_channels_last():
    import jax.numpy as jnp
    from syntx.syngs_jax import GeodesicShootingModelJAX
    mdl = GeodesicShootingModelJAX(dim=2, image_shape=(16, 20), velocity_shape=(8, 10))
    rng = np.random.default_rng(0)
    img = jnp.array(rng.random((1, 1, 16, 20)).astype('float32'))
    mdl.init_velocities_from_image_gradients(img, img)
    assert mdl.velocity_0_fwd.shape == (1, 8, 10, 2) and mdl.velocity_0_inv.shape == (1, 8, 10, 2)
    assert bool(jnp.isfinite(mdl.velocity_0_fwd).all())


@pytest.mark.parametrize("op", ["sobolev", "dsti"])
def test_tvf_jax_green_operators_match_torch_with_anisotropic_spacing(op):
    import torch
    import jax.numpy as jnp
    from syntx.tvf_jax import TVFModelJAX
    from syntx.core.smoothing import apply_sobolev_green_operator, apply_dsti_green_operator
    rng = np.random.default_rng(6)
    m = rng.standard_normal((1, 12, 17, 2)).astype('float32')
    sp = [2.0, 0.7]
    mdl = TVFModelJAX(dim=2, image_shape=(12, 17), velocity_shape=(12, 17))
    if op == "sobolev":
        vj = mdl._apply_sobolev_green_operator(jnp.array(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
        vt = apply_sobolev_green_operator(torch.tensor(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
    else:
        vj = mdl._apply_dsti_green_operator(jnp.array(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
        vt = apply_dsti_green_operator(torch.tensor(m), fluid_sigma=2.0, alpha=1.5, spacing=sp)
    np.testing.assert_allclose(np.asarray(vj), vt.numpy(), atol=1e-4)


def test_tvf_jax_metric_and_fail_loud():
    import syntx
    import jax.numpy as jnp
    from syntx.tvf_jax import TVFModelJAX
    f, m = _pair()
    kw = dict(backend='jax', regularizer='gaussian', optimizer='cfl', reg_iterations=[3, 2])
    out = {met: syntx.tvf(f, m, syn_metric=met, **kw)['warpedmovout'].numpy() for met in ('cc2', 'mattes')}
    assert all(np.isfinite(v).all() for v in out.values())
    assert np.abs(out['cc2'] - out['mattes']).max() > 1e-4          # the metric is used
    with pytest.raises(ValueError, match="similarity_metric"):
        syntx.tvf(f, m, syn_metric='box_lncc', **kw)
    with pytest.raises(TypeError, match="bootstrap_mode"):
        syntx.tvf(f, m, bootstrap_mode='antithetic', **kw)
    with pytest.raises(ValueError, match="energy_weight"):
        syntx.tvf(f, m, energy_weight=0.01, **kw)
    with pytest.raises(TypeError):
        TVFModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8), use_analytical_gradients=True)
    mdl = TVFModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8))
    with pytest.raises(TypeError, match="max_step_norm"):
        mdl.fit(jnp.zeros((1, 1, 8, 8)), jnp.zeros((1, 1, 8, 8)), levels=[1], epochs_per_level=[1],
                affine_epochs=0, max_step_norm=1.0)
