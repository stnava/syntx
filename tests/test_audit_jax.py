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
    r = syntx.syngs(f, m, backend="jax", reg_iterations=[3, 2])
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
