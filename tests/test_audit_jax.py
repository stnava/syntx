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
