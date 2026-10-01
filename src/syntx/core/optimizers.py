"""
Optimisers and step-size helpers for velocity-field registration: ``RegAdam`` (Adam whose step
is spatially smoothed and CFL-bounded; aliases ``SobolevAdam``, ``GaussianAdam``), ``LARS``,
and the helpers ``get_cfl_max_norm``, ``compute_cfl_step``, ``check_convergence``.
"""

import math
import numpy as np
import torch


class LARS(torch.optim.Optimizer):
    """
    Plain gradient descent with a LARS-style trust ratio per parameter tensor (no momentum).

    Update: p <- p - lr * trust * g, trust = trust_coefficient * max(||p||, 1) / (||g|| + eps)
    (norms over the whole tensor; trust = 1 when g = 0). The step length is therefore about
    lr * trust_coefficient * max(||p||, 1), independent of the gradient magnitude.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or parameter group dicts.
    lr : float, default=0.80
        Base learning rate.
    trust_coefficient : float, default=0.05
        Trust ratio scaling factor.
    eps : float, default=1e-8
        Added to ||g||.

    ``step(closure=None)`` updates in place and returns ``closure()`` (or None).
    """
    def __init__(self, params, lr=0.80, trust_coefficient=0.05, eps=1e-8):
        defaults = dict(lr=lr, trust_coefficient=trust_coefficient, eps=eps)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            trust_coeff = group['trust_coefficient']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                p_norm = torch.norm(p)
                g_norm = torch.norm(g)
                p_norm_effective = torch.clamp(p_norm, min=1.0)

                if g_norm > 0:
                    trust_ratio = trust_coeff * p_norm_effective / (g_norm + eps)
                else:
                    trust_ratio = 1.0

                local_lr = lr * trust_ratio
                p.sub_(g * local_lr)
        return loss


# regularizer names RegAdam accepts; 'bspline' (B-spline models smooth the field themselves)
# Gaussian-smooths the step like 'gaussian'
_REGADAM_REGULARIZERS = frozenset({'sobolev', 'gaussian', 'gauss', 'dsti', 'dsti1', 'none',
                                   'compact_gaussian', 'compact', 'fast_gaussian', 'erf', 'bspline'})


class RegAdam(torch.optim.Optimizer):
    """
    Adam whose step is spatially smoothed and then bounded (CFL-style), for dense velocity
    fields.

    Per parameter tensor: standard Adam moments with bias correction give the point-wise
    direction d = m_hat / (sqrt(v_hat) + floor); d is smoothed by the selected regulariser,
    scaled down if lr * max_x |d(x)| / min(spacing) > ``max_step_norm``, and applied as
    p <- p - lr * d.

    Parameters are expected channel-last, (B, *spatial, dim) (4-D / 5-D tensors) or
    (B, 1, *spatial, dim) (squeezed for smoothing); other shapes are not smoothed.
    """
    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8,
                 regularizer='sobolev', regularizer_fn=None,
                 sobolev_alpha=0.035, dsti_alpha=None, gaussian_sigma=1.5,
                 max_step_norm=0.50, spacing=None, eps_rel=0.0):
        """
        Parameters
        ----------
        params : iterable
            Parameters or parameter-group dicts.
        lr : float, default 0.80
        betas : (float, float), default (0.9, 0.999)
        eps : float, default 1e-8
            Absolute floor added to the Adam denominator.
        regularizer : str, default 'sobolev'
            Smoothing of the Adam direction (ignored if ``regularizer_fn`` is given):
            - 'sobolev': ``apply_sobolev_green_operator`` (FFT, periodic) with alpha =
              ``sobolev_alpha`` and ``spacing``; no smoothing if alpha <= 0.
            - 'dsti' / 'dsti1' (the same operator): ``apply_dsti_green_operator`` (zero
              Dirichlet) with alpha = ``dsti_alpha`` and ``spacing``; none if alpha <= 0.
            - 'gaussian' / 'gauss' / 'bspline': ``separable_gaussian_filter`` (Bessel, replicate
              padding), sigma = ``gaussian_sigma`` voxels (none if <= 0; 'bspline' models smooth
              the field themselves, the step is Gaussian-smoothed).
            - 'compact_gaussian' / 'compact' / 'fast_gaussian' / 'erf':
              ``fast_separable_gaussian_filter`` (zero padding), sigma = ``gaussian_sigma``.
            - 'none': no smoothing.
            Other strings raise ValueError.
        regularizer_fn : callable, optional
            d -> smoothed d, used instead of ``regularizer``.
        sobolev_alpha : float, default 0.035
            alpha of the spectral regularisers.
        dsti_alpha : float, optional
            alpha of 'dsti' / 'dsti1' (default ``sobolev_alpha``).
        gaussian_sigma : float, default 1.5
            Gaussian sigma in voxels (``spacing`` has no effect on it).
        max_step_norm : float, default 0.50
            Bound on lr * max |d| / min(spacing) (voxels when spacing is in mm); None or <= 0
            disables it. The whole tensor is rescaled, not clipped voxel-wise.
        spacing : sequence, optional
            ANTs (x, y, z) order; used by 'sobolev', 'dsti' / 'dsti1' and the step bound
            (None = 1).
        eps_rel : float
            Relative floor of the Adam denominator: ``max(eps, eps_rel * max(sqrt(v_hat)))``.
            Default 0 (absolute eps only -- the behaviour the canonical parameters were tuned
            with). With a purely absolute eps, components whose gradient is ~0 (background) are
            normalised to full-size steps, so any gradient noise there becomes displacement;
            e.g. 1e-2 keeps such components ~100x smaller. It was the first mitigation for MPS
            run-to-run differences (v5.4.65); the root cause -- non-deterministic GPU backward
            kernels -- is now fixed at the source (core.grid.DeterministicGridSample,
            core.losses.box_mean_nd), so it is opt-in (syngs/tvf keyword ``adam_eps_rel``).
        """
        if regularizer_fn is None and regularizer not in _REGADAM_REGULARIZERS:
            raise ValueError(f"RegAdam: unknown regularizer {regularizer!r}; expected one of "
                             f"{sorted(_REGADAM_REGULARIZERS)} or a regularizer_fn")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, eps_rel=eps_rel,
            regularizer=regularizer, regularizer_fn=regularizer_fn,
            sobolev_alpha=sobolev_alpha, dsti_alpha=dsti_alpha if dsti_alpha is not None else sobolev_alpha,
            gaussian_sigma=gaussian_sigma,
            max_step_norm=max_step_norm, spacing=spacing,
        )
        super(RegAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """One RegAdam update of every parameter with a gradient (in place). Returns
        ``closure()`` (evaluated with grad enabled) or None. The step bound is computed on the
        device; only ``eps_rel`` > 0 synchronises (``float(denom.max())``)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            eps_rel = group.get('eps_rel', 0.0)
            reg_mode = group.get('regularizer', 'sobolev')
            reg_fn = group.get('regularizer_fn')
            alpha = group.get('sobolev_alpha', 0.035)
            dsti_alpha = group.get('dsti_alpha', alpha)
            gauss_sig = group.get('gaussian_sigma', 1.5)
            spacing = group.get('spacing')
            max_step_norm = group.get('max_step_norm', 0.50)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                k = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Standard Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** k
                bias_corr2 = 1.0 - beta2 ** k

                # Raw point-wise step direction quotient
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_corr2)
                floor = max(eps, eps_rel * float(denom.max())) if eps_rel > 0 else eps
                denom = denom.add_(floor)
                raw_step = (exp_avg / bias_corr1) / denom

                # Apply elected regularization directly to the Adam step direction
                if reg_fn is not None:
                    smooth_step = reg_fn(raw_step)
                elif reg_mode in ('compact_gaussian', 'compact', 'fast_gaussian', 'erf'):
                    from .smoothing import fast_separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = fast_separable_gaussian_filter(s, sigma=gauss_sig)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = fast_separable_gaussian_filter(raw_step, sigma=gauss_sig)
                    else:
                        smooth_step = raw_step
                elif reg_mode in ('gaussian', 'gauss', 'bspline') and gauss_sig is not None and gauss_sig > 0:
                    from .smoothing import separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = separable_gaussian_filter(s, sigma=gauss_sig)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = separable_gaussian_filter(raw_step, sigma=gauss_sig)
                    else:
                        smooth_step = raw_step
                elif reg_mode == 'sobolev' and alpha is not None and alpha > 0:
                    from .smoothing import apply_sobolev_green_operator
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = apply_sobolev_green_operator(s, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = apply_sobolev_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                    else:
                        smooth_step = raw_step
                elif reg_mode in ('dsti', 'dsti1') and dsti_alpha is not None and dsti_alpha > 0:
                    # 'dsti' and 'dsti1' are the same operator (apply_dsti1_green_operator is an alias)
                    from .smoothing import apply_dsti_green_operator
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_step = apply_dsti_green_operator(s, fluid_sigma=dsti_alpha, alpha=dsti_alpha,
                                                                spacing=spacing).unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = apply_dsti_green_operator(raw_step, fluid_sigma=dsti_alpha, alpha=dsti_alpha,
                                                                spacing=spacing)
                    else:
                        smooth_step = raw_step
                else:
                    smooth_step = raw_step

                # Enforce Courant-Friedrichs-Lewy (CFL) step bound to prevent discrete trajectory crossover
                if max_step_norm is not None and max_step_norm > 0:
                    # on the device (no host sync): scale = min(1, max_step_norm / (lr * max|d| / min_sp))
                    min_sp = min(spacing) if spacing is not None else 1.0
                    step_mag = torch.sqrt(torch.sum(smooth_step ** 2, dim=-1))
                    effective_step = step_mag.max() / max(min_sp, 1e-4) * lr
                    scale = torch.clamp(max_step_norm / effective_step.clamp_min(1e-6), max=1.0)
                    smooth_step = smooth_step * scale

                p.sub_(smooth_step, alpha=lr)

        return loss


# Aliases for backwards compatibility and specialized naming
SobolevAdam = RegAdam
GaussianAdam = RegAdam



def get_cfl_max_norm(velocity: torch.Tensor, spacing: list) -> float:
    """
    max over voxels of |v(x) / spacing| for a channel-last field ``velocity`` (..., dim): each
    component divided by its spacing (``spacing`` in the same order as the components), i.e.
    the largest displacement in voxels. Returns a Python float (device sync).
    """
    device = velocity.device
    dim = velocity.shape[-1]
    # Normalize velocity vectors by voxel spacing
    spacing_t = torch.tensor(spacing, device=device, dtype=torch.float32).view(*([1] * (velocity.ndim - 1)), dim)
    v_norm_voxel = velocity / spacing_t
    max_norm = torch.max(torch.linalg.norm(v_norm_voxel, dim=-1)).item()
    return max_norm


def compute_cfl_step(kwargs: dict, shrink_ratio: float, default_grad_step: float = 0.25) -> float:
    """
    Step size for a pyramid level: ``kwargs['cfl_step']`` (else ``kwargs['grad_step']``, else
    ``default_grad_step``) times sqrt(shrink_ratio). Returns a float.
    """
    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', default_grad_step)))
    return float(cfl_step_val) * math.sqrt(shrink_ratio)


def check_convergence(losses, window_size: int = 10, slope_threshold: float = 1e-8) -> bool:
    """
    True if the least-squares slope (loss per iteration) of the last ``window_size`` values of
    ``losses`` has absolute value <= ``slope_threshold``; False while fewer values exist.
    """
    if len(losses) < window_size:
        return False
    y = np.array(losses[-window_size:])
    x = np.arange(window_size)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-8:
        return False
    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    return abs(slope) <= slope_threshold
