"""
Optimisers and step-size helpers for velocity-field registration: ``RegAdam`` (Adam whose step
is spatially smoothed and CFL-bounded; aliases ``SobolevAdam``, ``GaussianAdam``), ``LARS``,
and the helpers ``get_cfl_max_norm``, ``compute_cfl_step``, ``check_convergence``.
"""

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np
import torch

from .regularizers import (
    COUPLED_CONTINUUM_REGULARIZERS,
    SPECTRAL_REGULARIZERS,
    _require_spacing,
    get_regularizer,
)


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
_REGADAM_REGULARIZERS = SPECTRAL_REGULARIZERS | frozenset({
    'gaussian', 'gauss', 'none', 'identity',
    'compact_gaussian', 'compact', 'fast_gaussian', 'erf', 'bspline',
})


class RegAdam(torch.optim.Optimizer):
    """
    Adam whose step is spatially smoothed and then bounded (CFL-style), for dense velocity
    fields.

    Per parameter tensor: standard Adam moments with bias correction give the point-wise
    direction d = m_hat / (sqrt(v_hat) + floor); d is smoothed by the selected regulariser,
    scaled down if lr * max_x |d(x) / spacing| > ``max_step_norm``, and applied as
    p <- p - lr * d.

    Parameters are expected channel-last, (B, *spatial, dim) (4-D / 5-D tensors) or
    (B, 1, *spatial, dim) (squeezed for smoothing); other shapes are not smoothed.
    """
    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8,
                 regularizer='sobolev', regularizer_fn=None,
                 sobolev_alpha=None, dsti_alpha=None, gaussian_sigma=1.5,
                 max_step_norm=0.50, spacing=None, eps_rel=0.0,
                 origin=None, direction=None, extra_reg_kwargs=None):
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
              ``sobolev_alpha`` (legacy fallback 0.035 when None) and ``spacing``.
            - 'dsti' / 'dsti1' (the same operator): ``apply_dsti_green_operator`` (zero
              Dirichlet) with alpha = ``dsti_alpha`` and ``spacing``.
            - 'gaussian' / 'gauss' / 'bspline': ``separable_gaussian_filter`` (Bessel, replicate
              padding), sigma = ``gaussian_sigma`` voxels (none if <= 0; 'bspline' models smooth
              the field themselves, the step is Gaussian-smoothed).
            - 'compact_gaussian' / 'compact' / 'fast_gaussian' / 'erf':
              ``fast_separable_gaussian_filter`` (zero padding), sigma = ``gaussian_sigma``.
            - continuum operators: forwarded to ``get_regularizer`` with ``spacing``,
              ``origin``, ``direction``, and ``extra_reg_kwargs``.
            - 'none': no smoothing.
            Other strings raise ValueError.
        regularizer_fn : callable, optional
            d -> smoothed d, used instead of ``regularizer``.
        sobolev_alpha : float, optional
            alpha of the spectral regularisers (None lets continuum operators derive their
            own default; legacy 'sobolev' falls back to 0.035).
        dsti_alpha : float, optional
            alpha of 'dsti' / 'dsti1' (default ``sobolev_alpha``).
        gaussian_sigma : float, default 1.5
            Gaussian sigma in voxels (``spacing`` has no effect on it).
        max_step_norm : float, default 0.50
            Bound on lr * max |d / spacing| (voxels when spacing is in mm); None or <= 0
            disables it. The whole tensor is rescaled, not clipped voxel-wise.
        spacing : sequence, optional
            ANTs (x, y, z) order; used by spectral/continuum operators and the per-axis
            anisotropic step bound (None = unit spacing).
        eps_rel : float
            Relative floor of the Adam denominator: ``max(eps, eps_rel * max(sqrt(v_hat)))``.
        origin, direction : optional
            ITK physical origin and direction cosine matrix forwarded to continuum
            operators.
        extra_reg_kwargs : dict, optional
            Extra keyword arguments forwarded to ``get_regularizer``.
        """
        if regularizer_fn is None and str(regularizer).lower() not in _REGADAM_REGULARIZERS:
            raise ValueError(f"RegAdam: unknown regularizer {regularizer!r}; expected one of "
                             f"{sorted(_REGADAM_REGULARIZERS)} or a regularizer_fn")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, eps_rel=eps_rel,
            regularizer=regularizer, regularizer_fn=regularizer_fn,
            sobolev_alpha=sobolev_alpha,
            dsti_alpha=dsti_alpha if dsti_alpha is not None else sobolev_alpha,
            gaussian_sigma=gaussian_sigma,
            max_step_norm=max_step_norm, spacing=spacing,
            origin=origin, direction=direction,
            extra_reg_kwargs=dict(extra_reg_kwargs) if extra_reg_kwargs is not None else {},
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
            reg_mode = str(group.get('regularizer', 'sobolev')).lower()
            reg_fn = group.get('regularizer_fn')
            alpha = group.get('sobolev_alpha', None)
            dsti_alpha = group.get('dsti_alpha', alpha)
            gauss_sig = group.get('gaussian_sigma', 1.5)
            spacing = group.get('spacing')
            origin = group.get('origin')
            direction = group.get('direction')
            extra_reg_kwargs = group.get('extra_reg_kwargs') or {}
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
                elif reg_mode == 'sobolev':
                    alpha_sob = 0.035 if alpha is None else float(alpha)
                    if alpha_sob > 0:
                        from .smoothing import apply_sobolev_green_operator
                        if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                            s = raw_step.squeeze(1)
                            smooth_s = apply_sobolev_green_operator(
                                s, fluid_sigma=alpha_sob, alpha=alpha_sob, spacing=spacing
                            )
                            smooth_step = smooth_s.unsqueeze(1)
                        elif raw_step.ndim in (4, 5):
                            smooth_step = apply_sobolev_green_operator(
                                raw_step, fluid_sigma=alpha_sob, alpha=alpha_sob, spacing=spacing
                            )
                        else:
                            smooth_step = raw_step
                    else:
                        smooth_step = raw_step
                elif reg_mode in ('dsti', 'dsti1', 'dst_i', 'dst1', 'dirichlet'):
                    d_alpha = 0.035 if dsti_alpha is None else float(dsti_alpha)
                    if d_alpha > 0:
                        # 'dsti' and 'dsti1' are the same operator (apply_dsti1_green_operator is an alias)
                        from .smoothing import apply_dsti_green_operator
                        if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                            s = raw_step.squeeze(1)
                            smooth_step = apply_dsti_green_operator(
                                s, fluid_sigma=d_alpha, alpha=d_alpha, spacing=spacing
                            ).unsqueeze(1)
                        elif raw_step.ndim in (4, 5):
                            smooth_step = apply_dsti_green_operator(
                                raw_step, fluid_sigma=d_alpha, alpha=d_alpha, spacing=spacing
                            )
                        else:
                            smooth_step = raw_step
                    else:
                        smooth_step = raw_step
                elif reg_mode not in ('none', 'identity'):
                    if alpha is not None and float(alpha) <= 0.0:
                        smooth_step = raw_step
                    else:
                        reg_builder = get_regularizer(
                            reg_mode,
                            alpha=alpha,
                            sobolev_alpha=alpha,
                            fluid_sigma=gauss_sig if gauss_sig is not None else 3.0,
                            spacing=spacing,
                            origin=origin,
                            direction=direction,
                            **extra_reg_kwargs,
                        )
                        smooth_step = reg_builder(raw_step)
                else:
                    smooth_step = raw_step

                # Enforce anisotropic per-axis CFL step bound on device (no host sync)
                if max_step_norm is not None and max_step_norm > 0:
                    if spacing is not None:
                        from .inverse import _scaled_sq_sum
                        sp = _require_spacing(spacing, smooth_step.shape[-1], "RegAdam")
                        sp_zyx_t = torch.tensor(
                            tuple(reversed(sp)),
                            device=smooth_step.device,
                            dtype=smooth_step.dtype,
                        )
                        effective_step = _scaled_sq_sum(smooth_step, sp_zyx_t, divide=True).amax().sqrt() * lr
                    else:
                        effective_step = torch.sqrt(torch.sum(smooth_step ** 2, dim=-1)).amax() * lr
                    scale = torch.clamp(max_step_norm / effective_step.clamp_min(1e-6), max=1.0)
                    smooth_step = smooth_step * scale

                p.sub_(smooth_step, alpha=lr)

        return loss


# Aliases for backwards compatibility and specialized naming
SobolevAdam = RegAdam
GaussianAdam = RegAdam


_PLAIN_COMPONENTWISE_OPTIMIZERS = frozenset({'adam', 'adamw', 'rmsprop', 'rprop'})


def validate_regularizer_optimizer_pair(
    regularizer: str,
    optimizer: Optional[str],
    caller: str = "syntx",
) -> str:
    """Resolve default optimizer and validate compatibility with ``regularizer``.

    When ``optimizer is None``, automatically resolves to ``'cfl'`` for
    ``SPECTRAL_REGULARIZERS`` and ``'adam'`` for spatial/identity regularizers.
    When an explicit plain componentwise optimizer (``'adam'``, ``'adamw'``,
    ``'rmsprop'``, ``'rprop'``) is combined with a coupled continuum PDE regularizer
    (``COUPLED_CONTINUUM_REGULARIZERS``), raises ``ValueError`` because componentwise
    variance/sign scaling breaks ``div(v)`` / ``curl(v)`` operator coupling.
    """
    reg = str(regularizer).lower()
    if optimizer is None:
        return 'cfl' if reg in SPECTRAL_REGULARIZERS else 'adam'
    opt = str(optimizer).lower()
    if reg in COUPLED_CONTINUUM_REGULARIZERS and opt in _PLAIN_COMPONENTWISE_OPTIMIZERS:
        raise ValueError(
            f"{caller}: optimizer={optimizer!r} is incompatible with coupled continuum "
            f"regularizer={regularizer!r} because componentwise variance/sign division "
            f"destroys div(v)/curl(v) PDE coupling. Use optimizer='cfl' or optimizer='reg_adam'."
        )
    return opt


class FieldStepOptimizer:
    """Stateful, allocation-free, zero-host-sync field update engine for dense registration.

    Supports ``mode in ('cfl', 'adam', 'reg_adam')`` across both
    ``param_domain='normalized_grid'`` (``greedy``, ``scattered``) and
    ``param_domain='physical_mm'`` (``syn``, ``tvf``, ``syngs``).
    """

    def __init__(
        self,
        mode: str = 'cfl',
        param_domain: str = 'normalized_grid',
        *,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        cfl_momentum: float = 0.85,
        sobolev_alpha: Optional[float] = None,
    ):
        m = str(mode).lower()
        if m == 'regadam':
            m = 'reg_adam'
        if m not in ('cfl', 'adam', 'reg_adam'):
            raise ValueError(
                f"FieldStepOptimizer: mode must be 'cfl', 'adam', or 'reg_adam', got {mode!r}"
            )
        dom = str(param_domain).lower()
        if dom not in ('normalized_grid', 'physical_mm'):
            raise ValueError(
                f"FieldStepOptimizer: param_domain must be 'normalized_grid' or 'physical_mm', "
                f"got {param_domain!r}"
            )
        self.mode = m
        self.param_domain = dom
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.cfl_momentum = float(cfl_momentum)
        self.sobolev_alpha = None if sobolev_alpha is None else float(sobolev_alpha)

        self.vel_mom: Optional[torch.Tensor] = None
        self.exp_avg: Optional[torch.Tensor] = None
        self.exp_avg_sq: Optional[torch.Tensor] = None
        self._step: int = 0
        self._geom_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[float, ...], str, str],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
        ] = {}

    def init_state(self, template: torch.Tensor) -> None:
        """Allocate optimizer state matching ``template`` (one tensor for CFL, two for Adam)."""
        if self.mode == 'cfl':
            self.vel_mom = torch.zeros_like(template)
            self.exp_avg = None
            self.exp_avg_sq = None
        else:
            self.vel_mom = None
            self.exp_avg = torch.zeros_like(template)
            self.exp_avg_sq = torch.zeros_like(template)
        self._step = 0

    def on_level_change(self, new_shape: Sequence[int], *, interp_mode: str = 'trilinear') -> None:
        """Update optimizer buffers for a new pyramid level shape."""
        target = list(new_shape)
        if self.mode == 'cfl':
            if self.vel_mom is None or list(self.vel_mom.shape[1:-1]) != target:
                if self.vel_mom is not None:
                    dim = self.vel_mom.shape[-1]
                    self.vel_mom = torch.zeros(
                        (1, *target, dim),
                        dtype=self.vel_mom.dtype,
                        device=self.vel_mom.device,
                    )
        else:
            from .grid import resize_field
            if self.exp_avg is not None and list(self.exp_avg.shape[1:-1]) != target:
                self.exp_avg = resize_field(self.exp_avg, target, mode=interp_mode)
            if self.exp_avg_sq is not None and list(self.exp_avg_sq.shape[1:-1]) != target:
                self.exp_avg_sq = resize_field(self.exp_avg_sq, target, mode=interp_mode)
            self._step = 0

    def _get_level_geom(
        self,
        grid_shape: Sequence[int],
        spacing: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        shape_t = tuple(int(n) for n in grid_shape)
        dim = len(shape_t)
        sp_xyz = _require_spacing(spacing, dim, "FieldStepOptimizer")
        key = (shape_t, sp_xyz, str(device), str(dtype))
        cached = self._geom_cache.get(key)
        if cached is not None:
            return cached
        size_xyz = tuple(reversed(shape_t))
        sp_t_xyz = torch.tensor(sp_xyz, device=device, dtype=dtype).view(1, *([1] * dim), dim)
        sp_t_zyx = torch.tensor(tuple(reversed(sp_xyz)), device=device, dtype=dtype).view(
            1, *([1] * dim), dim
        )
        half_res_xyz = torch.tensor(
            [0.0 if n <= 1 else 1.0 / (n - 1) for n in size_xyz],
            device=device,
            dtype=dtype,
        ).view(1, *([1] * dim), dim)
        phys_ext_xyz = [max(1, n - 1) * float(s) for n, s in zip(size_xyz, sp_xyz)]
        min_ext = min(phys_ext_xyz)
        chain_scale_xyz = torch.tensor(
            [min_ext / e for e in phys_ext_xyz],
            device=device,
            dtype=dtype,
        ).view(1, *([1] * dim), dim)
        half_resolution = 1.0 / max(1, max(shape_t) - 1)
        out = (sp_t_xyz, sp_t_zyx, half_res_xyz, chain_scale_xyz, half_resolution)
        self._geom_cache[key] = out
        return out

    @torch.no_grad()
    def step(
        self,
        grad: torch.Tensor,
        learning_rate: float,
        *,
        grid_shape: Sequence[int],
        spacing: Sequence[float],
        reg_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        post_reg_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        grad_smooth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reg_smooth_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sobolev_fallback_fn: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute one displacement update ``delta`` (to be added/composed to the field)."""
        if (self.mode == 'cfl' and self.vel_mom is None) or (
            self.mode != 'cfl' and self.exp_avg is None
        ):
            self.init_state(grad)

        dim = grad.shape[-1]
        sp_t_xyz, sp_t_zyx, half_res_xyz, chain_scale_xyz, half_resolution = self._get_level_geom(
            grid_shape, spacing, grad.device, grad.dtype
        )

        if self.mode == 'cfl':
            if self.param_domain == 'normalized_grid':
                if reg_fn is not None:
                    v_vox = reg_fn(grad)
                else:
                    f_mm = grad * chain_scale_xyz
                    v_mm = grad_smooth_fn(f_mm) if grad_smooth_fn is not None else f_mm
                    v_vox = v_mm / sp_t_xyz
            else:
                v_mm = (
                    reg_fn(grad)
                    if reg_fn is not None
                    else (grad_smooth_fn(grad) if grad_smooth_fn is not None else grad)
                )
                v_vox = v_mm / sp_t_zyx

            if self.cfl_momentum > 0.0:
                self.vel_mom.mul_(self.cfl_momentum).add_(v_vox, alpha=1.0 - self.cfl_momentum)
                m_vox = self.vel_mom
                out = v_vox.clone() if v_vox.data_ptr() == grad.data_ptr() else v_vox
            else:
                m_vox = v_vox
                out = v_vox.clone() if v_vox.data_ptr() == grad.data_ptr() else v_vox

            v_norm_max = m_vox.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
            v_norm_max = v_norm_max.reshape(-1, *([1] * (dim + 1)))
            step_scale = half_res_xyz if self.param_domain == 'normalized_grid' else sp_t_zyx
            active = v_norm_max > 1e-20
            factor = torch.where(
                active,
                -float(learning_rate) * step_scale,
                torch.zeros_like(step_scale),
            )
            if out.data_ptr() != m_vox.data_ptr():
                torch.div(m_vox, v_norm_max.clamp(min=1e-20), out=out)
            else:
                out.div_(v_norm_max.clamp(min=1e-20))
            return out.mul_(factor)

        # 'adam' / 'reg_adam'
        g = grad
        if self.mode == 'reg_adam' and (reg_fn is not None or post_reg_fn is not None):
            # Continuum/spectral RegAdam: feed raw gradient into Adam moments, filter quotient ONCE
            pass
        elif reg_fn is not None:
            g = reg_fn(g)
        elif grad_smooth_fn is not None:
            g = grad_smooth_fn(g)

        self._step += 1
        self.exp_avg.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
        del g

        b1_corr = 1.0 - self.beta1 ** self._step
        b2_corr = 1.0 - self.beta2 ** self._step
        b2_sqrt = math.sqrt(b2_corr)
        step_size = b2_sqrt / b1_corr
        eps_scaled = self.eps * b2_sqrt
        num = self.exp_avg * step_size
        denom = self.exp_avg_sq.sqrt().add_(eps_scaled)
        raw_update = num.div_(denom)
        del denom

        if self.mode == 'reg_adam':
            active_post_fn = post_reg_fn if post_reg_fn is not None else reg_fn
            if active_post_fn is not None:
                u_reg = active_post_fn(raw_update)
                if u_reg.data_ptr() == raw_update.data_ptr():
                    u_reg = raw_update
                del raw_update
                max_norm = u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
                max_norm = max_norm.reshape(-1, *([1] * (dim + 1)))
                step_scale = half_res_xyz if self.param_domain == 'normalized_grid' else sp_t_zyx
                factor = torch.where(
                    max_norm > 1e-20,
                    -float(learning_rate) * step_scale,
                    torch.zeros_like(step_scale),
                )
                return u_reg.div_(max_norm.clamp(min=1e-20)).mul_(factor)
            if reg_smooth_fn is not None:
                u_reg = reg_smooth_fn(raw_update)
            else:
                ra_alpha = 0.035 if self.sobolev_alpha is None else float(self.sobolev_alpha)
                if ra_alpha > 0 and sobolev_fallback_fn is not None:
                    u_reg = sobolev_fallback_fn(raw_update, ra_alpha)
                else:
                    u_reg = raw_update
            del raw_update
        else:
            u_reg = raw_update

        gradmax = (
            self.eps + u_reg.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
        ).reshape(-1, *([1] * (dim + 1))).clamp(min=1.0)
        step_scale = half_resolution if self.param_domain == 'normalized_grid' else 1.0
        return u_reg.div_(gradmax).mul_(-float(learning_rate) * step_scale)



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


def check_convergence(losses, window_size: int = 10, slope_threshold: float = 1e-6) -> bool:
    """True when the per-iteration least-squares slope of the last ``window_size`` losses is within ``+-slope_threshold``
    (the loss has stopped changing); False while fewer values exist. The rule of ``core.convergence.LossSlopeStop``."""
    from .convergence import loss_slope

    if len(losses) < window_size:
        return False
    slope = loss_slope(range(window_size), list(losses)[-window_size:])
    return slope is not None and abs(slope) <= float(slope_threshold)


class PointStepTracker:
    """Per-frame convergence of an iterative registration by how far the sampled points move between checkpoints.

    ``y`` is the (B, S, dim) physical position of S fixed sample points under each frame's CURRENT transform (rigid, affine,
    eddy, group-bias: whatever the solver parametrises -- the test is on the displacement, not the parameters). A frame has
    converged when the RMS movement of its points between two consecutive checkpoints is below ``tol`` (same unit as ``y``)
    for ``patience`` checkpoints in a row. ``tol`` is chosen by the caller in units of the working voxel, so the criterion is
    the same relative accuracy for a mouse and a human image.
    """

    def __init__(self, tol: float, patience: int = 2):
        if not tol > 0:
            raise ValueError(f"tol must be positive, got {tol!r}")
        self.tol = float(tol)
        self.patience = int(patience)
        self.prev = None
        self.stable = None
        self.active = None
        self.last_step = None

    def start(self, y):
        """Record the starting positions; every frame is active."""
        self.prev = y.detach()
        n = y.shape[0]
        self.stable = torch.zeros(n, dtype=torch.long, device=y.device)
        self.active = torch.ones(n, dtype=torch.bool, device=y.device)
        self.last_step = torch.full((n,), float("inf"), device=y.device)

    def update(self, y, capacity=None, capacity_fraction: float = 0.25):
        """Compare with the previous checkpoint. Returns the boolean mask of frames that converged AT this checkpoint.

        ``capacity`` (same unit as ``y``) is the farthest the optimiser COULD have moved the points since the last checkpoint
        (iterations x current learning rate, expressed as a point displacement). A step that is small only because the learning
        rate is small (the tail of a cosine schedule) says nothing about convergence, so with a capacity a frame counts as
        stable only if its step is below ``tol`` AND below ``capacity_fraction`` of what the optimiser was free to move.
        Line-search optimisers (L-BFGS) have no such limit: pass ``None``.
        """
        y = y.detach()
        step = ((y - self.prev) ** 2).sum(dim=-1).mean(dim=-1).sqrt()
        self.prev = y
        self.last_step = torch.where(self.active, step, self.last_step)
        below = step < self.tol
        if capacity is not None:
            cap_bound = (
                capacity * float(capacity_fraction)
                if torch.is_tensor(capacity)
                else float(capacity_fraction) * float(capacity)
            )
            below = below & (step < cap_bound)
        self.stable = torch.where(self.active & below, self.stable + 1, torch.where(self.active, torch.zeros_like(self.stable), self.stable))
        newly = self.active & (self.stable >= self.patience)
        self.active = self.active & ~newly
        return newly

    def reject(self, mask):
        """Frames whose apparent convergence was NOT confirmed (e.g. by the manifold probe): active again, count reset."""
        self.active = self.active | mask
        self.stable = torch.where(mask, torch.zeros_like(self.stable), self.stable)

    @property
    def all_converged(self) -> bool:
        return not bool(self.active.any())
