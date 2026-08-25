import math
import numpy as np
import torch


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) Optimizer for TVF/SyNGS Velocity Parameters.

    Rescales parameter update magnitudes using trust ratio scaling:
    $$\\text{trust\\_ratio} = \\eta \\cdot \\frac{\\max(\\|p\\|_2, 1.0)}{\\|g\\|_2 + \\epsilon}$$

    Prevents momentum collapse in smooth LNCC similarity plateaus during non-linear deformable optimization.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or parameter group dicts.
    lr : float, default=0.80
        Base learning rate.
    trust_coefficient : float, default=0.05
        Trust ratio scaling factor $\\eta$.
    eps : float, default=1e-8
        Numerical stability epsilon denominator.
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


class RegAdam(torch.optim.Optimizer):
    """
    Universally Regularized Adam (RegAdam) with Adaptive CFL Bounding.

    Computes standard Adam first and second moments, applies the elected
    spatial regularizer (Sobolev, Gaussian, DST-I, or custom callable) directly to
    the raw Adam step direction quotient (m_hat / (sqrt(v_hat) + eps)), and bounds the
    resulting spatial displacement according to the Courant-Friedrichs-Lewy (CFL) limit.
    """
    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8,
                 regularizer='sobolev', regularizer_fn=None,
                 sobolev_alpha=0.035, dsti_alpha=None, gaussian_sigma=1.5,
                 max_step_norm=0.50, spacing=None, **kwargs):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            regularizer=regularizer, regularizer_fn=regularizer_fn,
            sobolev_alpha=sobolev_alpha, dsti_alpha=dsti_alpha if dsti_alpha is not None else sobolev_alpha,
            gaussian_sigma=gaussian_sigma,
            max_step_norm=max_step_norm, spacing=spacing, **kwargs
        )
        super(RegAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            reg_mode = group.get('regularizer', 'sobolev')
            reg_fn = group.get('regularizer_fn')
            alpha = group.get('sobolev_alpha', 0.035)
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
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                raw_step = (exp_avg / bias_corr1) / denom

                # Apply elected regularization directly to the Adam step direction
                if reg_fn is not None:
                    smooth_step = reg_fn(raw_step)
                elif reg_mode == 'gaussian' or (gauss_sig is not None and gauss_sig > 0 and reg_mode != 'sobolev'):
                    from .smoothing import separable_gaussian_filter
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = separable_gaussian_filter(s, sigma=gauss_sig, spacing=spacing)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = separable_gaussian_filter(raw_step, sigma=gauss_sig, spacing=spacing)
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
                elif reg_mode == 'dsti' and alpha is not None and alpha > 0:
                    from .smoothing import apply_dsti_green_operator
                    smooth_step = apply_dsti_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha)
                elif reg_mode == 'dsti1' and alpha is not None and alpha > 0:
                    from .smoothing import apply_dsti1_green_operator
                    smooth_step = apply_dsti1_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha)
                else:
                    smooth_step = raw_step

                # Enforce Courant-Friedrichs-Lewy (CFL) step bound to prevent discrete trajectory crossover
                if max_step_norm is not None and max_step_norm > 0:
                    min_sp = min(spacing) if spacing is not None else 1.0
                    step_mag = torch.sqrt(torch.sum(smooth_step ** 2, dim=-1))
                    max_disp = float(step_mag.max().item()) / max(min_sp, 1e-4)
                    effective_step = max_disp * lr
                    if effective_step > max_step_norm:
                        scale = max_step_norm / max(effective_step, 1e-6)
                        smooth_step = smooth_step * scale

                p.sub_(smooth_step, alpha=lr)

        return loss


# Aliases for backwards compatibility and specialized naming
SobolevAdam = RegAdam
GaussianAdam = RegAdam



def get_cfl_max_norm(velocity: torch.Tensor, spacing: list) -> float:
    """
    Computes the maximum per-voxel displacement (normalized by spacing) across the velocity field.
    Useful for applying CFL (Courant-Friedrichs-Lewy) limits to spatial grid deformations.
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
    Computes the effective CFL (Courant-Friedrichs-Lewy) constrained gradient step size.
    Takes into account the physical shrink ratio at the current pyramid level.
    """
    cfl_step_val = float(kwargs.get('cfl_step', kwargs.get('grad_step', default_grad_step)))
    return float(cfl_step_val) * math.sqrt(shrink_ratio)


def check_convergence(losses, window_size: int = 10, slope_threshold: float = 1e-8) -> bool:
    """
    Checks if optimization loss has converged over a sliding window.
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
    return slope >= -slope_threshold
