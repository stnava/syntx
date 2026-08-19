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


class SobolevAdam(torch.optim.Optimizer):
    """
    Riemannian Sobolev-preconditioned Adam optimizer for diffeomorphic TVF and SyN.
    Applies Sobolev Green operator (I - alpha Delta)^-s directly to the Adam step
    direction, preserving spatial smoothness across adaptive momentum updates.
    """
    def __init__(self, params, lr=0.80, betas=(0.9, 0.999), eps=1e-8, sobolev_alpha=0.08, spacing=None, regularizer_fn=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, sobolev_alpha=sobolev_alpha, spacing=spacing, regularizer_fn=regularizer_fn)
        super(SobolevAdam, self).__init__(params, defaults)

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
            alpha = group['sobolev_alpha']
            spacing = group['spacing']
            reg_fn = group.get('regularizer_fn')

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

                # Raw point-wise step direction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                raw_step = (exp_avg / bias_corr1) / denom

                # Apply Sobolev smoothing directly to the step direction
                if reg_fn is not None:
                    smooth_step = reg_fn(raw_step)
                elif alpha is not None and alpha > 0:
                    from .smoothing import apply_sobolev_green_operator
                    if raw_step.ndim in (5, 6) and raw_step.shape[1] == 1:
                        s = raw_step.squeeze(1)
                        smooth_s = apply_sobolev_green_operator(s, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                        smooth_step = smooth_s.unsqueeze(1)
                    elif raw_step.ndim in (4, 5):
                        smooth_step = apply_sobolev_green_operator(raw_step, fluid_sigma=alpha, alpha=alpha, spacing=spacing)
                    else:
                        smooth_step = raw_step
                else:
                    smooth_step = raw_step

                p.sub_(smooth_step, alpha=lr)

        return loss


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
