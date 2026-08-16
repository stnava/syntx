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
