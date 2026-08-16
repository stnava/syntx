import torch

def normalize_tensor(
    tensor: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8,
    p_min: float = 1.0,
    p_max: float = 99.0,
    dim=None,
    keepdim: bool = True
) -> torch.Tensor:
    """
    Normalizes input PyTorch tensor using specified strategy.

    Args:
        tensor: Input PyTorch tensor (any spatial dimension).
        method: Normalization strategy:
            - 'minmax': Rescales values linearly to [0, 1].
            - 'zscore': Subtracts mean and divides by standard deviation (zero-mean, unit-variance).
            - 'robust' / 'percentile': Rescales between p_min and p_max percentiles and clamps to [0, 1].
            - 'l2' / 'unit_norm': Scales tensor by its L2 norm.
            - 'l1' / 'unit_sum': Scales tensor by its L1 norm.
            - 'sigmoid': Applies logistic sigmoid transformation.
        eps: Numerical stability floor to prevent division by zero (default: 1e-8).
        p_min: Lower percentile threshold for 'robust' scaling (default: 1.0).
        p_max: Upper percentile threshold for 'robust' scaling (default: 99.0).
        dim: Dimension(s) over which to compute statistics. If None, computes globally over all elements.
        keepdim: Retain reduced dimensions when dim is specified.

    Returns:
        Normalized PyTorch tensor with same shape and dtype.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    method = method.lower().strip()

    if method in ('minmax', '01'):
        if dim is None:
            t_min = tensor.min()
            t_max = tensor.max()
        else:
            t_min = tensor.amin(dim=dim, keepdim=keepdim)
            t_max = tensor.amax(dim=dim, keepdim=keepdim)
        return (tensor - t_min) / (t_max - t_min + eps)

    elif method in ('zscore', 'standard'):
        if dim is None:
            t_mean = tensor.mean()
            t_std = tensor.std(unbiased=False)
        else:
            t_mean = tensor.mean(dim=dim, keepdim=keepdim)
            t_std = tensor.std(dim=dim, keepdim=keepdim, unbiased=False)
        return (tensor - t_mean) / (t_std + eps)

    elif method in ('robust', 'percentile'):
        if dim is None:
            q_min = torch.quantile(tensor.float(), p_min / 100.0).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0).to(tensor.dtype)
        else:
            q_min = torch.quantile(tensor.float(), p_min / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
            q_max = torch.quantile(tensor.float(), p_max / 100.0, dim=dim, keepdim=keepdim).to(tensor.dtype)
        res = (tensor - q_min) / (q_max - q_min + eps)
        return torch.clamp(res, 0.0, 1.0)

    elif method in ('l2', 'unit_norm'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=2)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('l1', 'unit_sum'):
        if dim is None:
            norm = torch.linalg.vector_norm(tensor, ord=1)
        else:
            norm = torch.linalg.vector_norm(tensor, ord=1, dim=dim, keepdim=keepdim)
        return tensor / (norm + eps)

    elif method in ('sigmoid', 'logistic'):
        return torch.sigmoid(tensor)

    else:
        raise ValueError(f"Unknown normalization method '{method}'. Options: 'minmax', 'zscore', 'robust', 'l2', 'l1', 'sigmoid'.")
