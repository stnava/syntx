import math
import numpy as np
import torch
import torch.nn.functional as F

from syntx.features import FeatureSpaceLoss
from syntx.syn import local_ncc_loss_nd, mattes_mi_loss_nd

def to_torch(x) -> torch.Tensor:
    """Converts inputs (ANTSImage, PyTorch/JAX/NumPy arrays) to PyTorch tensor."""
    if hasattr(x, 'numpy'):  # e.g., ants.ANTsImage
        x = x.numpy()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x.detach()
    # Check for JAX array
    type_name = type(x).__name__
    if type_name in ('Array', 'DeviceArray', 'ArrayImpl') or hasattr(x, '__jax_array__'):
        return torch.from_numpy(np.asarray(x))
    try:
        return torch.as_tensor(x)
    except Exception:
        return torch.from_numpy(np.asarray(x))

def standardize_tensor(t: torch.Tensor):
    """Squeezes batch and channel dimensions to return (H, W) or (D, H, W)."""
    sh = list(t.shape)
    while len(sh) > 3 and sh[0] == 1:
        t = t.squeeze(0)
        sh = list(t.shape)
    while len(sh) > 3 and sh[-1] == 1:
        t = t.squeeze(-1)
        sh = list(t.shape)
    if len(sh) == 4 and sh[0] == 1:
        t = t.squeeze(0)
        sh = list(t.shape)
    if len(sh) == 4 and sh[-1] == 1:
        t = t.squeeze(-1)
        sh = list(t.shape)
    ndim = t.ndim
    if ndim not in (2, 3):
        t = t.squeeze()
        ndim = t.ndim
    return t, ndim

def compute_histograms(a_np, b_np, bins=32):
    """Computes marginal and joint entropies for NMI and Joint Entropy."""
    hist_2d, _, _ = np.histogram2d(a_np.flatten(), b_np.flatten(), bins=bins)
    pxy = hist_2d / (np.sum(hist_2d) + 1e-8)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    def entropy(p):
        p_nz = p[p > 0]
        return -np.sum(p_nz * np.log2(p_nz))
        
    h_a = entropy(px)
    h_b = entropy(py)
    h_ab = entropy(pxy)
    
    return h_a, h_b, h_ab

def ssim_torch(img1, img2, window_size=11, size_average=True):
    """PyTorch implementation of 2D/3D Structural Similarity (SSIM)."""
    dim = img1.ndim - 2
    if dim == 2:
        pool_fn = F.avg_pool2d
    elif dim == 3:
        pool_fn = F.avg_pool3d
    else:
        raise ValueError("SSIM only supports 2D/3D inputs")
    
    pad = window_size // 2
    
    mu1 = pool_fn(img1, kernel_size=window_size, stride=1, padding=pad)
    mu2 = pool_fn(img2, kernel_size=window_size, stride=1, padding=pad)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = pool_fn(img1 * img1, kernel_size=window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = pool_fn(img2 * img2, kernel_size=window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = pool_fn(img1 * img2, kernel_size=window_size, stride=1, padding=pad) - mu1_mu2
    
    L = max(float((img1.max() - img1.min()).item()), 1e-8)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def ms_ssim_torch(img1, img2, window_size=11):
    """PyTorch implementation of Multi-Scale SSIM supporting 2D/3D."""
    dim = img1.ndim - 2
    if dim == 2:
        pool_fn = F.avg_pool2d
        downsample_fn = lambda x: F.avg_pool2d(x, kernel_size=2, stride=2)
    elif dim == 3:
        pool_fn = F.avg_pool3d
        downsample_fn = lambda x: F.avg_pool3d(x, kernel_size=2, stride=2)
    else:
        raise ValueError("MS-SSIM only supports 2D/3D inputs")
        
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    min_size = min(img1.shape[2:])
    
    w_size = window_size
    if min_size < w_size:
        w_size = min_size
        if w_size % 2 == 0:
            w_size = max(1, w_size - 1)
            
    max_scales = 1
    current_size = min_size
    while current_size >= w_size and max_scales <= len(weights):
        current_size //= 2
        max_scales += 1
    max_scales = max(1, max_scales - 1)
    
    active_weights = weights[:max_scales]
    weight_sum = sum(active_weights)
    active_weights = [w / weight_sum for w in active_weights]
    
    msssim = 1.0
    im1 = img1
    im2 = img2
    
    pad = w_size // 2
    
    for i, w in enumerate(active_weights):
        mu1 = pool_fn(im1, kernel_size=w_size, stride=1, padding=pad)
        mu2 = pool_fn(im2, kernel_size=w_size, stride=1, padding=pad)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = pool_fn(im1 * im1, kernel_size=w_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = pool_fn(im2 * im2, kernel_size=w_size, stride=1, padding=pad) - mu2_sq
        sigma12 = pool_fn(im1 * im2, kernel_size=w_size, stride=1, padding=pad) - mu1_mu2
        
        L = max(float((im1.max() - im1.min()).item()), 1e-8)
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        l_val = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1 + 1e-8)
        cs_val = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + 1e-8)
        
        l_val = torch.clamp(l_val, min=0.0)
        cs_val = torch.clamp(cs_val, min=0.0)
        
        if i == len(active_weights) - 1:
            val = (l_val ** w) * (cs_val ** w)
        else:
            val = cs_val ** w
            
        msssim = msssim * val.mean()
        
        if i < len(active_weights) - 1:
            im1 = downsample_fn(im1)
            im2 = downsample_fn(im2)
            
    return msssim

def compute_gradients_torch(img):
    """Computes spatial gradients of an N-D tensor."""
    dim = img.ndim - 2
    if dim == 2:
        img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
        dy = (img_pad[:, :, 2:, 1:-1] - img_pad[:, :, :-2, 1:-1]) / 2.0
        dx = (img_pad[:, :, 1:-1, 2:] - img_pad[:, :, 1:-1, :-2]) / 2.0
        return torch.cat([dy, dx], dim=1)
    elif dim == 3:
        img_pad = F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate')
        dz = (img_pad[:, :, 2:, 1:-1, 1:-1] - img_pad[:, :, :-2, 1:-1, 1:-1]) / 2.0
        dy = (img_pad[:, :, 1:-1, 2:, 1:-1] - img_pad[:, :, 1:-1, :-2, 1:-1]) / 2.0
        dx = (img_pad[:, :, 1:-1, 1:-1, 2:] - img_pad[:, :, 1:-1, 1:-1, :-2]) / 2.0
        return torch.cat([dz, dy, dx], dim=1)
    else:
        raise ValueError("Gradients only support 2D/3D inputs")

def compute_reconstructed_loss(extractor, a_3d, b_3d, loss_type='l1'):
    """Calculates triplanar reconstructed features for 2D networks applied to 3D."""
    D, H, W = a_3d.shape[2:]
    B = a_3d.shape[0]
    
    def get_vols(x):
        slices_ax = []
        for z in range(1, D - 1):
            if extractor.in_channels == 3:
                slices_ax.append(x[:, 0, z-1:z+2])
            else:
                slices_ax.append(x[:, 0:1, z:z+1])
        batch_ax = extractor.normalize(torch.cat(slices_ax, dim=0))
        
        slices_co = []
        for y in range(1, H - 1):
            if extractor.in_channels == 3:
                slices_co.append(x[:, 0, :, y-1:y+2, :].movedim(2, 1))
            else:
                slices_co.append(x[:, 0:1, :, y:y+1, :].movedim(2, 1))
        batch_co = extractor.normalize(torch.cat(slices_co, dim=0))
        
        slices_sa = []
        for xi in range(1, W - 1):
            if extractor.in_channels == 3:
                slices_sa.append(x[:, 0, :, :, xi-1:xi+2].movedim(3, 1))
            else:
                slices_sa.append(x[:, 0:1, :, :, xi:xi+1].movedim(3, 1))
        batch_sa = extractor.normalize(torch.cat(slices_sa, dim=0))
        
        feat_ax = extractor.extract(batch_ax)[-1]
        feat_co = extractor.extract(batch_co)[-1]
        feat_sa = extractor.extract(batch_sa)[-1]
        
        vol_ax = feat_ax.view(D-2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4)
        vol_co = feat_co.view(H-2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4)
        vol_sa = feat_sa.view(W-2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0)
        
        return vol_ax, vol_co, vol_sa
        
    vol_a_ax, vol_a_co, vol_a_sa = get_vols(a_3d)
    vol_b_ax, vol_b_co, vol_b_sa = get_vols(b_3d)
    
    if loss_type == 'l1':
        loss_ax = torch.mean(torch.abs(vol_a_ax - vol_b_ax))
        loss_co = torch.mean(torch.abs(vol_a_co - vol_b_co))
        loss_sa = torch.mean(torch.abs(vol_a_sa - vol_b_sa))
    elif loss_type == 'l2':
        loss_ax = torch.mean((vol_a_ax - vol_b_ax) ** 2)
        loss_co = torch.mean((vol_a_co - vol_b_co) ** 2)
        loss_sa = torch.mean((vol_a_sa - vol_b_sa) ** 2)
    elif loss_type == 'cos':
        loss_ax = 1.0 - torch.mean(F.cosine_similarity(vol_a_ax, vol_b_ax, dim=1))
        loss_co = 1.0 - torch.mean(F.cosine_similarity(vol_a_co, vol_b_co, dim=1))
        loss_sa = 1.0 - torch.mean(F.cosine_similarity(vol_a_sa, vol_b_sa, dim=1))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    return float((loss_ax + loss_co + loss_sa).item() / 3.0)

def compute_direct_loss(extractor, a_nd, b_nd, loss_type='l1'):
    """Calculates feature loss directly for 2D inputs or native 3D extractors."""
    if extractor.in_channels == 3 and a_nd.shape[1] == 1:
        a_rgb = a_nd.repeat(1, 3, 1, 1)
        b_rgb = b_nd.repeat(1, 3, 1, 1)
    else:
        a_rgb = a_nd
        b_rgb = b_nd
        
    feat_a = extractor.extract(extractor.normalize(a_rgb))[-1]
    feat_b = extractor.extract(extractor.normalize(b_rgb))[-1]
    
    if loss_type == 'l1':
        loss = torch.mean(torch.abs(feat_a - feat_b))
    elif loss_type == 'l2':
        loss = torch.mean((feat_a - feat_b) ** 2)
    elif loss_type == 'cos':
        loss = 1.0 - torch.mean(F.cosine_similarity(feat_a, feat_b, dim=1))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    return float(loss.item())

_ext_cache = {}

def get_cached_extractor(model_name: str, dim: int, layer: int, device):
    """Retrieves or creates and caches the requested feature extractor module."""
    key = (model_name, dim, layer, str(device))
    if key not in _ext_cache:
        if model_name == 'vgg':
            from syntx.features import VGG19Extractor
            ext = VGG19Extractor(feature_layers=[layer])
        elif model_name == 'dino':
            from syntx.features import DINOv2Extractor
            ext = DINOv2Extractor(version='vits14', feature_layers=[layer])
        elif model_name == 'resnet':
            from syntx.features import ResNet10Extractor
            ext = ResNet10Extractor(dim=dim, feature_layers=[layer])
        elif model_name == 'swin':
            from syntx.features import SwinUNETRExtractor
            ext = SwinUNETRExtractor(feature_layers=[layer])
        else:
            raise ValueError(f"Unknown extractor: {model_name}")
        ext = ext.to(device)
        ext.eval()
        _ext_cache[key] = ext
    return _ext_cache[key]

def deep_feature_loss(a_unsq, b_unsq, extractor, l, mtype, dim):
    """Orchestrates LNCC, L1, L2, and Cosine similarity computation on deep features."""
    if mtype == 'lncc':
        if dim == 3:
            loss_fn = FeatureSpaceLoss(extractor=extractor, mode='lncc_3d')
        else:
            loss_fn = FeatureSpaceLoss(extractor=extractor, mode='lncc')
        loss = loss_fn(a_unsq, b_unsq)
        return float(loss.item())
    
    if dim == 3 and not extractor.is_3d:
        return compute_reconstructed_loss(extractor, a_unsq, b_unsq, loss_type=mtype)
    else:
        return compute_direct_loss(extractor, a_unsq, b_unsq, loss_type=mtype)

def image_compare(a, b, metricname: str, **kwargs) -> float:
    """
    Computes image similarity or distance metric between images `a` and `b`.
    Standardized such that a lower score indicates better similarity.
    
    a, b: ANTsImage, PyTorch tensors, JAX arrays, or NumPy arrays.
    metricname: String identifier for the configuration to run.
    """
    a_t = to_torch(a)
    b_t = to_torch(b)
    
    # Check shape compatibility
    if a_t.shape != b_t.shape:
        a_std, ndim_a = standardize_tensor(a_t)
        b_std, ndim_b = standardize_tensor(b_t)
        if a_std.shape != b_std.shape:
            raise ValueError(f"Shape mismatch: {a_t.shape} vs {b_t.shape}")
        a_t, b_t = a_std, b_std
        ndim = ndim_a
    else:
        a_t, ndim = standardize_tensor(a_t)
        b_t, _ = standardize_tensor(b_t)
        
    device = kwargs.get('device', a_t.device)
    a_t = a_t.to(device).float()
    b_t = b_t.to(device).float()
    
    a_unsq = a_t.unsqueeze(0).unsqueeze(0)
    b_unsq = b_t.unsqueeze(0).unsqueeze(0)
    
    metricname = metricname.lower().strip()
    
    # Classical (18)
    if metricname == 'mse':
        return float(torch.mean((a_t - b_t) ** 2).item())
    elif metricname == 'mae':
        return float(torch.mean(torch.abs(a_t - b_t)).item())
    elif metricname == 'rmse':
        return float(torch.sqrt(torch.mean((a_t - b_t) ** 2)).item())
    elif metricname == 'psnr':
        r = float(torch.sqrt(torch.mean((a_t - b_t) ** 2)).item())
        if r < 1e-10:
            val = 100.0
        else:
            max_val = kwargs.get('max_val', None)
            if max_val is None:
                max_val = max(float(a_t.max().item()), float(b_t.max().item()))
            if max_val < 1e-10:
                max_val = 1.0
            val = 20.0 * math.log10(max_val / r)
        return -val
    elif metricname == 'ncc':
        a_mean = torch.mean(a_t)
        b_mean = torch.mean(b_t)
        a_diff = a_t - a_mean
        b_diff = b_t - b_mean
        num = torch.sum(a_diff * b_diff)
        den = torch.sqrt(torch.sum(a_diff ** 2) * torch.sum(b_diff ** 2) + 1e-8)
        val = float((num / den).item())
        return 1.0 - val
    elif metricname in ('nmi', 'joint_entropy'):
        a_np = a_t.detach().cpu().numpy()
        b_np = b_t.detach().cpu().numpy()
        h_a, h_b, h_ab = compute_histograms(a_np, b_np, bins=kwargs.get('bins', 32))
        if metricname == 'joint_entropy':
            return float(h_ab)
        else:
            val = (h_a + h_b) / (h_ab + 1e-8)
            return -float(val)
    elif metricname == 'lncc' or metricname.startswith('lncc_w'):
        if metricname == 'lncc':
            w_size = 5
        else:
            try:
                w_size = int(metricname.split('_w')[1])
            except Exception:
                raise ValueError(f"Invalid LNCC metric name: {metricname}")
        return float(local_ncc_loss_nd(a_unsq, b_unsq, window_size=w_size).item())
    elif metricname in ('mattes_mi', 'mattes') or metricname.startswith('mmi_b'):
        if metricname in ('mattes_mi', 'mattes'):
            bins = 32
        else:
            try:
                bins = int(metricname.split('_b')[1])
            except Exception:
                raise ValueError(f"Invalid MMI metric name: {metricname}")
        return float(mattes_mi_loss_nd(a_unsq, b_unsq, num_bins=bins).item())
    elif metricname == 'ssim':
        val = ssim_torch(a_unsq, b_unsq)
        return 1.0 - float(val.item())
        
    # Gradient/Spatial (6)
    elif metricname == 'gradient_mse':
        grad_a = compute_gradients_torch(a_unsq)
        grad_b = compute_gradients_torch(b_unsq)
        return float(torch.mean((grad_a - grad_b) ** 2).item())
    elif metricname == 'gradient_correlation':
        grad_a = compute_gradients_torch(a_unsq).flatten()
        grad_b = compute_gradients_torch(b_unsq).flatten()
        mean_a = torch.mean(grad_a)
        mean_b = torch.mean(grad_b)
        diff_a = grad_a - mean_a
        diff_b = grad_b - mean_b
        num = torch.sum(diff_a * diff_b)
        den = torch.sqrt(torch.sum(diff_a ** 2) * torch.sum(diff_b ** 2) + 1e-8)
        corr = float((num / den).item())
        return 1.0 - corr
    elif metricname.startswith('ngf_e'):
        suffix = metricname.split('_e')[1]
        if suffix == '01':
            eta = 0.1
        elif suffix == '1':
            eta = 1.0
        elif suffix == '10':
            eta = 10.0
        else:
            try:
                eta = float(suffix)
            except Exception:
                raise ValueError(f"Invalid NGF metric name: {metricname}")
        grad_a = compute_gradients_torch(a_unsq)
        grad_b = compute_gradients_torch(b_unsq)
        dot = torch.sum(grad_a * grad_b, dim=1, keepdim=True)
        norm_a_sq = torch.sum(grad_a ** 2, dim=1, keepdim=True)
        norm_b_sq = torch.sum(grad_b ** 2, dim=1, keepdim=True)
        sim = (dot ** 2) / ((norm_a_sq + eta ** 2) * (norm_b_sq + eta ** 2) + 1e-8)
        return 1.0 - float(sim.mean().item())
    elif metricname == 'ms_ssim':
        val = ms_ssim_torch(a_unsq, b_unsq)
        return 1.0 - float(val.item())
        
    # Deep Feature configurations
    else:
        parts = metricname.split('_')
        if len(parts) == 3:
            prefix, layer_str, mtype = parts
            try:
                layer = int(layer_str)
            except ValueError:
                raise ValueError(f"Unknown metric name: {metricname}")
                
            if prefix in ('vgg', 'dino', 'resnet', 'swin'):
                extractor = get_cached_extractor(prefix, ndim, layer, device)
                return deep_feature_loss(a_unsq, b_unsq, extractor, layer, mtype, ndim)
                
        raise ValueError(f"Unknown metric name: {metricname}")
