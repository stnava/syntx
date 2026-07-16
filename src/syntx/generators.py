import torch
import torch.nn.functional as F
import numpy as np
import ants
import contextlib
from .syn import separable_gaussian_filter

@contextlib.contextmanager
def temp_seed(seed):
    """
    Context manager to temporarily set random seeds for reproducibility.
    """
    if seed is None:
        yield
        return
    state_torch = torch.random.get_rng_state()
    state_np = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state_torch)
        np.random.set_state(state_np)


class CrossProductGenerator:
    """
    2D Generative Cross-Product Space of Intensity and Shape Changes.
    Generates 2D image pairs under different intensity and shape transformations,
    returning the ground truth displacement field and its physical L2 norm magnitude.
    """
    def __init__(self, base_image=None, spacing=None, direction=None, device='cpu'):
        """
        Initialize the generator.
        
        Args:
            base_image: PyTorch Tensor of shape (1, 1, H, W), (H, W), or ANTsImage.
                        If None, a default 2D geometric phantom is generated.
            spacing: Tuple (spacing_x, spacing_y) for physical coordinate mapping.
                     If base_image is ANTsImage, its spacing is used unless overridden.
            direction: 2x2 matrix for physical direction.
                       If base_image is ANTsImage, its direction is used unless overridden.
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.base_origin = (0.0, 0.0)
        
        # 1. Parse base_image
        if base_image is None:
            base_image = self._get_default_phantom()
            
        self.spacing = spacing
        self.direction = direction
        
        if isinstance(base_image, ants.ANTsImage):
            if self.spacing is None:
                self.spacing = base_image.spacing
            if self.direction is None:
                self.direction = base_image.direction
            self.base_origin = base_image.origin
            img_np = base_image.numpy()
            self.base_tensor = torch.tensor(img_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        else:
            if not isinstance(base_image, torch.Tensor):
                base_image = torch.tensor(base_image, dtype=torch.float32)
            
            if base_image.ndim == 2:
                self.base_tensor = base_image.unsqueeze(0).unsqueeze(0).to(self.device)
            elif base_image.ndim == 3:
                self.base_tensor = base_image.unsqueeze(0).to(self.device)
            elif base_image.ndim == 4:
                self.base_tensor = base_image.to(self.device)
            else:
                raise ValueError("base_image tensor must have 2, 3, or 4 dimensions")
                
            if self.spacing is None:
                self.spacing = (1.0, 1.0)
            if self.direction is None:
                self.direction = [[1.0, 0.0], [0.0, 1.0]]
        
        # Ensure direction is a 2x2 numpy array
        if isinstance(self.direction, torch.Tensor):
            self.direction = self.direction.cpu().numpy()
        self.direction = np.array(self.direction)
        
        # Normalize base tensor to [0, 1] to avoid instability with contrast mapping
        t_min = self.base_tensor.min()
        t_max = self.base_tensor.max()
        if t_max > t_min:
            self.base_tensor = (self.base_tensor - t_min) / (t_max - t_min)
        else:
            self.base_tensor = torch.zeros_like(self.base_tensor)

    @property
    def intensity_types(self):
        return ['noise', 'bias', 'inhomogeneity', 'modality', 'step', 'missing']

    @property
    def shape_types(self):
        return ['translation', 'rotation', 'affine', 'deformation']

    def _get_default_phantom(self):
        """
        Creates a default 2D geometric phantom with a circle and an inner circle.
        """
        vol = np.zeros((64, 64), dtype=np.float32)
        y, x = np.ogrid[:64, :64]
        # Larger outer circle
        mask1 = (x - 32)**2 + (y - 32)**2 < 18**2
        # Smaller inner circle
        mask2 = (x - 24)**2 + (y - 24)**2 < 8**2
        vol[mask1] = 0.6
        vol[mask2] = 1.0
        
        img = ants.from_numpy(vol, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        img = ants.smooth_image(img, 1.0)
        return img

    def _get_identity_grid(self, H, W, device, dtype):
        """
        Generates identity grid in range [-1, 1] with format (1, H, W, 2) [x, y].
        """
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return identity

    def _apply_shape_change(self, img, shape_type, seed=None, magnitude_level='small'):
        """
        Applies shape changes to the image and generates the displacement field.
        The transformation magnitudes are bounded to maintain >= 80% spatial overlap.
        """
        H, W = img.shape[-2:]
        device = img.device
        dtype = img.dtype
        identity = self._get_identity_grid(H, W, device, dtype)
        
        if isinstance(magnitude_level, (int, float)):
            mult = float(magnitude_level)
        else:
            mult = 1.0
            if magnitude_level == 'medium':
                mult = 2.5
            elif magnitude_level == 'large':
                mult = 5.0
        
        with temp_seed(seed):
            if shape_type is None:
                u_norm = torch.zeros_like(identity)
                
            elif shape_type == 'translation':
                # Bounded translation to keep overlap >= 80%
                tx = (torch.rand(1, device=device, dtype=dtype) * 0.10 * mult - 0.05 * mult).item()
                ty = (torch.rand(1, device=device, dtype=dtype) * 0.10 * mult - 0.05 * mult).item()
                
                u_norm = torch.zeros_like(identity)
                u_norm[..., 0] = tx
                u_norm[..., 1] = ty
                
            elif shape_type == 'rotation':
                # Bounded rotation (theta in [-0.12, 0.12] rad / ~7 degrees)
                theta = (torch.rand(1, device=device, dtype=dtype) * 0.24 * mult - 0.12 * mult).item()
                
                grid_x = identity[..., 0]
                grid_y = identity[..., 1]
                
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                rotated_x = grid_x * cos_t - grid_y * sin_t
                rotated_y = grid_x * sin_t + grid_y * cos_t
                
                u_norm = torch.zeros_like(identity)
                u_norm[..., 0] = rotated_x - grid_x
                u_norm[..., 1] = rotated_y - grid_y
                
            elif shape_type == 'affine':
                # Combined affine transform with bounded parameters
                sx = (torch.rand(1, device=device, dtype=dtype) * 0.08 * mult + 1.0 - 0.04 * mult).item()
                sy = (torch.rand(1, device=device, dtype=dtype) * 0.08 * mult + 1.0 - 0.04 * mult).item()
                hx = (torch.rand(1, device=device, dtype=dtype) * 0.06 * mult - 0.03 * mult).item()
                hy = (torch.rand(1, device=device, dtype=dtype) * 0.06 * mult - 0.03 * mult).item()
                tx = (torch.rand(1, device=device, dtype=dtype) * 0.06 * mult - 0.03 * mult).item()
                ty = (torch.rand(1, device=device, dtype=dtype) * 0.06 * mult - 0.03 * mult).item()
                
                grid_x = identity[..., 0]
                grid_y = identity[..., 1]
                
                new_x = sx * grid_x + hx * grid_y + tx
                new_y = hy * grid_x + sy * grid_y + ty
                
                u_norm = torch.zeros_like(identity)
                u_norm[..., 0] = new_x - grid_x
                u_norm[..., 1] = new_y - grid_y
                
            elif shape_type == 'deformation':
                # Smooth non-rigid deformation
                # Bounded grid random values
                low_res_disp = torch.randn(1, 2, 5, 5, device=device, dtype=dtype) * (0.035 * mult)
                disp = F.interpolate(low_res_disp, size=(H, W), mode='bilinear', align_corners=True)
                u_norm = disp.permute(0, 2, 3, 1) # (1, H, W, 2)
                u_norm = separable_gaussian_filter(u_norm, sigma=4.0)
                
            else:
                raise ValueError(f"Unknown shape_type: {shape_type}")
        
        # Warp using grid = identity + u_norm
        grid = identity + u_norm
        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_img, u_norm

    def _apply_intensity_change(self, img, intensity_type, seed=None):
        """
        Applies one of the 6 intensity changes to the image.
        """
        if intensity_type is None:
            return img
            
        with temp_seed(seed):
            if intensity_type == 'noise':
                # Additive Gaussian or Rician noise. Let's do Rician.
                sigma = 0.04
                n1 = torch.randn_like(img) * sigma
                n2 = torch.randn_like(img) * sigma
                return torch.sqrt((img + n1)**2 + n2**2)
                
            elif intensity_type == 'bias':
                # Multiplicative bias field (multiplicative low-frequency spatial inhomogeneity)
                H, W = img.shape[-2:]
                low_res = torch.randn(1, 1, 4, 4, device=img.device, dtype=img.dtype) * 0.12
                bias = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=True)
                bias = torch.exp(bias)
                return img * bias
                
            elif intensity_type == 'inhomogeneity':
                # Local Gaussian blob (hyper/hypo-intense)
                H, W = img.shape[-2:]
                y = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
                x = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                
                cx = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.6 - 0.3).item()
                cy = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.6 - 0.3).item()
                strength = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.3 + 0.15).item()
                if torch.rand(1, device=img.device).item() > 0.5:
                    strength = -strength
                sigma = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.08 + 0.12).item()
                
                dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
                blob = strength * torch.exp(-dist_sq / (2 * sigma**2))
                blob = blob.unsqueeze(0).unsqueeze(0)
                return torch.clamp(img + blob, min=0.0)
                
            elif intensity_type == 'modality':
                # Shuffle the three primary levels: 0.0 -> 1.0, 0.6 -> 0.0, 1.0 -> 0.6
                # using a piecewise continuous interpolation to handle smoothed edges
                new_img = torch.where(img < 0.6, 
                                      1.0 - (1.0 / 0.6) * img, 
                                      0.0 + (0.6 / 0.4) * (img - 0.6))
                return torch.clamp(new_img, 0.0, 1.0)
                
            elif intensity_type == 'step':
                # Quantized intensity step mapping
                num_bins = 4
                return torch.round(img * (num_bins - 1)) / (num_bins - 1)
                
            elif intensity_type == 'missing':
                # Local masked region set to 0
                H, W = img.shape[-2:]
                y = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
                x = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                
                cx = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.5 - 0.25).item()
                cy = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.5 - 0.25).item()
                mask_size = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.08 + 0.15).item() # size bounded to ~15-23%
                
                mask = (torch.abs(grid_x - cx) < mask_size / 2) & (torch.abs(grid_y - cy) < mask_size / 2)
                mask = mask.unsqueeze(0).unsqueeze(0)
                return img * (~mask)
                
            else:
                raise ValueError(f"Unknown intensity_type: {intensity_type}")

    def compute_physical_l2_norm(self, u_norm):
        """
        Computes the physical L2 norm of the normalized displacement field.
        
        Formula:
          u_vox = u_norm * (N - 1) / 2
          u_phys = D * (u_vox * spacing)
          L2 = sqrt( delta_V * sum( ||u_phys(x)||^2 ) )
        """
        if u_norm.ndim == 4:
            # Squeeze batch dimension to (H, W, 2)
            u_norm_sq = u_norm.squeeze(0)
        else:
            u_norm_sq = u_norm
            
        H, W, _ = u_norm_sq.shape
        device = u_norm.device
        dtype = u_norm.dtype
        
        # N corresponds to W (width for x displacement) and H (height for y displacement)
        # because u_norm has [x, y] coordinates in the last channel
        N = torch.tensor([W, H], dtype=dtype, device=device)
        u_vox = u_norm_sq * (N - 1) / 2.0
        
        spacing_t = torch.tensor(self.spacing, dtype=dtype, device=device)
        direction_t = torch.tensor(self.direction, dtype=dtype, device=device)
        
        u_vox_scaled = u_vox * spacing_t
        u_phys = torch.matmul(u_vox_scaled, direction_t.t())
        
        delta_V = float(np.prod(self.spacing))
        sum_sq = torch.sum(u_phys ** 2)
        norm = torch.sqrt(delta_V * sum_sq)
        
        return norm.item()

    def generate(self, intensity_type, shape_type, seed=None, magnitude_level='small'):
        """
        Generates an image pair.
        
        Returns:
            fixed_image: (1, 1, H, W) clean base image tensor
            moving_image: (1, 1, H, W) warped image with intensity change applied
            displacement_field: (1, H, W, 2) normalized ground-truth displacement field
            magnitude: float, physical L2 norm of the displacement field
        """
        if intensity_type not in self.intensity_types and intensity_type is not None:
            raise ValueError(f"Unknown intensity_type: {intensity_type}")
        if shape_type not in self.shape_types and shape_type is not None:
            raise ValueError(f"Unknown shape_type: {shape_type}")
            
        # Fixed image is the clean base image
        fixed_image = self.base_tensor.clone()
        
        # Apply shape change to get moving image and displacement field
        moving_warped, displacement_field = self._apply_shape_change(fixed_image, shape_type, seed=seed, magnitude_level=magnitude_level)
        
        # Apply intensity change on the warped image
        moving_image = self._apply_intensity_change(moving_warped, intensity_type, seed=seed)
        
        # Compute ground truth physical L2 norm magnitude
        magnitude = self.compute_physical_l2_norm(displacement_field)
        
        return fixed_image, moving_image, displacement_field, magnitude

    def to_ants_image(self, tensor_image):
        """
        Helper to convert a torch tensor of shape (1, 1, H, W) to an ANTsImage.
        """
        np_img = tensor_image.detach().cpu().squeeze(0).squeeze(0).numpy()
        return ants.from_numpy(
            np_img,
            origin=self.base_origin,
            spacing=self.spacing,
            direction=self.direction
        )
