"""
syntx.generators — synthetic 2-D test pairs and small benchmark datasets.

- ``CrossProductGenerator``: from one 2-D base image, makes a (fixed, moving) pair by applying
  one of 4 random spatial changes (translation, rotation, affine, smooth deformation; scaled by
  a magnitude multiplier) and then one of 6 intensity changes (Rician noise, smooth bias,
  local blob, non-monotone "modality" remap, quantisation, missing square). Returns the
  sampling displacement used and its integrated L2 size.
- ``benchmark_data``: loads or builds a named fixed / moving / label set ('r16_r64', 'c',
  'ellipse', 'mbhard'), caching phantom files on disk.
- ``temp_seed``: context manager that seeds and then restores the CPU torch / NumPy RNGs.
"""

import os
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .syn import separable_gaussian_filter


@contextlib.contextmanager
def temp_seed(seed: int = None):
    """
    Seed torch and NumPy for the duration of a ``with`` block, then restore their states.

    ``torch.manual_seed`` also seeds CUDA / MPS generators, but only the CPU torch state and
    the NumPy global state are saved and restored afterwards.

    Parameters
    ----------
    seed : int, optional
        Seed. None: do nothing (RNG state is not touched).

    Yields
    ------
    None
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
    Make synthetic 2-D (fixed, moving) pairs: one spatial change times one intensity change.

    The base image is min-max scaled to [0, 1] (a constant image becomes all zeros) and kept
    as a ``(1, 1, H, W)`` tensor. ``generate`` warps it with a random displacement and then
    alters the intensities of the warped copy.

    Parameters
    ----------
    base_image : torch.Tensor, np.ndarray or ants.ANTsImage, optional
        2-D image. Tensors / arrays may be (H, W), (C, H, W) or (B, C, H, W) (other ranks
        raise ValueError). An ANTsImage is used as ``image.numpy()``, i.e. in ANTs (x, y)
        axis order, not reversed. None: a 64 x 64 phantom (disc of value 0.6, radius 18, with
        a brighter offset disc of value 1.0, radius 8, Gaussian-smoothed with sigma 1).
    spacing : tuple of float, optional
        Spacing used by ``compute_physical_l2_norm`` and ``to_ants_image``. Default: the
        ANTsImage spacing, else (1, 1).
    direction : array-like (2, 2), optional
        Default: the ANTsImage direction, else identity. Stored as a NumPy array.
    device : str, default 'cpu'
        Torch device for the base tensor and all generation.

    Attributes
    ----------
    intensity_types : list of str
        `['noise', 'bias', 'inhomogeneity', 'modality', 'step', 'missing']`.
    shape_types : list of str
        `['translation', 'rotation', 'affine', 'deformation']`.
    base_tensor : torch.Tensor
        The scaled base image, shape (B, C, H, W).
    base_origin : tuple
        ANTsImage origin, else (0, 0).

    Notes
    -----
    Displacement component 0 is along the last tensor axis (W) and is multiplied by
    ``spacing[0]``; for an ANTsImage base that axis is ANTs y, so with anisotropic spacing the
    two spacings are swapped in ``compute_physical_l2_norm``.
    """

    def __init__(self, base_image=None, spacing=None, direction=None, device='cpu'):
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

        if isinstance(self.direction, torch.Tensor):
            self.direction = self.direction.cpu().numpy()
        self.direction = np.array(self.direction)

        # Normalize base tensor to [0, 1]
        t_min = self.base_tensor.min()
        t_max = self.base_tensor.max()
        if t_max > t_min:
            self.base_tensor = (self.base_tensor - t_min) / (t_max - t_min)
        else:
            self.base_tensor = torch.zeros_like(self.base_tensor)

    @property
    def intensity_types(self) -> list:
        """Names accepted as ``intensity_type`` by ``generate`` (besides None)."""
        return ['noise', 'bias', 'inhomogeneity', 'modality', 'step', 'missing']

    @property
    def shape_types(self) -> list:
        """Names accepted as ``shape_type`` by ``generate`` (besides None)."""
        return ['translation', 'rotation', 'affine', 'deformation']

    def _get_default_phantom(self) -> ants.ANTsImage:
        """64 x 64 two-disc phantom (0.6 disc r=18 at the centre, 1.0 disc r=8 at (24, 24)),
        smoothed with sigma 1, spacing 1."""
        vol = np.zeros((64, 64), dtype=np.float32)
        y, x = np.ogrid[:64, :64]
        mask1 = (x - 32) ** 2 + (y - 32) ** 2 < 18 ** 2
        mask2 = (x - 24) ** 2 + (y - 24) ** 2 < 8 ** 2
        vol[mask1] = 0.6
        vol[mask2] = 1.0

        img = ants.from_numpy(vol, spacing=(1.0, 1.0), origin=(0.0, 0.0))
        img = ants.smooth_image(img, 1.0)
        return img

    def _get_identity_grid(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Identity ``grid_sample`` grid in [-1, 1] (align_corners=True), shape (1, H, W, 2),
        last axis (x = along W, y = along H)."""
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return identity

    def _apply_shape_change(self, img: torch.Tensor, shape_type: str, seed: int = None, magnitude_level='small'):
        """Warp ``img`` with a random displacement; return ``(warped, u_norm)``.

        ``u_norm`` (1, H, W, 2) is in normalised [-1, 1] units (align_corners=True) and is a
        pull-back offset: ``warped(p) = img(p + u_norm(p))`` (bilinear, border padding). With
        multiplier m ('small' 1, 'medium' 2.5, 'large' 5, a number as given, any other string
        1): translation uniform in +-0.05 m per axis; rotation about the grid centre by an angle
        uniform in +-0.12 m rad; affine with scales 1 +- 0.04 m, shears and translations +-0.03 m;
        deformation from a 5 x 5 N(0, (0.035 m)^2) grid, bilinear upsampling and Gaussian
        smoothing (sigma 4 voxels); None gives zero displacement. Unknown names raise
        ValueError. Random numbers are drawn on ``img.device`` under ``temp_seed(seed)``.
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
                tx = (torch.rand(1, device=device, dtype=dtype) * 0.10 * mult - 0.05 * mult).item()
                ty = (torch.rand(1, device=device, dtype=dtype) * 0.10 * mult - 0.05 * mult).item()

                u_norm = torch.zeros_like(identity)
                u_norm[..., 0] = tx
                u_norm[..., 1] = ty

            elif shape_type == 'rotation':
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
                low_res_disp = torch.randn(1, 2, 5, 5, device=device, dtype=dtype) * (0.035 * mult)
                disp = F.interpolate(low_res_disp, size=(H, W), mode='bilinear', align_corners=True)
                u_norm = disp.permute(0, 2, 3, 1)
                u_norm = separable_gaussian_filter(u_norm, sigma=4.0)

            else:
                raise ValueError(f"Unknown shape_type: {shape_type}")

        grid = identity + u_norm
        warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_img, u_norm

    def _apply_intensity_change(self, img: torch.Tensor, intensity_type: str, seed: int = None) -> torch.Tensor:
        """Return an intensity-altered copy of ``img`` (values assumed in [0, 1]).

        'noise': Rician, sqrt((I + n1)^2 + n2^2) with n1, n2 ~ N(0, 0.04^2). 'bias': multiply by
        exp of a bilinearly upsampled 4 x 4 N(0, 0.12^2) field. 'inhomogeneity': add a Gaussian
        blob (centre +-0.3, width 0.12-0.20, amplitude +-[0.15, 0.45], normalised units) and
        clamp at 0. 'modality': non-monotone remap, I < 0.6 -> 1 - I / 0.6, else
        1.5 (I - 0.6), clamped to [0, 1]. 'step': round to 4 levels. 'missing': zero a square
        of side 0.15-0.23 (normalised units) centred within +-0.25. None: unchanged. Unknown
        names raise ValueError.
        """
        if intensity_type is None:
            return img

        with temp_seed(seed):
            if intensity_type == 'noise':
                sigma = 0.04
                n1 = torch.randn_like(img) * sigma
                n2 = torch.randn_like(img) * sigma
                return torch.sqrt((img + n1) ** 2 + n2 ** 2)

            elif intensity_type == 'bias':
                H, W = img.shape[-2:]
                low_res = torch.randn(1, 1, 4, 4, device=img.device, dtype=img.dtype) * 0.12
                bias = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=True)
                bias = torch.exp(bias)
                return img * bias

            elif intensity_type == 'inhomogeneity':
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

                dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
                blob = strength * torch.exp(-dist_sq / (2 * sigma ** 2))
                blob = blob.unsqueeze(0).unsqueeze(0)
                return torch.clamp(img + blob, min=0.0)

            elif intensity_type == 'modality':
                new_img = torch.where(img < 0.6,
                                      1.0 - (1.0 / 0.6) * img,
                                      0.0 + (0.6 / 0.4) * (img - 0.6))
                return torch.clamp(new_img, 0.0, 1.0)

            elif intensity_type == 'step':
                num_bins = 4
                return torch.round(img * (num_bins - 1)) / (num_bins - 1)

            elif intensity_type == 'missing':
                H, W = img.shape[-2:]
                y = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
                x = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

                cx = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.5 - 0.25).item()
                cy = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.5 - 0.25).item()
                mask_size = (torch.rand(1, device=img.device, dtype=img.dtype) * 0.08 + 0.15).item()

                mask = (torch.abs(grid_x - cx) < mask_size / 2) & (torch.abs(grid_y - cy) < mask_size / 2)
                mask = mask.unsqueeze(0).unsqueeze(0)
                return img * (~mask)

            else:
                raise ValueError(f"Unknown intensity_type: {intensity_type}")

    def compute_physical_l2_norm(self, u_norm: torch.Tensor) -> float:
        """
        Integrated L2 size of a normalised displacement field in physical units.

        $$L_2 = \\sqrt{\\Delta V \\sum_{x} \\|u_{\\text{phys}}(x)\\|_2^2}$$

        ``u_phys = direction @ (u_norm * (N - 1) / 2 * spacing)`` with N = (W, H) and
        Delta V = prod(spacing). This is a discrete integral over the domain (units mm times
        sqrt(area)), not a mean or maximum displacement; it grows with image size.

        Parameters
        ----------
        u_norm : torch.Tensor
            Shape `(1, H, W, 2)` or `(H, W, 2)`, normalised [-1, 1] units (align_corners=True).

        Returns
        -------
        float
        """
        if u_norm.ndim == 4:
            u_norm_sq = u_norm.squeeze(0)
        else:
            u_norm_sq = u_norm

        H, W, _ = u_norm_sq.shape
        device = u_norm.device
        dtype = u_norm.dtype

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

    def generate(self, intensity_type: str, shape_type: str, seed: int = None, magnitude_level='small'):
        """
        Make one (fixed, moving) pair: warp the base image, then alter the warped intensities.

        See ``_apply_shape_change`` / ``_apply_intensity_change`` for the exact random models.

        Parameters
        ----------
        intensity_type : str or None
            'noise', 'bias', 'inhomogeneity', 'modality', 'step', 'missing', or None (no
            change). Other values raise ValueError.
        shape_type : str or None
            'translation', 'rotation', 'affine', 'deformation', or None (identity). Other values
            raise ValueError.
        seed : int, optional
            Seed applied (via ``temp_seed``) separately to the shape and the intensity draws.
            None: use the current RNG state.
        magnitude_level : str or float, default 'small'
            Spatial magnitude multiplier: 'small' 1, 'medium' 2.5, 'large' 5, or a number; any
            other string silently means 1. Does not affect the intensity change.

        Returns
        -------
        fixed_image : torch.Tensor
            Copy of the scaled base image, `(B, C, H, W)` (normally `(1, 1, H, W)`).
        moving_image : torch.Tensor
            Warped and intensity-altered image, same shape.
        displacement_field : torch.Tensor
            `(1, H, W, 2)`, normalised units, components (x along W, y along H), with
            ``moving_before_intensity(p) = fixed(p + u(p))``.
        magnitude : float
            ``compute_physical_l2_norm(displacement_field)``.
        """
        if intensity_type not in self.intensity_types and intensity_type is not None:
            raise ValueError(f"Unknown intensity_type: {intensity_type}")
        if shape_type not in self.shape_types and shape_type is not None:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        fixed_image = self.base_tensor.clone()
        moving_warped, displacement_field = self._apply_shape_change(fixed_image, shape_type, seed=seed, magnitude_level=magnitude_level)
        moving_image = self._apply_intensity_change(moving_warped, intensity_type, seed=seed)
        magnitude = self.compute_physical_l2_norm(displacement_field)

        return fixed_image, moving_image, displacement_field, magnitude

    def to_ants_image(self, tensor_image: torch.Tensor) -> ants.ANTsImage:
        """Convert a `(1, 1, H, W)` tensor to an ANTsImage with the generator's origin, spacing
        and direction (array axes used as is)."""
        np_img = tensor_image.detach().cpu().squeeze(0).squeeze(0).numpy()
        return ants.from_numpy(
            np_img,
            origin=self.base_origin,
            spacing=self.spacing,
            direction=self.direction
        )


def benchmark_data(key: str = 'r16_r64', data_dir: str = None) -> dict:
    """
    Load (or build and cache) a named fixed / moving pair with label maps.

    Parameters
    ----------
    key : str, default 'r16_r64'
        Case-insensitive.

        - 'r16_r64' (also '2d', 'r16', 'r64'): ANTs sample slices r16 (fixed) and r64
          (moving); labels from 3-class Otsu (``fixed_labels`` / ``moving_labels``: 'otsu',
          'class2' = class 2 only, 'class2_3' = classes 2-3). Nothing is written.
        - 'c' (also 'c_halfc', 'half_c', 'c_phantom'): 256 x 256 ring (radius 30-75) with a
          60-degree gap (fixed) vs a 120-degree gap (moving); images are the binary masks
          smoothed with sigma 1, labels the binary masks.
        - 'ellipse' (also 'ellipse_circle', 'circle'): 256 x 256 ellipse with semi-axes 70 / 40
          voxels (fixed) vs a circle of radius 53 (moving), same construction.
        - 'mbhard' (also '3d', 'mindboggle_hard', 'mb_hard', 'hard_pair'): Mindboggle
          NKI-TRT-20-2 (fixed) and MMRR-21-2 (moving) T1 brains with DKT31 manual labels,
          read from ``syntx.benchmark.data.resolve_data_dir()`` if all four files exist there,
          else from ``data_dir/mbhard/``. If those are missing too, a synthetic 64^3 sphere
          (radius 20) / ellipsoid (18, 24, 20) pair is written there under the Mindboggle file
          names and returned -- with the same 'mbhard' key and Mindboggle description.
    data_dir : str, optional
        Cache directory, default ``~/.syntx/benchmark_data`` (created if needed, for every
        key).

    Returns
    -------
    dict
        ``'key'`` (canonical: 'r16_r64', 'c', 'ellipse' or 'mbhard'), ``'fixed'``,
        ``'moving'``, ``'fixed_label'``, ``'moving_label'`` (ANTsImage), ``'fixed_labels'`` /
        ``'moving_labels'`` (dict name -> label ANTsImage) and ``'description'`` (str).

    Raises
    ------
    ValueError
        Unknown ``key``.

    Notes
    -----
    Side effects: writes the 'c' / 'ellipse' phantoms (and the synthetic 'mbhard' fallback)
    as NIfTI files in ``data_dir`` on first use; later calls read those files.
    """
    if data_dir is None:
        data_dir = os.path.expanduser("~/.syntx/benchmark_data")
    os.makedirs(data_dir, exist_ok=True)

    key_lower = str(key).lower().strip()

    # 1. 2D Brain Slices: r16 -> r64
    if key_lower in ('2d', 'r16_r64', 'r16', 'r64'):
        fixed = ants.image_read(ants.get_ants_data('r16'))
        moving = ants.image_read(ants.get_ants_data('r64'))

        fixed_otsu = ants.threshold_image(fixed, "Otsu", 3)
        moving_otsu = ants.threshold_image(moving, "Otsu", 3)

        fixed_c2 = fixed_otsu.threshold_image(2, 2)
        moving_c2 = moving_otsu.threshold_image(2, 2)

        fixed_c23 = fixed_otsu.threshold_image(2, 3)
        moving_c23 = moving_otsu.threshold_image(2, 3)

        return {
            'key': 'r16_r64',
            'fixed': fixed,
            'moving': moving,
            'fixed_label': fixed_otsu,
            'moving_label': moving_otsu,
            'fixed_labels': {
                'otsu': fixed_otsu,
                'class2': fixed_c2,
                'class2_3': fixed_c23,
            },
            'moving_labels': {
                'otsu': moving_otsu,
                'class2': moving_c2,
                'class2_3': moving_c23,
            },
            'description': "2D Brain Slices (r16 fixed -> r64 moving) with 3-class Otsu tissue segmentations"
        }

    # 2. Classic 2D C to Half-C Phantom
    elif key_lower in ('c', 'c_halfc', 'half_c', 'c_phantom'):
        fixed_path = os.path.join(data_dir, "c_fixed.nii.gz")
        moving_path = os.path.join(data_dir, "c_moving.nii.gz")
        fixed_lbl_path = os.path.join(data_dir, "c_fixed_lbl.nii.gz")
        moving_lbl_path = os.path.join(data_dir, "c_moving_lbl.nii.gz")

        if not (os.path.exists(fixed_path) and os.path.exists(moving_path) and
                os.path.exists(fixed_lbl_path) and os.path.exists(moving_lbl_path)):
            H, W = 256, 256
            cy, cx = H / 2.0, W / 2.0
            y, x = np.ogrid[:H, :W]
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            theta = np.arctan2(y - cy, x - cx)

            ring = (r >= 30) & (r <= 75)
            cutout_c = (theta >= -np.pi / 6) & (theta <= np.pi / 6)
            c_mask = ring & (~cutout_c)

            cutout_halfc = (theta >= -np.pi / 3) & (theta <= np.pi / 3)
            halfc_mask = ring & (~cutout_halfc)

            img_c = ants.smooth_image(ants.from_numpy(c_mask.astype(np.float32)), 1.0)
            img_halfc = ants.smooth_image(ants.from_numpy(halfc_mask.astype(np.float32)), 1.0)
            lbl_c = ants.from_numpy(c_mask.astype(np.uint32))
            lbl_halfc = ants.from_numpy(halfc_mask.astype(np.uint32))

            ants.image_write(img_c, fixed_path)
            ants.image_write(img_halfc, moving_path)
            ants.image_write(lbl_c, fixed_lbl_path)
            ants.image_write(lbl_halfc, moving_lbl_path)
        else:
            img_c = ants.image_read(fixed_path)
            img_halfc = ants.image_read(moving_path)
            lbl_c = ants.image_read(fixed_lbl_path)
            lbl_halfc = ants.image_read(moving_lbl_path)

        return {
            'key': 'c',
            'fixed': img_c,
            'moving': img_halfc,
            'fixed_label': lbl_c,
            'moving_label': lbl_halfc,
            'fixed_labels': {'c': lbl_c},
            'moving_labels': {'c': lbl_halfc},
            'description': "2D Classic C-shape fixed -> half-C shape moving phantom pair with binary mask labels"
        }

    # 3. Simple 2D Ellipse to Circle Phantom
    elif key_lower in ('ellipse', 'ellipse_circle', 'circle'):
        fixed_path = os.path.join(data_dir, "ellipse_fixed.nii.gz")
        moving_path = os.path.join(data_dir, "ellipse_moving.nii.gz")
        fixed_lbl_path = os.path.join(data_dir, "ellipse_fixed_lbl.nii.gz")
        moving_lbl_path = os.path.join(data_dir, "ellipse_moving_lbl.nii.gz")

        if not (os.path.exists(fixed_path) and os.path.exists(moving_path) and
                os.path.exists(fixed_lbl_path) and os.path.exists(moving_lbl_path)):
            H, W = 256, 256
            cy, cx = H / 2.0, W / 2.0
            y, x = np.ogrid[:H, :W]

            ellipse_mask = ((x - cx) ** 2 / 70.0 ** 2 + (y - cy) ** 2 / 40.0 ** 2) <= 1.0
            circle_mask = ((x - cx) ** 2 / 53.0 ** 2 + (y - cy) ** 2 / 53.0 ** 2) <= 1.0

            img_el = ants.smooth_image(ants.from_numpy(ellipse_mask.astype(np.float32)), 1.0)
            img_circ = ants.smooth_image(ants.from_numpy(circle_mask.astype(np.float32)), 1.0)
            lbl_el = ants.from_numpy(ellipse_mask.astype(np.uint32))
            lbl_circ = ants.from_numpy(circle_mask.astype(np.uint32))

            ants.image_write(img_el, fixed_path)
            ants.image_write(img_circ, moving_path)
            ants.image_write(lbl_el, fixed_lbl_path)
            ants.image_write(lbl_circ, moving_lbl_path)
        else:
            img_el = ants.image_read(fixed_path)
            img_circ = ants.image_read(moving_path)
            lbl_el = ants.image_read(fixed_lbl_path)
            lbl_circ = ants.image_read(moving_lbl_path)

        return {
            'key': 'ellipse',
            'fixed': img_el,
            'moving': img_circ,
            'fixed_label': lbl_el,
            'moving_label': lbl_circ,
            'fixed_labels': {'ellipse': lbl_el},
            'moving_labels': {'circle': lbl_circ},
            'description': "2D Ellipse fixed -> Circle moving phantom pair with binary mask labels"
        }

    # 4. 3D Mindboggle Hard Case
    elif key_lower in ('mbhard', '3d', 'mindboggle_hard', 'mb_hard', 'hard_pair'):
        try:
            from syntx.benchmark.data import resolve_data_dir
            base_data_dir = resolve_data_dir()
            fi_test = os.path.join(base_data_dir, 'NKI-TRT-20_volumes', 'NKI-TRT-20-2', 't1weighted_brain.nii.gz')
            fi_lbl_test = os.path.join(base_data_dir, 'NKI-TRT-20_volumes', 'NKI-TRT-20-2', 'labels.DKT31.manual.nii.gz')
            mi_test = os.path.join(base_data_dir, 'MMRR-21_volumes', 'MMRR-21-2', 't1weighted_brain.nii.gz')
            mi_lbl_test = os.path.join(base_data_dir, 'MMRR-21_volumes', 'MMRR-21-2', 'labels.DKT31.manual.nii.gz')
            if os.path.exists(fi_test) and os.path.exists(fi_lbl_test) and os.path.exists(mi_test) and os.path.exists(mi_lbl_test):
                fi_path, fi_lbl_path = fi_test, fi_lbl_test
                mi_path, mi_lbl_path = mi_test, mi_lbl_test
            else:
                fi_path = fi_lbl_path = mi_path = mi_lbl_path = None
        except Exception:
            fi_path = fi_lbl_path = mi_path = mi_lbl_path = None

        if fi_path is None or not os.path.exists(fi_path):
            mb_dir = os.path.join(data_dir, "mbhard")
            os.makedirs(mb_dir, exist_ok=True)
            fi_path = os.path.join(mb_dir, "NKI-TRT-20-2_t1brain.nii.gz")
            fi_lbl_path = os.path.join(mb_dir, "NKI-TRT-20-2_dkt31.nii.gz")
            mi_path = os.path.join(mb_dir, "MMRR-21-2_t1brain.nii.gz")
            mi_lbl_path = os.path.join(mb_dir, "MMRR-21-2_dkt31.nii.gz")

            if not (os.path.exists(fi_path) and os.path.exists(fi_lbl_path) and
                    os.path.exists(mi_path) and os.path.exists(mi_lbl_path)):
                grid_3d = (64, 64, 64)
                vol_f = np.zeros(grid_3d, dtype=np.float32)
                vol_m = np.zeros(grid_3d, dtype=np.float32)
                z, y, x = np.ogrid[:64, :64, :64]

                mask_f = ((x - 32) ** 2 + (y - 32) ** 2 + (z - 32) ** 2) <= 20 ** 2
                mask_m = ((x - 32) ** 2 / 18.0 ** 2 + (y - 32) ** 2 / 24.0 ** 2 + (z - 32) ** 2 / 20.0 ** 2) <= 1.0

                vol_f[mask_f] = 1.0
                vol_m[mask_m] = 1.0

                img_f = ants.from_numpy(vol_f, spacing=(1.0, 1.0, 1.0))
                img_m = ants.from_numpy(vol_m, spacing=(1.0, 1.0, 1.0))
                lbl_f = ants.from_numpy(mask_f.astype(np.uint32), spacing=(1.0, 1.0, 1.0))
                lbl_m = ants.from_numpy(mask_m.astype(np.uint32), spacing=(1.0, 1.0, 1.0))

                ants.image_write(img_f, fi_path)
                ants.image_write(img_m, mi_path)
                ants.image_write(lbl_f, fi_lbl_path)
                ants.image_write(lbl_m, mi_lbl_path)

        fi_img = ants.image_read(fi_path)
        fi_lbl_img = ants.image_read(fi_lbl_path)
        mi_img = ants.image_read(mi_path)
        mi_lbl_img = ants.image_read(mi_lbl_path)

        return {
            'key': 'mbhard',
            'fixed': fi_img,
            'moving': mi_img,
            'fixed_label': fi_lbl_img,
            'moving_label': mi_lbl_img,
            'fixed_labels': {'dkt31': fi_lbl_img},
            'moving_labels': {'dkt31': mi_lbl_img},
            'description': "3D Mindboggle Hard Pair 00 (NKI-TRT-20-2 fixed -> MMRR-21-2 moving) with DKT31 manual labels"
        }

    else:
        raise ValueError(
            f"Unknown benchmark dataset key '{key}'. Supported keys are: 'r16_r64' ('2d'), 'c', 'ellipse', 'mbhard' ('3d')."
        )
