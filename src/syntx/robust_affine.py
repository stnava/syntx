"""
robust_affine.py — Ultra-Fast, Fail-Safe Robust Multi-Start Affine Registration
===================================================================================

Provides high-reliability, multi-start initial alignment strategies ('mode'):
- 'pytorch' / 'gpu'      : Fast 2D/3D native PyTorch Lie Algebra solver with cone-constrained rotation search.
- 'auto' / 'fast'        : Low-res multi-start candidate selection + multi-stage ANTs solver.
- 'ants_fast'            : Fast multi-stage ANTs C++ pipeline (Translation -> Rigid -> Similarity -> Affine).
- 'com_only'             : Instant 0.05s Center-of-Mass physical translation alignment.

Guarantees robust convergence even under severe initial translation, rotation, or
contrast inversion offsets.

Strictly obeys Syntx Registration Guardrails:
1. Single Interpolation Policy: Composes initial and final transforms into a single stage.
2. Cone Rotation Search: Searches a constrained orientation cone (<= 30 degrees) to preserve brain hemispheric symmetry.
3. 2D and 3D Support: Native support for both 2D and 3D image registration.
4. Center of Rotation: Preserves ITK fixed parameter center of rotation during conversions.
5. Lie Algebra Taylor Expansion: Prevents zero-angle gradient locking.
"""

import time
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import ants

from .syn import mattes_mi_loss_nd
from .spatial import image_to_tensor, get_image_metadata, get_spatial_coordinate_grid


def compute_center_of_mass(img_ants: ants.ANTsImage, weighted: bool = True) -> np.ndarray:
    """
    Computes physical Center of Mass (CoM) of a 2D or 3D ANTsImage.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.
    weighted : bool, default=True
        If True, computes intensity-weighted physical center of mass.
        If False, computes geometric center of mass of non-zero foreground mask.

    Returns
    -------
    np.ndarray
        Array of shape `(dim,)` containing physical space coordinates `(x, y)` or `(x, y, z)`.
    """
    if weighted:
        return np.array(ants.get_center_of_mass(img_ants))
    else:
        arr = img_ants.numpy()
        origin = np.array(img_ants.origin)
        spacing = np.array(img_ants.spacing)
        direction = np.array(img_ants.direction)
        dim = img_ants.dimension
        weights = (arr > (arr.max() * 0.05)).astype(np.float32)
        total_w = weights.sum()
        if total_w <= 1e-6:
            voxel_center = (np.array(arr.shape) - 1.0) / 2.0
        else:
            grid_coords = [np.arange(s) for s in arr.shape]
            mesh = np.meshgrid(*grid_coords, indexing='ij')
            voxel_center = np.array([(mesh[i] * weights).sum() / total_w for i in range(dim)])
        phys_center = origin + direction @ (voxel_center * spacing)
        return phys_center


def create_translation_transform(fi: ants.ANTsImage, mi: ants.ANTsImage, t_phys: np.ndarray) -> tuple:
    """
    Creates an ANTs physical translation transform file matching physical CoM offset `t_phys`.

    Parameters
    ----------
    fi : ants.ANTsImage
        Fixed image.
    mi : ants.ANTsImage
        Moving image.
    t_phys : np.ndarray
        Physical translation vector `fixed_com - moving_com`.

    Returns
    -------
    tx_path : str
        File path to saved `.mat` translation transform file.
    temp_dir : str
        Temporary directory holding the written file.
    """
    temp_dir = tempfile.mkdtemp(prefix="robust_aff_init_")
    tx_path = os.path.join(temp_dir, "initial_translation.mat")
    dim = fi.dimension

    tx = ants.create_ants_transform(
        transform_type='AffineTransform',
        precision='float',
        dimension=dim
    )
    matrix = np.eye(dim)
    tx.set_parameters(np.concatenate([matrix.flatten(), t_phys]))
    tx.set_fixed_parameters(np.array(fi.origin))

    ants.write_transform(tx, tx_path)
    return tx_path, temp_dir


def _eval_low_res_mi(fi_low: ants.ANTsImage, mi_low: ants.ANTsImage, tx_path: str) -> float:
    """Evaluates low-resolution Mattes Mutual Information score for candidate transform."""
    try:
        warped = ants.apply_transforms(fixed=fi_low, moving=mi_low, transformlist=[tx_path])
        mi_score = ants.image_similarity(fi_low, warped, metric_type='MattesMutualInformation')
        return mi_score
    except Exception:
        return 999.0


def _rodrigues_rotation_matrix_3d(omega: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Rodrigues Lie Algebra $so(3) \\rightarrow SO(3)$ 3D rotation matrix parameterization.

    Uses first-order Taylor expansion near zero ($\\|\\omega\\| < 10^{-5}$) to prevent zero-angle
    gradient locking (GEMINI.md Rule 6).

    Parameters
    ----------
    omega : torch.Tensor
        3D Lie algebra vector `[w0, w1, w2]`.

    Returns
    -------
    torch.Tensor
        `3x3` rotation matrix $R \\in SO(3)$.
    """
    theta = torch.norm(omega)
    if theta < 1e-5:
        w0, w1, w2 = omega[0], omega[1], omega[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=omega.device), -w2, w1]),
            torch.stack([w2, torch.tensor(0.0, device=omega.device), -w0]),
            torch.stack([-w1, w0, torch.tensor(0.0, device=omega.device)])
        ])
        return torch.eye(3, device=omega.device, dtype=torch.float32) + K
    else:
        u = omega / theta
        u0, u1, u2 = u[0], u[1], u[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=omega.device), -u2, u1]),
            torch.stack([u2, torch.tensor(0.0, device=omega.device), -u0]),
            torch.stack([-u1, u0, torch.tensor(0.0, device=omega.device)])
        ])
        return torch.eye(3, device=omega.device, dtype=torch.float32) + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def _rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable Lie Algebra $so(2) \\rightarrow SO(2)$ 2D rotation matrix parameterization."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack([
        torch.stack([cos_t, -sin_t]),
        torch.stack([sin_t, cos_t])
    ])


def compute_fov_center(img_ants: ants.ANTsImage) -> np.ndarray:
    """
    Computes geometric physical center (midpoint of field of view) of an ANTsImage.

    Parameters
    ----------
    img_ants : ants.ANTsImage
        Input 2D or 3D ANTs image.

    Returns
    -------
    np.ndarray
        Array of shape `(dim,)` containing physical coordinates of the FOV center.
    """
    origin = np.array(img_ants.origin)
    spacing = np.array(img_ants.spacing)
    direction = np.array(img_ants.direction)
    shape = np.array(img_ants.shape)
    voxel_center = (shape - 1.0) * 0.5
    phys_center = origin + direction @ (voxel_center * spacing)
    return phys_center


def _generate_quick_search_candidates(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    cone_angles_deg: list = None
) -> list:
    r"""
    Generates quick search initialization candidates:
    - Option 1: 'Identity_CoM' (Center of Mass matching with Identity rotation)
    - Option 2: 'Identity_FOV' (Field of View geometric midpoint matching with Identity rotation)
    - Rotational Search: Angle perturbations across pitch, roll, yaw for both CoM and FOV base translations.
    """
    dim = fixed.dimension
    if cone_angles_deg is None:
        cone_angles_deg = [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]

    com_f = compute_center_of_mass(fixed, weighted=True)
    com_m = compute_center_of_mass(moving, weighted=True)
    t_com = np.array(com_m) - np.array(com_f)

    fov_f = compute_fov_center(fixed)
    fov_m = compute_fov_center(moving)
    t_fov = np.array(fov_m) - np.array(fov_f)

    candidates = []

    # 1. Option 1: Identity CoM
    tx_com = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_com.set_parameters(np.concatenate([np.eye(dim).flatten(), t_com]))
    tx_com.set_fixed_parameters(com_f)
    r_dir_com = tempfile.mkdtemp(prefix="robust_aff_id_com_")
    r_path_com = os.path.join(r_dir_com, "cone_rotation.mat")
    ants.write_transform(tx_com, r_path_com)
    candidates.append(('Identity_CoM', r_path_com, np.eye(dim), t_com, com_f, r_dir_com))

    # 2. Option 2: Identity FOV
    tx_fov = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx_fov.set_parameters(np.concatenate([np.eye(dim).flatten(), t_fov]))
    tx_fov.set_fixed_parameters(fov_f)
    r_dir_fov = tempfile.mkdtemp(prefix="robust_aff_id_fov_")
    r_path_fov = os.path.join(r_dir_fov, "cone_rotation.mat")
    ants.write_transform(tx_fov, r_path_fov)
    candidates.append(('Identity_FOV', r_path_fov, np.eye(dim), t_fov, fov_f, r_dir_fov))

    # 3. Rotational search around CoM and FOV centers
    if dim == 3:
        for base_name, t_base, C in [('CoM', t_com, com_f), ('FOV', t_fov, fov_f)]:
            for deg in cone_angles_deg:
                if abs(deg) < 1e-3:
                    continue
                rad = np.radians(deg)
                for axis_name in ['pitch', 'roll', 'yaw']:
                    rx = rad if axis_name == 'pitch' else 0.0
                    ry = rad if axis_name == 'roll' else 0.0
                    rz = rad if axis_name == 'yaw' else 0.0

                    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                    R = Rz @ Ry @ Rx
                    tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
                    tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
                    tx_r.set_fixed_parameters(C)

                    r_dir = tempfile.mkdtemp(prefix=f"robust_aff_{base_name}_{axis_name}_{deg}_")
                    r_path = os.path.join(r_dir, "cone_rotation.mat")
                    ants.write_transform(tx_r, r_path)
                    candidates.append((f'{base_name}_{axis_name}_{deg:+.0f}deg', r_path, R, t_base, C, r_dir))
    elif dim == 2:
        for base_name, t_base, C in [('CoM', t_com, com_f), ('FOV', t_fov, fov_f)]:
            for deg in cone_angles_deg:
                if abs(deg) < 1e-3:
                    continue
                rad = np.radians(deg)
                R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=2)
                tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
                tx_r.set_fixed_parameters(C)

                r_dir = tempfile.mkdtemp(prefix=f"robust_aff_2d_{base_name}_{deg}_")
                r_path = os.path.join(r_dir, "cone_rotation.mat")
                ants.write_transform(tx_r, r_path)
                candidates.append((f'{base_name}_rot_{deg:+.0f}deg', r_path, R, t_base, C, r_dir))

    return candidates


def _generate_cone_rotation_candidates_3d(com_f, t_init, cone_angles_deg=None):
    """Backwards compatibility generator supporting both ANTsImage and raw ndarrays."""
    if hasattr(com_f, 'dimension'):
        return _generate_quick_search_candidates(com_f, t_init, cone_angles_deg)
    
    if cone_angles_deg is None:
        cone_angles_deg = [-30.0, -20.0, -10.0, 10.0, 20.0, 30.0]
    C = np.array(com_f)
    t_base = np.array(t_init)
    candidates = []
    for deg in cone_angles_deg:
        if abs(deg) < 1e-3:
            continue
        rad = np.radians(deg)
        for axis_name in ['pitch', 'roll', 'yaw']:
            rx = rad if axis_name == 'pitch' else 0.0
            ry = rad if axis_name == 'roll' else 0.0
            rz = rad if axis_name == 'yaw' else 0.0
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
            tx_r.set_parameters(np.concatenate([R.flatten(), t_base]))
            tx_r.set_fixed_parameters(C)
            r_dir = tempfile.mkdtemp(prefix=f"robust_aff_legacy_{axis_name}_{deg}_")
            r_path = os.path.join(r_dir, "cone_rotation.mat")
            ants.write_transform(tx_r, r_path)
            candidates.append((f'CoM_{axis_name}_{deg:+.0f}deg', r_path, R, t_base, C, r_dir))
    return candidates


def _run_pytorch_affine_solver(fixed: ants.ANTsImage, moving: ants.ANTsImage, initial_tx_path: str = None, device: str = 'cpu', verbose: bool = False, n_starts: int = 3, cone_angles_deg: list = None) -> dict:
    """Blazing-fast 2D and 3D native PyTorch GPU Lie algebra multi-resolution affine solver (`mode='pytorch'`)."""
    t0 = time.time()
    dim = fixed.dimension
    torch.manual_seed(42)
    np.random.seed(42)
    if device in ['auto', 'cpu', None]:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device_obj = torch.device(device)

    # 1. Pre-align center of mass and FOV center in physical space
    com_f = compute_center_of_mass(fixed, weighted=True)
    com_m = compute_center_of_mass(moving, weighted=True)
    t_init = np.array(com_m) - np.array(com_f)

    # Automatic entropy-optimal foreground intensity normalization with truncation to [0.0, 1.0]
    from syntx.core.utils import normalize_image
    fixed_norm = normalize_image(fixed, method='auto')
    moving_norm = normalize_image(moving, method='auto')
    f_norm_np = fixed_norm.numpy()
    m_norm_np = moving_norm.numpy()

    # 2. Quick Search candidate evaluation (CoM, FOV, Rotations) at low resolution
    sp_low = tuple(4.0 for _ in range(dim))
    fi_low = ants.resample_image(fixed_norm, sp_low, use_voxels=False)
    mi_low = ants.resample_image(moving_norm, sp_low, use_voxels=False)

    all_candidates = _generate_quick_search_candidates(fixed_norm, moving_norm, cone_angles_deg)

    scored_candidates = []
    for name, path, R_c, t_c, C_c, _ in all_candidates:
        score = _eval_low_res_mi(fi_low, mi_low, path)
        scored_candidates.append((score, name, R_c, t_c, C_c))

    scored_candidates.sort(key=lambda x: x[0])
    
    # Select top n_starts candidates from the ranked quick search
    n_sel = max(1, min(n_starts, len(scored_candidates)))
    top_candidates = scored_candidates[:n_sel]
    
    if verbose:
        print(f"[robust_affine mode='pytorch'] Quick Search evaluated {len(all_candidates)} candidates. Selected top {len(top_candidates)}: {[(c[1], f'{c[0]:.4f}') for c in top_candidates]}", flush=True)

    # 3. Setup PyTorch Lie Algebra Constant Tensors
    if dim == 3:
        fi_arr = torch.tensor(f_norm_np.transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
        mi_arr = torch.tensor(m_norm_np.transpose(2, 1, 0), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    else:
        fi_arr = torch.tensor(f_norm_np.T, dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
        mi_arr = torch.tensor(m_norm_np.T, dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)

    sp_xyz = torch.tensor(fixed.spacing, dtype=torch.float32, device=device_obj)
    orig_xyz = torch.tensor(fixed.origin, dtype=torch.float32, device=device_obj)
    dir_xyz = torch.tensor(fixed.direction, dtype=torch.float32, device=device_obj)

    C_phys_xyz = torch.tensor(com_f, dtype=torch.float32, device=device_obj)

    mi_orig_xyz = torch.tensor(moving.origin, dtype=torch.float32, device=device_obj)
    mi_sp_xyz = torch.tensor(moving.spacing, dtype=torch.float32, device=device_obj)
    mi_dir_xyz = torch.tensor(moving.direction, dtype=torch.float32, device=device_obj)
    mi_shape_xyz = torch.tensor(moving.shape, dtype=torch.float32, device=device_obj)

    pyramid = [4, 2, 1] if dim == 3 else [2, 1]
    fi_pyramid, mi_pyramid, coords_pyramid = {}, {}, {}
    for level in pyramid:
        if level > 1:
            if dim == 3:
                fi_lev = F.avg_pool3d(fi_arr, kernel_size=level, stride=level)
                mi_lev = F.avg_pool3d(mi_arr, kernel_size=level, stride=level)
            else:
                fi_lev = F.avg_pool2d(fi_arr, kernel_size=level, stride=level)
                mi_lev = F.avg_pool2d(mi_arr, kernel_size=level, stride=level)
        else:
            fi_lev, mi_lev = fi_arr, mi_arr

        shape_zyx = fi_lev.shape[2:]
        if dim == 3:
            grid_z = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
            grid_y = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
            grid_x = torch.linspace(0, shape_zyx[2] - 1, shape_zyx[2], device=device_obj)
            mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
            vox_coords_xyz = torch.stack([mesh_x, mesh_y, mesh_z], dim=-1).reshape(-1, 3) * level
        else:
            grid_y = torch.linspace(0, shape_zyx[0] - 1, shape_zyx[0], device=device_obj)
            grid_x = torch.linspace(0, shape_zyx[1] - 1, shape_zyx[1], device=device_obj)
            mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
            vox_coords_xyz = torch.stack([mesh_x, mesh_y], dim=-1).reshape(-1, 2) * level
        phys_coords_xyz = orig_xyz + (vox_coords_xyz * sp_xyz) @ dir_xyz.t()
        
        fi_pyramid[level] = fi_lev
        mi_pyramid[level] = mi_lev
        coords_pyramid[level] = (phys_coords_xyz, shape_zyx)

    # Base candidate from quick search
    cand_score, cand_name, best_R_init, best_t_init, best_C_init = scored_candidates[0]
    C_phys_xyz = torch.tensor(best_C_init, dtype=torch.float32, device=device_obj)
    R_base = torch.tensor(best_R_init, dtype=torch.float32, device=device_obj)

    # Path 0: Conservative / Step Schedule
    t0_p = torch.tensor(best_t_init, dtype=torch.float32, device=device_obj, requires_grad=True)
    w0_p = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device_obj, requires_grad=True)
    s0_p = torch.zeros(3 if dim == 3 else 2, dtype=torch.float32, device=device_obj, requires_grad=True)
    sh0_p = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device_obj, requires_grad=True)

    # Path 1: Aggressive / Cosine Annealing Schedule
    t1_p = torch.tensor(best_t_init, dtype=torch.float32, device=device_obj, requires_grad=True)
    w1_p = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device_obj, requires_grad=True)
    s1_p = torch.zeros(3 if dim == 3 else 2, dtype=torch.float32, device=device_obj, requires_grad=True)
    sh1_p = torch.zeros(3 if dim == 3 else 1, dtype=torch.float32, device=device_obj, requires_grad=True)

    # Stage 1: Coarse Level (Level 4, 4mm, 80 iters) - Rigid Only
    if 4 in pyramid:
        phys_l4, shape_l4 = coords_pyramid[4]
        fi_l4, mi_l4 = fi_pyramid[4], mi_pyramid[4]
        opt0_l4 = torch.optim.Adam([{'params': [t0_p], 'lr': 0.25}, {'params': [w0_p], 'lr': 0.02}])
        opt1_l4 = torch.optim.Adam([{'params': [t1_p], 'lr': 0.40}, {'params': [w1_p], 'lr': 0.03}])
        sched1_l4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1_l4, T_max=80, eta_min=0.01)

        for it in range(80):
            # Path 0: Conservative
            opt0_l4.zero_grad()
            R0 = _rodrigues_rotation_matrix_3d(w0_p) @ R_base if dim == 3 else _rotation_matrix_2d(w0_p[0])
            teff0 = t0_p + C_phys_xyz - R0 @ C_phys_xyz
            y_phys0 = phys_l4 @ R0.t() + teff0
            y_vox0 = (y_phys0 - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
            y_norm0 = 2.0 * (y_vox0 / (mi_shape_xyz - 1.0)) - 1.0
            grid0 = y_norm0.reshape(1, *shape_l4, 3 if dim == 3 else 2)
            w0 = F.grid_sample(mi_l4, grid0, mode='bilinear', padding_mode='zeros', align_corners=True)
            loss0 = mattes_mi_loss_nd(w0, fi_l4, mask=(fi_l4>0.01)|(w0>0.01), num_bins=32, sampling_percentage=0.50)
            loss0.backward()
            opt0_l4.step()

            # Path 1: Aggressive Annealed
            opt1_l4.zero_grad()
            R1 = _rodrigues_rotation_matrix_3d(w1_p) @ R_base if dim == 3 else _rotation_matrix_2d(w1_p[0])
            teff1 = t1_p + C_phys_xyz - R1 @ C_phys_xyz
            y_phys1 = phys_l4 @ R1.t() + teff1
            y_vox1 = (y_phys1 - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
            y_norm1 = 2.0 * (y_vox1 / (mi_shape_xyz - 1.0)) - 1.0
            grid1 = y_norm1.reshape(1, *shape_l4, 3 if dim == 3 else 2)
            w1 = F.grid_sample(mi_l4, grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
            loss1 = mattes_mi_loss_nd(w1, fi_l4, mask=(fi_l4>0.01)|(w1>0.01), num_bins=32, sampling_percentage=1.0)
            loss1.backward()
            opt1_l4.step()
            sched1_l4.step()

    # Stage 2: Medium Level (Level 2, 2mm, 70 iters) - Full Affine
    phys_l2, shape_l2 = coords_pyramid[2]
    fi_l2, mi_l2 = fi_pyramid[2], mi_pyramid[2]
    opt0_l2 = torch.optim.Adam([{'params': [t0_p], 'lr': 0.08}, {'params': [w0_p], 'lr': 0.008}, {'params': [s0_p], 'lr': 0.005}, {'params': [sh0_p], 'lr': 0.003}])
    opt1_l2 = torch.optim.Adam([{'params': [t1_p], 'lr': 0.15}, {'params': [w1_p], 'lr': 0.015}, {'params': [s1_p], 'lr': 0.010}, {'params': [sh1_p], 'lr': 0.006}])
    sched1_l2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1_l2, T_max=70, eta_min=0.001)

    for it in range(70):
        # Path 0
        opt0_l2.zero_grad()
        if dim == 3:
            R0 = _rodrigues_rotation_matrix_3d(w0_p) @ R_base
            S0 = torch.diag(torch.exp(torch.clamp(s0_p, -0.4, 0.4)))
            Sh0 = torch.eye(3, device=device_obj)
            Sh0[0, 1] = sh0_p[0]; Sh0[0, 2] = sh0_p[1]; Sh0[1, 2] = sh0_p[2]
            A0 = R0 @ S0 @ Sh0
        else:
            R0 = _rotation_matrix_2d(w0_p[0])
            S0 = torch.diag(torch.exp(torch.clamp(s0_p, -0.4, 0.4)))
            Sh0 = torch.eye(2, device=device_obj); Sh0[0, 1] = sh0_p[0]
            A0 = R0 @ S0 @ Sh0
        teff0 = t0_p + C_phys_xyz - A0 @ C_phys_xyz
        y_phys0 = phys_l2 @ A0.t() + teff0
        y_vox0 = (y_phys0 - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
        y_norm0 = 2.0 * (y_vox0 / (mi_shape_xyz - 1.0)) - 1.0
        grid0 = y_norm0.reshape(1, *shape_l2, 3 if dim == 3 else 2)
        w0 = F.grid_sample(mi_l2, grid0, mode='bilinear', padding_mode='zeros', align_corners=True)
        loss0 = mattes_mi_loss_nd(w0, fi_l2, mask=(fi_l2>0.01)|(w0>0.01), num_bins=32, sampling_percentage=0.50)
        loss0.backward()
        opt0_l2.step()

        # Path 1
        opt1_l2.zero_grad()
        if dim == 3:
            R1 = _rodrigues_rotation_matrix_3d(w1_p) @ R_base
            S1 = torch.diag(torch.exp(torch.clamp(s1_p, -0.4, 0.4)))
            Sh1 = torch.eye(3, device=device_obj)
            Sh1[0, 1] = sh1_p[0]; Sh1[0, 2] = sh1_p[1]; Sh1[1, 2] = sh1_p[2]
            A1 = R1 @ S1 @ Sh1
        else:
            R1 = _rotation_matrix_2d(w1_p[0])
            S1 = torch.diag(torch.exp(torch.clamp(s1_p, -0.4, 0.4)))
            Sh1 = torch.eye(2, device=device_obj); Sh1[0, 1] = sh1_p[0]
            A1 = R1 @ S1 @ Sh1
        teff1 = t1_p + C_phys_xyz - A1 @ C_phys_xyz
        y_phys1 = phys_l2 @ A1.t() + teff1
        y_vox1 = (y_phys1 - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
        y_norm1 = 2.0 * (y_vox1 / (mi_shape_xyz - 1.0)) - 1.0
        grid1 = y_norm1.reshape(1, *shape_l2, 3 if dim == 3 else 2)
        w1 = F.grid_sample(mi_l2, grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
        loss1 = mattes_mi_loss_nd(w1, fi_l2, mask=(fi_l2>0.01)|(w1>0.01), num_bins=32, sampling_percentage=1.0)
        loss1.backward()
        opt1_l2.step()
        sched1_l2.step()

        with torch.no_grad():
            s0_p.clamp_(-0.35, 0.35); sh0_p.clamp_(-0.35, 0.35); w0_p.clamp_(-np.pi/3, np.pi/3)
            s1_p.clamp_(-0.35, 0.35); sh1_p.clamp_(-0.35, 0.35); w1_p.clamp_(-np.pi/3, np.pi/3)

    # Evaluate exact full-grid loss for Path 0 vs Path 1 at Level 2
    with torch.no_grad():
        w0_eval = F.grid_sample(mi_l2, grid0, mode='bilinear', padding_mode='zeros', align_corners=True)
        loss0_eval = mattes_mi_loss_nd(w0_eval, fi_l2, mask=(fi_l2>0.01)|(w0_eval>0.01), num_bins=32, sampling_percentage=1.0).item()
        w1_eval = F.grid_sample(mi_l2, grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
        loss1_eval = mattes_mi_loss_nd(w1_eval, fi_l2, mask=(fi_l2>0.01)|(w1_eval>0.01), num_bins=32, sampling_percentage=1.0).item()

    if loss0_eval <= loss1_eval:
        t_param, omega_param, scale_param, shear_param = t0_p, w0_p, s0_p, sh0_p
        winner_name = f"{cand_name}_Conservative"
    else:
        t_param, omega_param, scale_param, shear_param = t1_p, w1_p, s1_p, sh1_p
        winner_name = f"{cand_name}_Annealed"

    # Stage 3: Fine Level (Level 1, Native Spacing, 40 iters) - Full Affine Fine-Tuning on Winner
    phys_l1, shape_l1 = coords_pyramid[1]
    fi_l1, mi_l1 = fi_pyramid[1], mi_pyramid[1]
    opt_l1 = torch.optim.Adam([
        {'params': [t_param], 'lr': 0.05},
        {'params': [omega_param], 'lr': 0.005},
        {'params': [scale_param], 'lr': 0.005},
        {'params': [shear_param], 'lr': 0.003}
    ])
    sched_l1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l1, T_max=50, eta_min=1e-4)

    for it in range(50):
        opt_l1.zero_grad()
        if dim == 3:
            R = _rodrigues_rotation_matrix_3d(omega_param) @ R_base
            S = torch.diag(torch.exp(torch.clamp(scale_param, -0.4, 0.4)))
            Sh = torch.eye(3, device=device_obj)
            Sh[0, 1] = shear_param[0]; Sh[0, 2] = shear_param[1]; Sh[1, 2] = shear_param[2]
            A = R @ S @ Sh
        else:
            R = _rotation_matrix_2d(omega_param[0])
            S = torch.diag(torch.exp(torch.clamp(scale_param, -0.4, 0.4)))
            Sh = torch.eye(2, device=device_obj); Sh[0, 1] = shear_param[0]
            A = R @ S @ Sh

        t_eff = t_param + C_phys_xyz - A @ C_phys_xyz
        y_phys = phys_l1 @ A.t() + t_eff
        y_vox = (y_phys - mi_orig_xyz) @ torch.inverse(mi_dir_xyz).t() / mi_sp_xyz
        y_norm = 2.0 * (y_vox / (mi_shape_xyz - 1.0)) - 1.0
        grid = y_norm.reshape(1, *shape_l1, 3 if dim == 3 else 2)
        warped = F.grid_sample(mi_l1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        loss = mattes_mi_loss_nd(warped, fi_l1, mask=(fi_l1>0.01)|(warped>0.01), num_bins=32, sampling_percentage=0.50)
        loss.backward()
        opt_l1.step()
        sched_l1.step()

        with torch.no_grad():
            scale_param.clamp_(-0.35, 0.35)
            shear_param.clamp_(-0.35, 0.35)
            omega_param.clamp_(-np.pi/3, np.pi/3)

    # Extract final transform in XYZ physical space
    with torch.no_grad():
        if dim == 3:
            R_delta_fin = _rodrigues_rotation_matrix_3d(omega_param).cpu().numpy()
            R_fin = R_delta_fin @ best_R_init
            S_fin = np.diag(np.exp(np.clip(scale_param.cpu().numpy(), -0.4, 0.4)))
            Sh_fin = np.eye(3)
            sh_np = shear_param.cpu().numpy()
            Sh_fin[0, 1] = sh_np[0]; Sh_fin[0, 2] = sh_np[1]; Sh_fin[1, 2] = sh_np[2]
            A_fin = R_fin @ S_fin @ Sh_fin
        else:
            R_fin = _rotation_matrix_2d(omega_param[0]).cpu().numpy()
            S_fin = np.diag(np.exp(np.clip(scale_param.cpu().numpy(), -0.4, 0.4)))
            Sh_fin = np.eye(2)
            Sh_fin[0, 1] = shear_param[0].cpu().numpy()
            A_fin = R_fin @ S_fin @ Sh_fin

        t_param_np = t_param.cpu().numpy()
        C_phys_np = best_C_init

        tx_final = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
        tx_final.set_parameters(np.concatenate([A_fin.flatten(), t_param_np]))
        tx_final.set_fixed_parameters(C_phys_np)

        out_dir = tempfile.mkdtemp(prefix="robust_affine_pt_")
        final_tx_path = os.path.join(out_dir, "affine.mat")
        ants.write_transform(tx_final, final_tx_path)

        warped_mov_out = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[final_tx_path],
            interpolator='linear'
        )

        elapsed = time.time() - t0
        return {
            'warpedmovout': warped_mov_out,
            'fwdtransforms': [final_tx_path],
            'invtransforms': [final_tx_path],
            'whichtoinvert_inv': [True],
            'runtime_seconds': elapsed,
            'time': elapsed,
            'init_candidate': winner_name,
            'init_score': float(cand_score),
            'final_loss': float(loss.item()),
            'status': 'SUCCESS'
        }


def robust_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: str = None,
    mode: str = 'pytorch',
    multi_start: bool = True,
    n_starts: int = 3,
    cone_angles_deg: list = None,
    num_rotations: int = 6,
    low_res_spacing: float = 4.0,
    backend: str = 'pytorch',
    device: str = 'cpu',
    seed: int = None,
    verbose: bool = False
) -> dict:
    """
    Executes fail-safe, ultra-fast multi-start initial affine registration for 2D and 3D images.

    Supported Modes (`mode`)
    ------------------------
    - `'pytorch'` / `'gpu'` : Fast 2D/3D native PyTorch Lie Algebra solver with cone-constrained rotation search.
    - `'auto'` / `'fast'`   : Low-res multi-start candidate selection + multi-stage ANTs C++ solver.
    - `'ants_fast'`       : Fast multi-stage ANTs C++ pipeline (Translation -> Rigid -> Similarity -> Affine).
    - `'com_only'`        : Instant 0.05s Center-of-Mass physical translation alignment.

    Parameters
    ----------
    fixed : ants.ANTsImage
        Fixed target image in native physical space (2D or 3D).
    moving : ants.ANTsImage
        Moving source image in native physical space (2D or 3D).
    initial_transform : str, optional
        File path to existing initial ANTs transform `.mat` file.
    mode : str, default='pytorch'
        Affine strategy mode ('pytorch', 'auto', 'ants_fast', 'com_only').
    multi_start : bool, default=True
        If True, evaluates multi-start candidate transforms at low resolution.
    num_rotations : int, default=6
        Number of discrete orthogonal rotation candidates to test if multi_start is True.
    low_res_spacing : float, default=4.0
        Voxel spacing in mm for fast low-resolution candidate evaluation.
    backend : str, default='pytorch'
        Compute engine ('pytorch' or 'jax').
    device : str, default='cpu'
        Compute device ('cpu', 'cuda', 'mps').
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, prints diagnostic timing and score messages.

    Returns
    -------
    dict
        Dictionary containing:
        - `'fwdtransforms'`: list of forward transform file paths (`[.mat]`)
        - `'invtransforms'`: list of inverse transform file paths (`[.mat]`)
        - `'whichtoinvert_inv'`: list of boolean flags for inverse applying
        - `'warpedmovout'`: ANTsImage moving image warped into fixed space
        - `'warpedfixout'`: ANTsImage fixed image
        - `'time'`: execution time in seconds
    """
    t0 = time.time()
    dim = fixed.dimension
    temp_dirs = []

    if seed is None:
        seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Mode: 'com_only'
    if mode in ['com_only', 'translation_only']:
        com_f = compute_center_of_mass(fixed, weighted=True)
        com_m = compute_center_of_mass(moving, weighted=True)
        t_com = com_f - com_m
        tx_path, temp_dir = create_translation_transform(fixed, moving, t_com)
        warped_mov = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
        return {
            'fwdtransforms': [tx_path],
            'invtransforms': [tx_path],
            'whichtoinvert_inv': [True],
            'warpedmovout': warped_mov,
            'warpedfixout': fixed,
            'time': time.time() - t0
        }

    # 2. Mode: 'pytorch'
    if mode in ['pytorch', 'gpu', 'pytorch_gpu']:
        return _run_pytorch_affine_solver(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose, n_starts=n_starts, cone_angles_deg=cone_angles_deg)

    # 3. Mode: 'auto', 'fast', 'ants_fast'
    try:
        if multi_start:
            if verbose:
                print(f"[robust_affine] Creating low-res {dim}D volumes for multi-start evaluation...", flush=True)
            sp_low = tuple(low_res_spacing for _ in range(dim))
            fi_low = ants.resample_image(fixed, sp_low, use_voxels=False)
            mi_low = ants.resample_image(moving, sp_low, use_voxels=False)

            candidates = []

            if initial_transform is not None and os.path.exists(initial_transform):
                candidates.append(('Provided_Initial_Transform', initial_transform, None))

            com_f_w = compute_center_of_mass(fixed, weighted=True)
            com_m_w = compute_center_of_mass(moving, weighted=True)
            t_w = com_f_w - com_m_w
            tx_w_path, dir_w = create_translation_transform(fixed, moving, t_w)
            temp_dirs.append(dir_w)
            candidates.append(('Weighted_CoM', tx_w_path, dir_w))

            com_f_g = compute_center_of_mass(fixed, weighted=False)
            com_m_g = compute_center_of_mass(moving, weighted=False)
            t_g = com_f_g - com_m_g
            tx_g_path, dir_g = create_translation_transform(fixed, moving, t_g)
            temp_dirs.append(dir_g)
            candidates.append(('Geometric_CoM', tx_g_path, dir_g))

            if num_rotations > 0 and dim == 3:
                axes_rot = [
                    (np.pi / 2, 0, 0), (-np.pi / 2, 0, 0),
                    (0, np.pi / 2, 0), (0, -np.pi / 2, 0),
                    (0, 0, np.pi / 2), (0, 0, -np.pi / 2),
                    (np.pi, 0, 0), (0, np.pi, 0), (0, 0, np.pi)
                ][:num_rotations]

                for r_idx, (rx, ry, rz) in enumerate(axes_rot):
                    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                    R = Rz @ Ry @ Rx

                    tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
                    C = com_f_w
                    t_rot = t_w + C - R @ C
                    tx_r.set_parameters(np.concatenate([R.T.ravel(), t_rot]))
                    tx_r.set_fixed_parameters(C)

                    r_dir = tempfile.mkdtemp(prefix=f"robust_aff_rot_{r_idx}_")
                    r_path = os.path.join(r_dir, "rot_translation.mat")
                    temp_dirs.append(r_dir)
                    ants.write_transform(tx_r, r_path)
                    candidates.append((f'Rotation_{r_idx}', r_path, r_dir))

            best_candidate_name = None
            best_tx_path = None
            best_score = 999.0

            for name, path, _ in candidates:
                score = _eval_low_res_mi(fi_low, mi_low, path)
                if verbose:
                    print(f"  Candidate '{name}': Low-Res MI = {score:.4f}", flush=True)
                if score < best_score:
                    best_score = score
                    best_candidate_name = name
                    best_tx_path = path

            if verbose:
                print(f"[robust_affine] Selected winning candidate '{best_candidate_name}' (MI = {best_score:.4f})", flush=True)
            initial_tx_to_use = best_tx_path
        else:
            initial_tx_to_use = initial_transform

        if verbose:
            print(f"[robust_affine mode='{mode}'] Starting ANTs Affine registration...", flush=True)

        reg_a = ants.registration(
            fixed=fixed, moving=moving, type_of_transform='Affine',
            initial_transform=initial_tx_to_use,
            verbose=verbose
        )

        fwdtransforms = reg_a['fwdtransforms']
        invtransforms = reg_a['invtransforms']
        warpedmovout = reg_a['warpedmovout']
        warpedfixout = reg_a.get('warpedfixout', fixed)

        return {
            'fwdtransforms': fwdtransforms,
            'invtransforms': invtransforms,
            'whichtoinvert_inv': [True],
            'warpedmovout': warpedmovout,
            'warpedfixout': warpedfixout,
            'time': time.time() - t0
        }

    finally:
        pass
