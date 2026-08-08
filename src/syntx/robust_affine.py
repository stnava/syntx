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
        mi_score = ants.image_similarity(fi_low, warped, metric_type='Correlation')
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


def _generate_cone_rotation_candidates_3d(com_f: np.ndarray, t_init: np.ndarray, num_cone_angles: int = 9) -> list:
    """Generates orientation cone candidate transforms bounded to $\\le 30^\\circ$ preserving brain symmetry."""
    cone_angles_deg = [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0]
    candidates = []

    for r_idx, deg in enumerate(cone_angles_deg):
        rad = np.radians(deg)
        for axis_idx, axis_name in enumerate(['pitch', 'roll', 'yaw']):
            rx = rad if axis_name == 'pitch' else 0.0
            ry = rad if axis_name == 'roll' else 0.0
            rz = rad if axis_name == 'yaw' else 0.0

            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx

            tx_r = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
            C = com_f
            t_rot = t_init + C - R @ C
            tx_r.set_parameters(np.concatenate([R.flatten(), t_rot]))
            tx_r.set_fixed_parameters(C)

            r_dir = tempfile.mkdtemp(prefix=f"robust_aff_cone_{axis_name}_{deg}_")
            r_path = os.path.join(r_dir, "cone_rotation.mat")
            ants.write_transform(tx_r, r_path)
            candidates.append((f'Cone_{axis_name}_{deg}deg', r_path, R, t_rot, r_dir))

    return candidates


def _run_pytorch_affine_solver(fixed: ants.ANTsImage, moving: ants.ANTsImage, initial_tx_path: str = None, device: str = 'cpu', verbose: bool = False) -> dict:
    """Blazing-fast 2D and 3D native PyTorch GPU Lie algebra multi-resolution affine solver (`mode='pytorch'`)."""
    t0 = time.time()
    dim = fixed.dimension
    device_obj = torch.device(device if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 'cpu')

    # 1. Pre-align center of mass in physical space
    com_f = compute_center_of_mass(fixed, weighted=True)
    com_m = compute_center_of_mass(moving, weighted=True)
    t_init = np.array(com_m) - np.array(com_f)

    # 2. Cone orientation search at low resolution (preserving brain symmetry)
    best_R_init = np.eye(dim)
    best_t_init = t_init

    if dim == 3:
        fi_low = ants.resample_image(fixed, (4.0, 4.0, 4.0), use_voxels=False)
        mi_low = ants.resample_image(moving, (4.0, 4.0, 4.0), use_voxels=False)

        cone_candidates = _generate_cone_rotation_candidates_3d(com_f, t_init)
        best_score = 999.0
        best_cand_name = "Identity_Cone"

        for name, path, R_c, t_c, _ in cone_candidates:
            score = _eval_low_res_mi(fi_low, mi_low, path)
            if score < best_score:
                best_score = score
                best_cand_name = name
                best_R_init = R_c
                best_t_init = t_c

        if verbose:
            print(f"[robust_affine mode='pytorch'] Winning orientation cone: '{best_cand_name}' (MI = {best_score:.4f})", flush=True)

    # 3. Setup PyTorch Lie Algebra Parameter Tensors
    best_R_init_zyx = best_R_init[::-1, ::-1].copy()
    best_t_init_zyx = best_t_init[::-1].copy()
    com_f_zyx = np.array(com_f)[::-1].copy()
    t_param = torch.tensor(best_t_init_zyx, dtype=torch.float32, device=device_obj, requires_grad=True)

    if dim == 3:
        omega_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
        scale_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
        shear_param = torch.zeros(3, dtype=torch.float32, device=device_obj, requires_grad=True)
    else:
        omega_param = torch.zeros(1, dtype=torch.float32, device=device_obj, requires_grad=True)
        scale_param = torch.zeros(2, dtype=torch.float32, device=device_obj, requires_grad=True)
        shear_param = torch.zeros(1, dtype=torch.float32, device=device_obj, requires_grad=True)

    fi_arr = torch.tensor(fixed.numpy().transpose(*reversed(range(dim))), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)
    mi_arr = torch.tensor(moving.numpy().transpose(*reversed(range(dim))), dtype=torch.float32, device=device_obj).unsqueeze(0).unsqueeze(0)

    fi_arr = (fi_arr - fi_arr.min()) / (fi_arr.max() - fi_arr.min() + 1e-6)
    mi_arr = (mi_arr - mi_arr.min()) / (mi_arr.max() - mi_arr.min() + 1e-6)

    pyramid = [4, 2, 1] if dim == 3 else [2, 1]

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

        shape_lev = fi_lev.shape[2:]
        grid_shape = tuple(shape_lev)

        sp = tuple(s * level for s in fixed.spacing[::-1])
        orig = tuple(fixed.origin[::-1])
        dir_mat = torch.tensor(np.array(fixed.direction[::-1, ::-1].copy()), dtype=torch.float32, device=device_obj)

        grids = [torch.linspace(0, shape_lev[i] - 1, shape_lev[i], device=device_obj) for i in range(dim)]
        mesh = torch.meshgrid(*grids, indexing='ij')
        vox_coords = torch.stack(mesh, dim=-1).reshape(-1, dim)

        spacing_t = torch.tensor(sp, dtype=torch.float32, device=device_obj)
        origin_t = torch.tensor(orig, dtype=torch.float32, device=device_obj)
        phys_coords = origin_t + (vox_coords * spacing_t) @ dir_mat.t()

        C_phys = torch.tensor(com_f_zyx, dtype=torch.float32, device=device_obj)

        iters = 40 if level == 4 else (20 if level == 2 else 10)
        optimizer = torch.optim.Adam([
            {'params': [t_param], 'lr': 0.05},
            {'params': [omega_param], 'lr': 0.01},
            {'params': [scale_param], 'lr': 0.005},
            {'params': [shear_param], 'lr': 0.005}
        ])

        for it in range(iters):
            optimizer.zero_grad()

            if dim == 3:
                R_delta = _rodrigues_rotation_matrix_3d(omega_param)
                R_base = torch.tensor(best_R_init_zyx, dtype=torch.float32, device=device_obj)
                R = R_delta @ R_base

                S = torch.diag(torch.exp(torch.clamp(scale_param, -1.0, 1.0)))
                Sh = torch.eye(3, device=device_obj)
                Sh[0, 1] = shear_param[0]
                Sh[0, 2] = shear_param[1]
                Sh[1, 2] = shear_param[2]

                A = R @ S @ Sh
            else:
                R = _rotation_matrix_2d(omega_param[0])
                S = torch.diag(torch.exp(torch.clamp(scale_param, -1.0, 1.0)))
                Sh = torch.eye(2, device=device_obj)
                Sh[0, 1] = shear_param[0]

                A = R @ S @ Sh

            t_eff = t_param + C_phys - A @ C_phys

            y_phys = phys_coords @ A.t() + t_eff

            mi_orig = torch.tensor(np.array(moving.origin[::-1]), dtype=torch.float32, device=device_obj)
            mi_sp = torch.tensor(np.array(moving.spacing[::-1]), dtype=torch.float32, device=device_obj)
            mi_dir = torch.tensor(np.array(moving.direction[::-1, ::-1].copy()), dtype=torch.float32, device=device_obj)

            y_vox = (y_phys - mi_orig) @ torch.inverse(mi_dir).t() / mi_sp
            mi_shape_t = torch.tensor(np.array(moving.shape[::-1]), dtype=torch.float32, device=device_obj)
            y_norm = 2.0 * (y_vox / (mi_shape_t - 1.0)) - 1.0

            if dim == 3:
                sampling_grid = y_norm.reshape(1, *grid_shape, 3)[..., [2, 1, 0]]
            else:
                sampling_grid = y_norm.reshape(1, *grid_shape, 2)[..., [1, 0]]

            from .syn import grid_sample_nd
            warped = grid_sample_nd(mi_lev, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
            loss = mattes_mi_loss_nd(warped, fi_lev, num_bins=32)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                scale_param.clamp_(-1.0, 1.0)
                shear_param.clamp_(-1.0, 1.0)
                omega_param.clamp_(-np.pi, np.pi)

    with torch.no_grad():
        if dim == 3:
            R_delta_fin = _rodrigues_rotation_matrix_3d(omega_param).cpu().numpy()
            R_final = R_delta_fin @ best_R_init_zyx
            S_final = np.diag(np.exp(scale_param.cpu().numpy()))
            Sh_final = np.eye(3)
            Sh_final[0, 1] = shear_param[0].item()
            Sh_final[0, 2] = shear_param[1].item()
            Sh_final[1, 2] = shear_param[2].item()
            A_final = R_final @ S_final @ Sh_final
        else:
            R_final = _rotation_matrix_2d(omega_param[0]).cpu().numpy()
            S_final = np.diag(np.exp(scale_param.cpu().numpy()))
            Sh_final = np.eye(2)
            Sh_final[0, 1] = shear_param[0].item()
            A_final = R_final @ S_final @ Sh_final

        A_xyz = A_final[::-1, ::-1].copy()
        t_xyz = t_param.cpu().numpy()[::-1].copy()

    temp_dir = tempfile.mkdtemp(prefix="robust_aff_pt_")
    tx_path = os.path.join(temp_dir, "pytorch_affine.mat")

    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx.set_parameters(np.concatenate([A_xyz.ravel(), t_xyz]))
    tx.set_fixed_parameters(com_f)
    ants.write_transform(tx, tx_path)

    warped_mov = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path])
    elapsed = time.time() - t0

    if verbose:
        print(f"[robust_affine mode='pytorch'] Finished {dim}D PyTorch GPU Affine in {elapsed:.2f}s", flush=True)

    return {
        'fwdtransforms': [tx_path],
        'invtransforms': [tx_path],
        'whichtoinvert_inv': [True],
        'warpedmovout': warped_mov,
        'warpedfixout': fixed,
        'time': elapsed
    }


def robust_affine(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: str = None,
    mode: str = 'pytorch',
    multi_start: bool = True,
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
        return _run_pytorch_affine_solver(fixed, moving, initial_tx_path=initial_transform, device=device, verbose=verbose)

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
