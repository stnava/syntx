"""
SyNtx I/O Module — Serialization & Deserialization of Registration Results

Provides intuitive, well-documented read and write functions for saving and loading 
syntx registration results (warped images, displacement fields, transforms, and provenance metadata).
"""

import os
import json
from typing import Dict, Any, Optional, Union, List
import ants


def write_registration(
    reg_dict: Dict[str, Any],
    prefix: str,
    save_warp: bool = True,
    format: str = "nii.gz",
    verbose: bool = True
) -> Dict[str, str]:
    """
    Saves a syntx registration output dictionary to disk.

    Parameters:
        reg_dict (dict): Dictionary returned by TVFModel.fit(), TVFModel.registration(), or syntx.syn().
                         Expected keys: 'warpedmovout', 'fwdtransforms', 'invtransforms', 'provenance'.
        prefix (str): File path prefix for output files (e.g. '/path/to/output/sub01_').
        save_warp (bool): Whether to write vector displacement fields to disk. Default: True.
        format (str): Image extension format ('nii.gz' or 'nii'). Default: 'nii.gz'.
        verbose (bool): If True, prints status messages. Default: True.

    Returns:
        dict: Dictionary mapping result keys to saved absolute file paths on disk.

    Example:
        >>> reg = model.fit(fixed, moving)
        >>> saved_paths = syntx.write_registration(reg, prefix='./results/pair12_')
        >>> print(saved_paths['warpedmovout'])
        '/path/to/results/pair12_warpedmovout.nii.gz'
    """
    prefix_dir = os.path.dirname(os.path.abspath(prefix))
    if prefix_dir:
        os.makedirs(prefix_dir, exist_ok=True)

    saved_paths = {}

    # 1. Save Warped Moving Image
    warped_img = reg_dict.get("warpedmovout", None)
    if isinstance(warped_img, ants.ANTsImage):
        w_path = f"{prefix}warpedmovout.{format}"
        ants.image_write(warped_img, w_path)
        saved_paths["warpedmovout"] = os.path.abspath(w_path)

    # 2. Save Inverse Warped Fixed Image (if present)
    warped_fix = reg_dict.get("warpedfixout", None)
    if isinstance(warped_fix, ants.ANTsImage):
        wf_path = f"{prefix}warpedfixout.{format}"
        ants.image_write(warped_fix, wf_path)
        saved_paths["warpedfixout"] = os.path.abspath(wf_path)

    # 3. Save Forward Transforms
    fwd_txs = reg_dict.get("fwdtransforms", [])
    saved_fwd = []
    for idx, tx in enumerate(fwd_txs):
        if isinstance(tx, str) and os.path.exists(tx):
            saved_fwd.append(os.path.abspath(tx))
        elif isinstance(tx, ants.ANTsImage) and save_warp:
            tx_path = f"{prefix}1Warp_{idx}.{format}"
            ants.image_write(tx, tx_path)
            saved_fwd.append(os.path.abspath(tx_path))
    if saved_fwd:
        saved_paths["fwdtransforms"] = saved_fwd

    # 4. Save Inverse Transforms
    inv_txs = reg_dict.get("invtransforms", [])
    saved_inv = []
    for idx, tx in enumerate(inv_txs):
        if isinstance(tx, str) and os.path.exists(tx):
            saved_inv.append(os.path.abspath(tx))
        elif isinstance(tx, ants.ANTsImage) and save_warp:
            tx_path = f"{prefix}1InverseWarp_{idx}.{format}"
            ants.image_write(tx, tx_path)
            saved_inv.append(os.path.abspath(tx_path))
    if saved_inv:
        saved_paths["invtransforms"] = saved_inv

    # 5. Save Provenance Metadata JSON
    provenance = reg_dict.get("provenance", {})
    prov_path = f"{prefix}provenance.json"
    try:
        with open(prov_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)
        saved_paths["provenance"] = os.path.abspath(prov_path)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not save provenance JSON to {prov_path}: {e}")

    if verbose:
        print(f"Successfully saved syntx registration results with prefix '{prefix}':")
        for k, v in saved_paths.items():
            print(f"  - {k}: {v}")

    return saved_paths


def read_registration(prefix: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Reads previously saved syntx registration results from disk.

    Parameters:
        prefix (str): File path prefix used when saving (e.g. '/path/to/output/sub01_').
        verbose (bool): If True, prints loaded components. Default: True.

    Returns:
        dict: Registration dictionary with 'warpedmovout', 'fwdtransforms', 'invtransforms', and 'provenance'.

    Example:
        >>> reg = syntx.read_registration('./results/pair12_')
        >>> warped_img = reg['warpedmovout']
    """
    reg_dict = {}

    # 1. Load Warped Moving Image
    for ext in ["nii.gz", "nii"]:
        w_path = f"{prefix}warpedmovout.{ext}"
        if os.path.exists(w_path):
            reg_dict["warpedmovout"] = ants.image_read(w_path)
            break

    # 2. Load Inverse Warped Fixed Image
    for ext in ["nii.gz", "nii"]:
        wf_path = f"{prefix}warpedfixout.{ext}"
        if os.path.exists(wf_path):
            reg_dict["warpedfixout"] = ants.image_read(wf_path)
            break

    # 3. Locate Forward Transforms
    fwd_txs = []
    # Check numbered warp files or generic affine
    for ext in ["nii.gz", "nii"]:
        idx = 0
        while True:
            warp_p = f"{prefix}1Warp_{idx}.{ext}"
            if os.path.exists(warp_p):
                fwd_txs.append(os.path.abspath(warp_p))
                idx += 1
            else:
                break
    generic_aff = f"{prefix}0GenericAffine.mat"
    if os.path.exists(generic_aff):
        fwd_txs.append(os.path.abspath(generic_aff))
    reg_dict["fwdtransforms"] = fwd_txs

    # 4. Locate Inverse Transforms
    inv_txs = []
    for ext in ["nii.gz", "nii"]:
        idx = 0
        while True:
            warp_p = f"{prefix}1InverseWarp_{idx}.{ext}"
            if os.path.exists(warp_p):
                inv_txs.append(os.path.abspath(warp_p))
                idx += 1
            else:
                break
    if os.path.exists(generic_aff):
        inv_txs.append(os.path.abspath(generic_aff))
    reg_dict["invtransforms"] = inv_txs

    # 5. Load Provenance JSON
    prov_path = f"{prefix}provenance.json"
    if os.path.exists(prov_path):
        try:
            with open(prov_path, "r") as f:
                reg_dict["provenance"] = json.load(f)
        except Exception:
            reg_dict["provenance"] = {}

    if verbose:
        print(f"Loaded syntx registration dictionary from prefix '{prefix}':")
        if "warpedmovout" in reg_dict:
            print(f"  - warpedmovout: {reg_dict['warpedmovout']}")
        print(f"  - fwdtransforms: {reg_dict['fwdtransforms']}")
        print(f"  - invtransforms: {reg_dict['invtransforms']}")

    return reg_dict
