"""
Canonical deep feature space similarity losses.

Applies frozen pretrained feature extractors (VGG19, DINOv2, ResNet10, SwinUNETR)
to moving/fixed image pairs and computes negative LNCC on their feature maps.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .cross_correlation import local_ncc_loss_nd


def _call_lncc(
    f_in: torch.Tensor,
    f_tg: torch.Tensor,
    mask: torch.Tensor | None = None,
    window_size: int = 9,
) -> torch.Tensor:
    """Evaluate LNCC, respecting monkeypatching on syntx.syn if present."""
    import sys

    # Feature maps generally have different spatial dimensions than full-res input mask
    if mask is not None and mask.shape[2:] != f_in.shape[2:]:
        mask = None

    syn_mod = sys.modules.get("syntx.syn")
    if syn_mod is not None:
        fn = getattr(syn_mod, "local_ncc_loss_nd", None)
        if fn is not None and fn is not local_ncc_loss_nd:
            try:
                return fn(f_in, f_tg, window_size=window_size)
            except TypeError:
                return fn(f_in, f_tg, mask=mask, window_size=window_size)
    return local_ncc_loss_nd(f_in, f_tg, mask=mask, window_size=window_size)


class FeatureSpaceLoss(nn.Module):
    """
    Negative LNCC between the feature maps of a moving and a fixed image.

    The loss is ``sum over feature maps of local_ncc_loss_nd(f_moving, f_fixed)``.
    How the maps are obtained depends on the extractor and the input dimension:
    - 3-D extractor, 3-D input: features of the whole volume.
    - 2-D extractor, 2-D input: features of the image.
    - 2-D extractor, 3-D input, ``mode='lncc_3d'``: every interior slice along each axis is
      encoded, stacked into feature volumes and evaluated with 3-D LNCC.
    - 2-D extractor, 3-D input, ``mode='triplanar'`` (alias 'lncc'): orthogonal slice batching.

    Parameters
    ----------
    extractor : FeatureExtractor or Any
        Frozen feature extractor module.
    mode : {'lncc_3d', 'triplanar', 'lncc'}, default 'lncc_3d'
    num_slices : int, default 4
    lncc_window : int, default 9
    """

    MODES = ("lncc_3d", "triplanar", "lncc")

    def __init__(
        self,
        extractor: Any,
        mode: str = "lncc_3d",
        num_slices: int = 4,
        lncc_window: int = 9,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"FeatureSpaceLoss: unknown mode {mode!r}; expected one of {self.MODES}")
        self.extractor = extractor
        self.mode = mode
        self.num_slices = num_slices
        self.lncc_window = lncc_window

    def forward(
        self,
        input_nd: torch.Tensor,
        target_nd: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Feature-space LNCC loss between a warped moving image and the fixed image.
        """
        dim = len(input_nd.shape) - 2

        if getattr(self.extractor, "is_3d", False):
            if dim == 2:
                raise ValueError("Cannot run 3D feature extractor on 2D input.")
            return self._forward_3d(input_nd, target_nd, mask=mask)
        else:
            if dim == 2:
                return self._forward_2d_direct(input_nd, target_nd, mask=mask)
            else:
                if self.mode == "lncc_3d":
                    return self._forward_2d_reconstruct_3d(input_nd, target_nd, mask=mask)
                else:
                    return self._forward_2d_triplanar(input_nd, target_nd, mask=mask)

    def _forward_3d(
        self,
        input_nd: torch.Tensor,
        target_nd: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """3-D extractor on 3-D volumes; sum of 3-D LNCC losses over the requested layers."""
        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))

        loss = 0.0
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss = loss + _call_lncc(f_in, f_tg, mask=mask, window_size=self.lncc_window)
        return loss

    def _forward_2d_direct(
        self,
        input_nd: torch.Tensor,
        target_nd: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """2-D extractor on 2-D images; sum of 2-D LNCCs."""
        in_ch = getattr(self.extractor, "in_channels", 1)
        if in_ch == 3 and input_nd.shape[1] == 1:
            input_nd = input_nd.repeat(1, 3, 1, 1)
            target_nd = target_nd.repeat(1, 3, 1, 1)

        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))

        loss = 0.0
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss = loss + _call_lncc(f_in, f_tg, mask=mask, window_size=self.lncc_window)
        return loss

    def _forward_2d_triplanar(
        self,
        input_nd: torch.Tensor,
        target_nd: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """2-D extractor on orthogonal slices (triplanar path)."""
        D, H, W = input_nd.shape[2:]
        device = input_nd.device
        in_ch = getattr(self.extractor, "in_channels", 1)

        z_indices = torch.linspace(D // 4, 3 * D // 4, self.num_slices, dtype=torch.long, device=device)
        y_indices = torch.linspace(H // 4, 3 * H // 4, self.num_slices, dtype=torch.long, device=device)
        x_indices = torch.linspace(W // 4, 3 * W // 4, self.num_slices, dtype=torch.long, device=device)
        if in_ch == 3:
            if min(D, H, W) < 3:
                raise ValueError("FeatureSpaceLoss: a 3-channel extractor needs >= 3 slices per axis")
            z_indices = z_indices.clamp(1, D - 2)
            y_indices = y_indices.clamp(1, H - 2)
            x_indices = x_indices.clamp(1, W - 2)

        target_size = max(D, H, W)
        slices_in = []
        slices_tg = []

        # Axial
        for z in z_indices:
            if in_ch == 3:
                slice_in = input_nd[:, 0, z - 1 : z + 2]
                slice_tg = target_nd[:, 0, z - 1 : z + 2]
            else:
                slice_in = input_nd[:, :, z]
                slice_tg = target_nd[:, :, z]

            if H != target_size or W != target_size:
                slice_in = F.interpolate(
                    slice_in, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
                slice_tg = F.interpolate(
                    slice_tg, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        # Coronal
        for y in y_indices:
            if in_ch == 3:
                slice_in = input_nd[:, 0, :, y - 1 : y + 2, :].movedim(2, 1)
                slice_tg = target_nd[:, 0, :, y - 1 : y + 2, :].movedim(2, 1)
            else:
                slice_in = input_nd[:, :, :, y, :]
                slice_tg = target_nd[:, :, :, y, :]

            if D != target_size or W != target_size:
                slice_in = F.interpolate(
                    slice_in, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
                slice_tg = F.interpolate(
                    slice_tg, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        # Sagittal
        for xi in x_indices:
            if in_ch == 3:
                slice_in = input_nd[:, 0, :, :, xi - 1 : xi + 2].movedim(3, 1)
                slice_tg = target_nd[:, 0, :, :, xi - 1 : xi + 2].movedim(3, 1)
            else:
                slice_in = input_nd[:, :, :, :, xi]
                slice_tg = target_nd[:, :, :, :, xi]

            if D != target_size or H != target_size:
                slice_in = F.interpolate(
                    slice_in, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
                slice_tg = F.interpolate(
                    slice_tg, size=(target_size, target_size), mode="bilinear", align_corners=True
                )
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        input_batch = torch.cat(slices_in, dim=0)
        target_batch = torch.cat(slices_tg, dim=0)

        feats_in = self.extractor.extract(self.extractor.normalize(input_batch))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_batch))

        loss = 0.0
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss = loss + _call_lncc(f_in, f_tg, mask=mask, window_size=self.lncc_window)
        return loss

    def _forward_2d_reconstruct_3d(
        self,
        input_nd: torch.Tensor,
        target_nd: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """2-D extractor on every interior slice along z, y and x of a 3-D volume ('lncc_3d')."""
        D, H, W = input_nd.shape[2:]
        B = input_nd.shape[0]
        in_ch = getattr(self.extractor, "in_channels", 1)

        def reconstruct_3d_features(x):
            slices_ax = []
            for z in range(1, D - 1):
                if in_ch == 3:
                    slices_ax.append(x[:, 0, z - 1 : z + 2])
                else:
                    slices_ax.append(x[:, :, z])
            batch_ax = self.extractor.normalize(torch.cat(slices_ax, dim=0))

            slices_co = []
            for y in range(1, H - 1):
                if in_ch == 3:
                    slices_co.append(x[:, 0, :, y - 1 : y + 2, :].movedim(2, 1))
                else:
                    slices_co.append(x[:, :, :, y, :])
            batch_co = self.extractor.normalize(torch.cat(slices_co, dim=0))

            slices_sa = []
            for xi in range(1, W - 1):
                if in_ch == 3:
                    slices_sa.append(x[:, 0, :, :, xi - 1 : xi + 2].movedim(3, 1))
                else:
                    slices_sa.append(x[:, :, :, :, xi])
            batch_sa = self.extractor.normalize(torch.cat(slices_sa, dim=0))

            vols = []
            for feat_ax, feat_co, feat_sa in zip(
                self.extractor.extract(batch_ax),
                self.extractor.extract(batch_co),
                self.extractor.extract(batch_sa),
            ):
                vols.append(
                    (
                        feat_ax.view(D - 2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4),
                        feat_co.view(H - 2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4),
                        feat_sa.view(W - 2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0),
                    )
                )
            return vols

        loss = 0.0
        for vols_in, vols_tg in zip(reconstruct_3d_features(input_nd), reconstruct_3d_features(target_nd)):
            for v_in, v_tg in zip(vols_in, vols_tg):
                loss = loss + _call_lncc(v_in, v_tg, mask=mask, window_size=self.lncc_window)
        return loss
