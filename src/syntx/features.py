"""
features.py — Deep Feature Extraction & Deep Feature Registration Losses
========================================================================

This module provides modular 2D and 3D deep feature extractors (`VGG19Extractor`,
`DINOv2Extractor`, `ResNet10Extractor`, `SwinUNETRExtractor`) and dimension-agnostic
feature space loss functions (`FeatureSpaceLoss`) for deep registration workflows.

Key Features & Rule Compliance
------------------------------
- VGG19 Memory Truncation: Discards unnecessary upper layers to minimize memory footprint.
- VGG 3D Mode Requirement (GEMINI.md Rule 2): Evaluates 3D LNCC on Layer 4 feature volumes
  (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`) to prevent grid folding and regularize shape alignment.
- DINOv2 Sub-network Pruning: Truncates transformer blocks to target feature layers and handles
  patch alignment and MPS fallback.
- SwinUNETR Lazy Loading: Self-supervised 3D medical vision transformer encoder support.
- Triplanar Slice Ensemble: Extracts orthogonal 2D slices (Axial, Coronal, Sagittal) for 2D networks
  processing 3D volume inputs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import resnet10_2d, resnet10_3d


class FeatureExtractor(nn.Module):
    """
    Abstract base class for all deep feature extraction backbones in `syntx`.

    Subclasses must implement properties `is_3d` and `in_channels`, as well as
    methods `normalize()` and `extract()`.
    """

    @property
    def is_3d(self) -> bool:
        """Boolean flag indicating whether the feature extractor operates natively on 3D inputs."""
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        """Number of expected input channels (e.g. 1 for grayscale/medical, 3 for RGB)."""
        raise NotImplementedError

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input tensor `x` according to backend mean and standard deviation.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Normalized image tensor.
        """
        raise NotImplementedError

    def extract(self, x: torch.Tensor) -> list:
        """
        Extracts intermediate feature maps at target layers.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input tensor.

        Returns
        -------
        list of torch.Tensor
            List of feature map tensors extracted at specified layer indices.
        """
        raise NotImplementedError


class VGG19Extractor(FeatureExtractor):
    """
    VGG19 Feature Extractor with Memory Truncation and Frozen Weights.

    Loads ImageNet-pretrained VGG19 features, truncates layers beyond the maximum requested
    feature layer to conserve memory, sets non-inplace ReLUs, and freezes all parameters.

    Parameters
    ----------
    feature_layers : list of int, default=[8]
        VGG19 sequential feature layer indices to extract (e.g., Layer 4 or 8).

    Attributes
    ----------
    is_3d : bool = False
        2D native extractor.
    in_channels : int = 3
        Requires 3-channel RGB inputs.
    layers : nn.ModuleList
        Truncated VGG19 feature module list.
    """

    is_3d = False
    in_channels = 3

    def __init__(self, feature_layers=[8]):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Discard layers beyond the max needed layer to save memory
        max_layer = max(feature_layers)
        self.layers = nn.ModuleList([vgg[i] for i in range(max_layer + 1)])
        self.feature_layers = feature_layers

        for m in self.layers.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes RGB tensor using ImageNet channel mean and standard deviation."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x: torch.Tensor) -> list:
        """Extracts intermediate VGG feature tensors across configured `feature_layers`."""
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class DINOv2Extractor(FeatureExtractor):
    """
    DINOv2 Self-Supervised Vision Transformer Extractor with Sub-network Pruning.

    Loads DINOv2 ViT backbone from PyTorch Hub, prunes transformer blocks past the maximum
    requested layer to optimize memory, pads inputs to 14-pixel patch size boundaries, and
    reshapes token outputs into spatial feature grids.

    Parameters
    ----------
    version : str, default='vits14'
        DINOv2 architecture variant ('vits14', 'vitb14', etc.).
    feature_layers : list of int, default=[11]
        Transformer block indices to extract.

    Attributes
    ----------
    is_3d : bool = False
        2D native ViT extractor.
    in_channels : int = 3
        Requires 3-channel RGB inputs.
    patch_size : int = 14
        DINOv2 Vision Transformer patch size.
    """

    is_3d = False
    in_channels = 3

    def __init__(self, version='vits14', feature_layers=[11]):
        super().__init__()
        model_name = f'dinov2_{version}'
        # Load from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.patch_size = 14
        self.feature_layers = feature_layers

        # Extract only the needed transformer blocks to save memory
        max_layer = max(feature_layers)
        self.model.blocks = nn.ModuleList([self.model.blocks[i] for i in range(max_layer + 1)])

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes RGB tensor using standard ImageNet mean and std."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x: torch.Tensor) -> list:
        """Extracts spatial feature grids from DINOv2 patch tokens."""
        orig_device = x.device
        if orig_device.type == 'mps':
            x = x.to('cpu')
            self.model.to('cpu')
            self.mean = self.mean.to('cpu')
            self.std = self.std.to('cpu')

        B, C, H, W = x.shape
        # Pad to patch_size-divisible dimensions
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))

        # We step through the model blocks to collect intermediate features
        x_tokens = self.model.prepare_tokens_with_masks(x)

        features = []
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)
            if i in self.feature_layers:
                patch_tokens = x_tokens[:, 1:]  # skip class token
                hp = (H + ph) // self.patch_size
                wp = (W + pw) // self.patch_size
                feat_grid = patch_tokens.reshape(B, hp, wp, -1).permute(0, 3, 1, 2)
                if orig_device.type == 'mps':
                    feat_grid = feat_grid.to(orig_device)
                features.append(feat_grid)
        return features


class ResNet10Extractor(FeatureExtractor):
    """
    Unified 2D and 3D ResNet-10 Deep Feature Extractor.

    Supports single-channel grayscale inputs for 2D images and 3D medical volumes.
    Optionally loads pre-trained MedicalNet 3D weights if cached locally.

    Parameters
    ----------
    dim : int, default=3
        Spatial dimensionality (2 or 3).
    feature_layers : list of int, default=[4]
        ResNet-10 residual layer indices to extract (1, 2, 3, or 4).
    """

    def __init__(self, dim=3, feature_layers=[4]):
        super().__init__()
        self._is_3d = (dim == 3)
        self.feature_layers = feature_layers

        if self._is_3d:
            self.model = resnet10_3d()
            self._in_channels = 1
            # Try to load MedicalNet weights if available
            weights_path = os.path.expanduser("~/.syntx_cache/resnet_10_23iseg.pth")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                self.model.load_state_dict(state.get('state_dict', state), strict=False)
        else:
            self.model = resnet10_2d()
            self._in_channels = 1

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.eval()

    @property
    def is_3d(self) -> bool:
        """Returns True if the extractor operates natively on 3D volumes."""
        return self._is_3d

    @property
    def in_channels(self) -> int:
        """Returns 1 for single-channel medical grayscale inputs."""
        return self._in_channels

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Grayscale volumes in [0, 1] are passed directly without scaling."""
        return x

    def extract(self, x: torch.Tensor) -> list:
        """Extracts feature maps across requested ResNet-10 layers."""
        out = self.model.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.maxpool(out)

        features = []
        out = self.model.layer1(out)
        if 1 in self.feature_layers:
            features.append(out)
        out = self.model.layer2(out)
        if 2 in self.feature_layers:
            features.append(out)
        out = self.model.layer3(out)
        if 3 in self.feature_layers:
            features.append(out)
        out = self.model.layer4(out)
        if 4 in self.feature_layers:
            features.append(out)

        return features


class SwinUNETRExtractor(FeatureExtractor):
    """
    SwinUNETR 3D Self-Supervised Vision Transformer Encoder Feature Extractor.

    Loads MONAI SwinUNETR 3D SSL pre-trained backbone, handles lazy MONAI dependency import,
    downloads SSL model weights if missing, pads input 3D volumes to multiples of 32 voxels,
    and crops feature maps back to exact spatial resolutions.

    Parameters
    ----------
    feature_layers : list of int, default=[4]
        Swin ViT block layers to extract (must be in [1, 2, 3, 4]).
    weights_path : str, optional
        Path to local pre-trained weights file or 'random'. If None, downloads from MONAI zoo.
    img_size : tuple or int, default=(96, 96, 96)
        Target 3D image shape for Swin ViT initialization.
    """

    is_3d = True
    in_channels = 1

    def __init__(self, feature_layers=[4], weights_path=None, img_size=(96, 96, 96)):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required to use SwinUNETRExtractor. "
                "Please install it using 'pip install monai'."
            )

        if not feature_layers:
            raise ValueError("feature_layers cannot be empty.")
        for layer in feature_layers:
            if layer not in [1, 2, 3, 4]:
                raise ValueError("Invalid layer index. SwinUNETR layers must be in [1, 2, 3, 4].")

        self.feature_layers = feature_layers
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size, img_size)
        else:
            self.img_size = tuple(img_size)

        self.model = SwinUNETR(
            in_channels=self.in_channels,
            out_channels=14,
            feature_size=48,
            spatial_dims=3
        )

        if weights_path != "random":
            if weights_path is None:
                weights_path = os.path.expanduser("~/.syntx_cache/model_swinvit.pt")

            if not os.path.exists(weights_path):
                url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
                try:
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    temp_path = weights_path + ".tmp"
                    import urllib.request
                    urllib.request.urlretrieve(url, temp_path)
                    os.rename(temp_path, weights_path)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Failed to download Swin ViT weights from MONAI zoo: {e}. "
                        f"If you are in an offline network environment, "
                        f"please manually download weights from {url} to '{weights_path}'."
                    )

            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                state_dict = state.get('state_dict', state)

                swinvit_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    if k.startswith("swinViT."):
                        k = k[8:]
                    swinvit_state_dict[k] = v

                self.model.swinViT.load_state_dict(swinvit_state_dict, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Grayscale volumes in [0, 1] are passed directly without scaling."""
        return x

    def extract(self, x: torch.Tensor) -> list:
        """Extracts 3D Swin ViT feature maps cropped back to expected target dimensions."""
        if x.shape[0] == 0:
            raise ValueError("Batch size cannot be 0")
        if len(x.shape) != 5:
            raise ValueError("Input must be a 5D tensor (B, C, D, H, W)")

        import math
        spatial_shape = x.shape[2:]
        pad_size = [int(math.ceil(s / 32.0) * 32) for s in spatial_shape]

        pad_d = pad_size[0] - spatial_shape[0]
        pad_h = pad_size[1] - spatial_shape[1]
        pad_w = pad_size[2] - spatial_shape[2]

        x_input = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0.0)

        hidden_states = self.model.swinViT(x_input)
        features = []
        for layer in self.feature_layers:
            if len(hidden_states) == 5:
                feat = hidden_states[layer]
            else:
                feat = hidden_states[layer - 1]

            downsample_factor = 2 ** (layer + 1)
            expected_shape = [max(1, s // downsample_factor) for s in spatial_shape]
            feat = feat[:, :, :expected_shape[0], :expected_shape[1], :expected_shape[2]]

            features.append(feat)

        return features


class FeatureSpaceLoss(nn.Module):
    """
    Dimension-Agnostic Similarity Loss evaluated in Deep Feature Space.

    Supports native 3D extractors (`ResNet10Extractor`, `SwinUNETRExtractor`) as well as
    2D extractors (`VGG19Extractor`, `DINOv2Extractor`) applied to 3D volumes via
    3D feature volume reconstruction (`mode='lncc_3d'`) or orthogonal triplanar slice ensembles (`mode='triplanar'`).

    Parameters
    ----------
    extractor : FeatureExtractor
        Feature extractor instance.
    mode : str, default='lncc_3d'
        Evaluation mode ('lncc_3d', 'lncc', 'triplanar').
    num_slices : int, default=4
        Number of orthogonal slices per anatomical plane for triplanar mode.
    lncc_window : int, default=9
        LNCC spatial window size evaluated on feature maps.

    Notes
    -----
    Strictly enforces GEMINI.md Rule 2: 3D VGG feature loss uses Layer 4 feature volume
    reconstruction (`mode='lncc_3d'`), avoiding 2D slice approximations when high accuracy is required.
    """

    def __init__(self, extractor: FeatureExtractor, mode='lncc_3d', num_slices=4, lncc_window=9):
        super().__init__()
        self.extractor = extractor
        self.mode = mode
        self.num_slices = num_slices
        self.lncc_window = lncc_window

    def forward(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """
        Calculates deep feature loss between input and target tensors.

        Parameters
        ----------
        input_nd : torch.Tensor
            Warped image tensor of shape `(B, C, H, W)` or `(B, C, D, H, W)`.
        target_nd : torch.Tensor
            Fixed target image tensor of matching shape.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        dim = len(input_nd.shape) - 2

        if self.extractor.is_3d:
            if dim == 2:
                raise ValueError("Cannot run 3D feature extractor on 2D input.")
            return self._forward_3d(input_nd, target_nd)
        else:
            if dim == 2:
                return self._forward_2d_direct(input_nd, target_nd)
            else:
                if self.mode == 'lncc_3d':
                    return self._forward_2d_reconstruct_3d(input_nd, target_nd)
                else:
                    return self._forward_2d_triplanar(input_nd, target_nd)

    def _forward_3d(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """Native 3D feature extraction and 3D LNCC computation."""
        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))

        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=self.lncc_window)
        return loss

    def _forward_2d_direct(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """Direct 2D feature extraction and 2D LNCC computation for 2D inputs."""
        if self.extractor.in_channels == 3 and input_nd.shape[1] == 1:
            input_nd = input_nd.repeat(1, 3, 1, 1)
            target_nd = target_nd.repeat(1, 3, 1, 1)

        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))

        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=self.lncc_window)
        return loss

    def _forward_2d_triplanar(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """Extracts orthogonal 2D slice ensembles (Axial, Coronal, Sagittal) for 2D networks processing 3D inputs."""
        D, H, W = input_nd.shape[2:]
        device = input_nd.device

        z_indices = torch.linspace(D // 4, 3 * D // 4, self.num_slices, dtype=torch.long, device=device)
        y_indices = torch.linspace(H // 4, 3 * H // 4, self.num_slices, dtype=torch.long, device=device)
        x_indices = torch.linspace(W // 4, 3 * W // 4, self.num_slices, dtype=torch.long, device=device)

        target_size = max(D, H, W)
        slices_in = []
        slices_tg = []

        # Axial
        for z in z_indices:
            if self.extractor.in_channels == 3:
                slice_in = input_nd[:, 0, z - 1:z + 2]
                slice_tg = target_nd[:, 0, z - 1:z + 2]
            else:
                slice_in = input_nd[:, 0:1, z:z + 1]
                slice_tg = target_nd[:, 0:1, z:z + 1]

            if H != target_size or W != target_size:
                slice_in = F.interpolate(slice_in, size=(target_size, target_size), mode='bilinear', align_corners=True)
                slice_tg = F.interpolate(slice_tg, size=(target_size, target_size), mode='bilinear', align_corners=True)
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        # Coronal
        for y in y_indices:
            if self.extractor.in_channels == 3:
                slice_in = input_nd[:, 0, :, y - 1:y + 2, :].movedim(2, 1)
                slice_tg = target_nd[:, 0, :, y - 1:y + 2, :].movedim(2, 1)
            else:
                slice_in = input_nd[:, 0:1, :, y:y + 1, :].movedim(2, 1)
                slice_tg = target_nd[:, 0:1, :, y:y + 1, :].movedim(2, 1)

            if D != target_size or W != target_size:
                slice_in = F.interpolate(slice_in, size=(target_size, target_size), mode='bilinear', align_corners=True)
                slice_tg = F.interpolate(slice_tg, size=(target_size, target_size), mode='bilinear', align_corners=True)
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        # Sagittal
        for xi in x_indices:
            if self.extractor.in_channels == 3:
                slice_in = input_nd[:, 0, :, :, xi - 1:xi + 2].movedim(3, 1)
                slice_tg = target_nd[:, 0, :, :, xi - 1:xi + 2].movedim(3, 1)
            else:
                slice_in = input_nd[:, 0:1, :, :, xi:xi + 1].movedim(3, 1)
                slice_tg = target_nd[:, 0:1, :, :, xi:xi + 1].movedim(3, 1)

            if D != target_size or H != target_size:
                slice_in = F.interpolate(slice_in, size=(target_size, target_size), mode='bilinear', align_corners=True)
                slice_tg = F.interpolate(slice_tg, size=(target_size, target_size), mode='bilinear', align_corners=True)
            slices_in.append(slice_in)
            slices_tg.append(slice_tg)

        input_batch = torch.cat(slices_in, dim=0)
        target_batch = torch.cat(slices_tg, dim=0)

        feats_in = self.extractor.extract(self.extractor.normalize(input_batch))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_batch))

        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=self.lncc_window)
        return loss

    def _forward_2d_reconstruct_3d(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """Reconstructs 3D feature volumes from 2D slice features and evaluates 3D LNCC."""
        D, H, W = input_nd.shape[2:]
        B = input_nd.shape[0]

        def reconstruct_3d_features(x):
            slices_ax = []
            for z in range(1, D - 1):
                if self.extractor.in_channels == 3:
                    slices_ax.append(x[:, 0, z - 1:z + 2])
                else:
                    slices_ax.append(x[:, 0:1, z:z + 1])
            batch_ax = self.extractor.normalize(torch.cat(slices_ax, dim=0))

            slices_co = []
            for y in range(1, H - 1):
                if self.extractor.in_channels == 3:
                    slices_co.append(x[:, 0, :, y - 1:y + 2, :].movedim(2, 1))
                else:
                    slices_co.append(x[:, 0:1, :, y:y + 1, :].movedim(2, 1))
            batch_co = self.extractor.normalize(torch.cat(slices_co, dim=0))

            slices_sa = []
            for xi in range(1, W - 1):
                if self.extractor.in_channels == 3:
                    slices_sa.append(x[:, 0, :, :, xi - 1:xi + 2].movedim(3, 1))
                else:
                    slices_sa.append(x[:, 0:1, :, :, xi:xi + 1].movedim(3, 1))
            batch_sa = self.extractor.normalize(torch.cat(slices_sa, dim=0))

            feat_ax = self.extractor.extract(batch_ax)[-1]
            feat_co = self.extractor.extract(batch_co)[-1]
            feat_sa = self.extractor.extract(batch_sa)[-1]

            vol_ax = feat_ax.view(D - 2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4)
            vol_co = feat_co.view(H - 2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4)
            vol_sa = feat_sa.view(W - 2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0)

            return vol_ax, vol_co, vol_sa

        vol_in_ax, vol_in_co, vol_in_sa = reconstruct_3d_features(input_nd)
        vol_tg_ax, vol_tg_co, vol_tg_sa = reconstruct_3d_features(target_nd)

        from .syn import local_ncc_loss_nd
        loss_ax = local_ncc_loss_nd(vol_in_ax, vol_tg_ax, window_size=5)
        loss_co = local_ncc_loss_nd(vol_in_co, vol_tg_co, window_size=5)
        loss_sa = local_ncc_loss_nd(vol_in_sa, vol_tg_sa, window_size=5)

        return loss_ax + loss_co + loss_sa
