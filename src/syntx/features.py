"""
Frozen pretrained feature extractors and an LNCC loss computed on their feature maps.

Extractors (all parameters frozen, ``requires_grad=False``; gradients still flow to the input):

- ``VGG19Extractor``: torchvision ImageNet VGG19 ``features``, truncated after the last
  requested layer. 2-D, 3-channel input.
- ``DINOv2Extractor``: DINOv2 ViT from ``torch.hub`` (network download on first use),
  transformer blocks truncated after the last requested block. 2-D, 3-channel input.
- ``ResNet10Extractor``: ``syntx.resnet`` ResNet-10, 2-D or 3-D, 1-channel input. Random
  weights unless a MedicalNet checkpoint is found at ``~/.syntx_cache/resnet_10_23iseg.pth``
  (3-D only).
- ``SwinUNETRExtractor``: MONAI SwinUNETR Swin-ViT encoder, 3-D, 1-channel input; downloads the
  MONAI self-supervised weights to ``~/.syntx_cache/model_swinvit.pt`` if missing.

``FeatureSpaceLoss`` applies an extractor to a moving / fixed pair and sums the negative LNCC
(``syntx.core.losses.local_ncc_loss_nd``) of the feature maps. 2-D extractors are applied to
3-D volumes either slice-by-slice along all three axes with the slice features stacked back into
volumes (``mode='lncc_3d'``) or on a few orthogonal slices (any other mode, "triplanar").
``syntx.syn_jax`` builds these losses for metric names such as 'vgg_4_lncc', 'dino_2_lncc',
'resnet10' and 'swinunetr'.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import resnet10_2d, resnet10_3d


class FeatureExtractor(nn.Module):
    """
    Interface used by ``FeatureSpaceLoss``.

    Subclasses provide ``is_3d``, ``in_channels``, ``normalize()`` and ``extract()``. The base
    methods raise NotImplementedError.
    """

    @property
    def is_3d(self) -> bool:
        """True if the network takes 3-D input ``(B, C, D, H, W)``, False for ``(B, C, H, W)``."""
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        """Number of input channels the network expects (1 or 3)."""
        raise NotImplementedError

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map raw intensities to the network's expected input scaling.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor ``(B, C, *spatial)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape.
        """
        raise NotImplementedError

    def extract(self, x: torch.Tensor) -> list:
        """
        Run the network and collect the feature maps of the configured layers.

        Parameters
        ----------
        x : torch.Tensor
            Normalised input ``(B, in_channels, *spatial)``.

        Returns
        -------
        list of torch.Tensor
            One feature map ``(B, C_feat, *spatial_feat)`` per requested layer, in network order.
        """
        raise NotImplementedError


class VGG19Extractor(FeatureExtractor):
    """
    Frozen ImageNet VGG19 (torchvision ``VGG19_Weights.DEFAULT``) feature extractor, 2-D.

    Keeps ``vgg19().features[0 : max(feature_layers) + 1]``, makes every ReLU non-inplace (so the
    stored outputs are not overwritten), freezes the weights and puts the module in eval mode.
    Downloads the torchvision weights on first use.

    Parameters
    ----------
    feature_layers : list of int, default [8]
        Indices into ``vgg19().features`` whose outputs are returned. For example 4 is the first
        max-pool (64 channels, 1/2 resolution) and 8 is relu2_2 (128 channels, 1/2 resolution).

    Attributes
    ----------
    is_3d : False
    in_channels : 3
    layers : nn.ModuleList
        The truncated VGG19 layers.
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
        """Per-channel (x - mean) / std with ImageNet statistics; ``x`` (B, 3, H, W) in [0, 1]."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x: torch.Tensor) -> list:
        """Run ``x`` (B, 3, H, W) through the kept layers; return the ``feature_layers`` outputs."""
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class DINOv2Extractor(FeatureExtractor):
    """
    Frozen DINOv2 ViT feature extractor, 2-D.

    Loads ``torch.hub.load('facebookresearch/dinov2', 'dinov2_' + version)`` (network access /
    hub cache needed) and keeps only blocks ``0 .. max(feature_layers)``.

    Parameters
    ----------
    version : str, default 'vits14'
        Hub model suffix, e.g. 'vits14', 'vitb14', or a register-token variant ('vits14_reg');
        ``extract`` drops the class token and the model's ``num_register_tokens``.
    feature_layers : list of int, default [11]
        Transformer block indices whose patch-token outputs are returned.

    Attributes
    ----------
    is_3d : False
    in_channels : 3
    patch_size : int
        14 (fixed; not read from the model).
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
        """Per-channel (x - mean) / std with ImageNet statistics; ``x`` (B, 3, H, W) in [0, 1]."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x: torch.Tensor) -> list:
        """
        Return the patch tokens of each requested block as a spatial grid.

        ``x`` (B, 3, H, W) is zero-padded on the bottom / right to a multiple of 14; the class
        and register tokens are dropped and the patch tokens are reshaped to ``(B, embed_dim,
        ceil(H/14), ceil(W/14))`` -- every token of that grid overlaps the image (the last
        row / column partly covers padding), and fixed / moving inputs of one shape get the same
        grid. Block outputs are taken before the model's final norm. On an MPS input the
        computation runs on a frozen CPU copy of the model (made once; ``self.model`` stays on its
        device, and the autograd graph -- gradients flow to ``x`` -- keeps its CPU tensors) and the
        outputs are returned on MPS.
        """
        orig_device = x.device
        if orig_device.type == 'mps':
            if getattr(self, '_cpu_model', None) is None:
                import copy
                self._cpu_model = copy.deepcopy(self.model).to('cpu').eval()
                for p_ in self._cpu_model.parameters():
                    p_.requires_grad_(False)
            return [f.to(orig_device) for f in self._extract_cpu(x.to('cpu'), self._cpu_model)]
        return self._extract_cpu(x)

    def _extract_cpu(self, x: torch.Tensor, model=None) -> list:
        """``extract`` with ``model`` (default ``self.model``) on its current device."""
        model = self.model if model is None else model
        B, C, H, W = x.shape
        # Pad to patch_size-divisible dimensions
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))

        # We step through the model blocks to collect intermediate features
        x_tokens = model.prepare_tokens_with_masks(x)

        features = []
        for i, blk in enumerate(model.blocks):
            x_tokens = blk(x_tokens)
            if i in self.feature_layers:
                n_skip = 1 + int(getattr(model, "num_register_tokens", 0) or 0)
                patch_tokens = x_tokens[:, n_skip:]  # skip class + register tokens
                hp = (H + ph) // self.patch_size
                wp = (W + pw) // self.patch_size
                features.append(patch_tokens.reshape(B, hp, wp, -1).permute(0, 3, 1, 2))
        return features


def _load_state_dict_checked(module: nn.Module, state_dict: dict, strip_prefixes=("module.",),
                             rename: dict = None) -> dict:
    """Load a checkpoint into ``module`` by name, reporting what did not match.

    Key prefixes in ``strip_prefixes`` are removed and ``rename`` substrings replaced (e.g.
    MedicalNet ``downsample`` -> ``shortcut``); keys whose tensor shape differs are skipped.
    Raises RuntimeError when no parameter matched (a silent all-random load); warns when some
    did not. Returns ``{'loaded', 'missing', 'unexpected', 'shape_mismatch'}``.
    """
    import warnings
    own = module.state_dict()
    mapped = {}
    for k, v in state_dict.items():
        for pfx in strip_prefixes:
            if k.startswith(pfx):
                k = k[len(pfx):]
        for a, b in (rename or {}).items():
            k = k.replace(a, b)
        mapped[k] = v
    usable = {k: v for k, v in mapped.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    report = {"loaded": len(usable),
              "missing": sorted(k for k in own if k not in usable),
              "unexpected": sorted(k for k in mapped if k not in own),
              "shape_mismatch": sorted(k for k in mapped if k in own and k not in usable)}
    if not usable:
        raise RuntimeError(f"checkpoint matches no parameter of {type(module).__name__} "
                           f"(e.g. checkpoint keys {list(mapped)[:3]})")
    module.load_state_dict(usable, strict=False)
    if report["missing"] or report["shape_mismatch"]:
        warnings.warn(f"{type(module).__name__}: loaded {report['loaded']} tensors; "
                      f"{len(report['missing'])} missing, {len(report['shape_mismatch'])} shape "
                      f"mismatches (e.g. {(report['missing'] + report['shape_mismatch'])[:3]})")
    return report


class ResNet10Extractor(FeatureExtractor):
    """
    Frozen ResNet-10 (``syntx.resnet``) feature extractor, 2-D or 3-D, 1-channel input.

    For ``dim=3``, weights are loaded from ``~/.syntx_cache/resnet_10_23iseg.pth`` (MedicalNet)
    if that file exists, via ``_load_state_dict_checked`` ("module." stripped, ``downsample`` ->
    ``shortcut``; RuntimeError if nothing matches, a warning for partial matches; the report is
    ``self.weights_report``). Otherwise, and always for 2-D (no 2-D checkpoint exists), the
    network keeps its random initialisation and ``self.weights_report`` is None.

    Parameters
    ----------
    dim : int, default 3
        3 builds the 3-D network (``is_3d`` True); any other value builds the 2-D one.
    feature_layers : list of int, default [4]
        Residual stages to return, from {1, 2, 3, 4} (64 / 128 / 256 / 512 channels at 1/4,
        1/8, 1/16, 1/32 resolution). Other values are ignored.
    """

    def __init__(self, dim=3, feature_layers=[4]):
        super().__init__()
        self._is_3d = (dim == 3)
        self.feature_layers = feature_layers

        self.weights_report = None
        if self._is_3d:
            self.model = resnet10_3d()
            self._in_channels = 1
            # MedicalNet weights, if available
            weights_path = os.path.expanduser("~/.syntx_cache/resnet_10_23iseg.pth")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                self.weights_report = _load_state_dict_checked(
                    self.model, state.get('state_dict', state), rename={"downsample": "shortcut"})
        else:
            self.model = resnet10_2d()
            self._in_channels = 1

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.eval()

    @property
    def is_3d(self) -> bool:
        """True when built with ``dim=3``."""
        return self._is_3d

    @property
    def in_channels(self) -> int:
        """Always 1."""
        return self._in_channels

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: the input is used unscaled."""
        return x

    def extract(self, x: torch.Tensor) -> list:
        """Run stem, max-pool and stages 1-4; return the requested stages' outputs in order."""
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
    Frozen MONAI SwinUNETR Swin-ViT encoder feature extractor, 3-D, 1-channel input.

    Builds ``monai.networks.nets.SwinUNETR(in_channels=1, out_channels=14, feature_size=48,
    spatial_dims=3)`` and loads the MONAI self-supervised Swin-ViT weights into its ``swinViT``
    sub-module via ``_load_state_dict_checked`` ("module." / "swinViT." prefixes stripped;
    RuntimeError if nothing matches). Only the encoder is used.

    Parameters
    ----------
    feature_layers : list of int, default [4]
        Encoder outputs to return, each in {1, 2, 3, 4}; output ``k`` has 1 / 2**(k+1) of the
        input resolution. Empty or other values raise ValueError.
    weights_path : str, optional
        'random': keep the random initialisation. None: use ``~/.syntx_cache/model_swinvit.pt``.
        If the file does not exist it is downloaded from the MONAI-extra-test-data release (the
        directory is created); a failed download raises RuntimeError (pass 'random' to run
        without pretrained weights).

    Raises
    ------
    ImportError
        If MONAI is not installed.
    ValueError
        For an empty or invalid ``feature_layers``.
    RuntimeError
        Weights could not be downloaded, or the checkpoint matches no parameter.
    """

    is_3d = True
    in_channels = 1

    def __init__(self, feature_layers=[4], weights_path=None):
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
                    raise RuntimeError(
                        f"Failed to download Swin ViT weights from MONAI zoo: {e}. Download "
                        f"{url} to '{weights_path}' manually, or pass weights_path='random'."
                    ) from e

            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                state_dict = state.get('state_dict', state)

                self.weights_report = _load_state_dict_checked(
                    self.model.swinViT, state_dict, strip_prefixes=("module.", "swinViT."))

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: the input is used unscaled."""
        return x

    def extract(self, x: torch.Tensor) -> list:
        """
        Return the requested Swin-ViT encoder outputs for ``x`` (B, 1, D, H, W).

        The volume is zero-padded at the far end of each axis to a multiple of 32; output ``k``
        is cropped to ``max(1, s // 2**(k+1))`` along each axis ``s`` of the unpadded input.
        Raises ValueError for batch size 0 or a non-5-D input.
        """
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
    Negative LNCC between the feature maps of a moving and a fixed image.

    The loss is ``sum over feature maps of local_ncc_loss_nd(f_moving, f_fixed)`` (each term in
    [-1, 0], -1 = perfectly correlated). How the maps are obtained depends on the extractor and
    the input dimension (see ``forward``):

    - 3-D extractor, 3-D input: features of the whole volume, one term per requested layer.
    - 2-D extractor, 2-D input: features of the image, one term per requested layer.
    - 2-D extractor, 3-D input, ``mode='lncc_3d'``: every interior slice along each axis is
      encoded; for each requested layer the slice features are stacked into three feature
      volumes (one per slicing axis) and a 3-D LNCC (window ``lncc_window``) is taken on each;
      three terms per layer.
    - 2-D extractor, 3-D input, ``mode='triplanar'`` (alias 'lncc'): ``num_slices`` slices per
      axis (between 1/4 and 3/4 of the extent), resized to a square of the largest volume
      dimension, encoded as one batch; 2-D LNCC per requested layer.

    For 3-channel extractors a 1-channel 2-D input is repeated to 3 channels; for 3-D input the
    three channels are three adjacent slices of channel 0.

    Parameters
    ----------
    extractor : FeatureExtractor
        Frozen extractor (see the module docstring).
    mode : {'lncc_3d', 'triplanar', 'lncc'}, default 'lncc_3d'
        Path for a 2-D extractor with a 3-D input ('lncc' = 'triplanar'). Other values raise
        ValueError.
    num_slices : int, default 4
        Slices per axis in the triplanar path only.
    lncc_window : int, default 9
        LNCC window (feature-map voxels) for every path. ``local_ncc_loss_nd`` shrinks it to
        the smallest feature-map dimension if larger.
    """

    MODES = ('lncc_3d', 'triplanar', 'lncc')

    def __init__(self, extractor: FeatureExtractor, mode='lncc_3d', num_slices=4, lncc_window=9):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"FeatureSpaceLoss: unknown mode {mode!r}; expected one of {self.MODES}")
        self.extractor = extractor
        self.mode = mode
        self.num_slices = num_slices
        self.lncc_window = lncc_window

    def forward(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """
        Feature-space LNCC loss between a warped moving image and the fixed image.

        Parameters
        ----------
        input_nd : torch.Tensor
            Warped moving image, ``(B, C, H, W)`` or ``(B, C, D, H, W)``, tensor order.
        target_nd : torch.Tensor
            Fixed image of the same shape.

        Returns
        -------
        torch.Tensor
            Scalar: the sum of the negative-LNCC terms (see the class docstring), so in
            ``[-n_terms, 0]``; lower is better. Differentiable with respect to both inputs.

        Raises
        ------
        ValueError
            A 3-D extractor with a 2-D input.
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
        """3-D extractor on 3-D volumes; sum of 3-D LNCC losses over the requested layers."""
        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))

        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=self.lncc_window)
        return loss

    def _forward_2d_direct(self, input_nd: torch.Tensor, target_nd: torch.Tensor) -> torch.Tensor:
        """2-D extractor on 2-D images (1 channel repeated to 3 if needed); sum of 2-D LNCCs."""
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
        """
        2-D extractor on ``num_slices`` slices per axis of a 3-D volume (triplanar path).

        Slice indices are ``linspace(n // 4, 3n // 4, num_slices)`` along z, y and x (kept in
        [1, n - 2] for 3-channel extractors, which take three adjacent slices); slices are
        resized (bilinear) to ``max(D, H, W)`` squared, concatenated into one batch and compared
        with 2-D LNCC per requested layer.
        """
        D, H, W = input_nd.shape[2:]
        device = input_nd.device

        z_indices = torch.linspace(D // 4, 3 * D // 4, self.num_slices, dtype=torch.long, device=device)
        y_indices = torch.linspace(H // 4, 3 * H // 4, self.num_slices, dtype=torch.long, device=device)
        x_indices = torch.linspace(W // 4, 3 * W // 4, self.num_slices, dtype=torch.long, device=device)
        if self.extractor.in_channels == 3:
            # three adjacent slices per sample: keep the centre in [1, n - 2]
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
            if self.extractor.in_channels == 3:
                slice_in = input_nd[:, 0, z - 1:z + 2]
                slice_tg = target_nd[:, 0, z - 1:z + 2]
            else:
                slice_in = input_nd[:, :, z]
                slice_tg = target_nd[:, :, z]

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
                slice_in = input_nd[:, :, :, y, :]
                slice_tg = target_nd[:, :, :, y, :]

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
                slice_in = input_nd[:, :, :, :, xi]
                slice_tg = target_nd[:, :, :, :, xi]

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
        """
        2-D extractor on every interior slice along z, y and x of a 3-D volume ('lncc_3d').

        For every requested layer the slice features along each axis are stacked into a
        feature volume (full slice count along that axis, network resolution in-plane), giving
        three volumes per image and layer; the loss is the sum of the 3-D LNCC terms (window
        ``lncc_window``). All slices of an axis are encoded in one batch, so memory grows with
        the volume size.
        """
        D, H, W = input_nd.shape[2:]
        B = input_nd.shape[0]

        def reconstruct_3d_features(x):
            slices_ax = []
            for z in range(1, D - 1):
                if self.extractor.in_channels == 3:
                    slices_ax.append(x[:, 0, z - 1:z + 2])
                else:
                    slices_ax.append(x[:, :, z])
            batch_ax = self.extractor.normalize(torch.cat(slices_ax, dim=0))

            slices_co = []
            for y in range(1, H - 1):
                if self.extractor.in_channels == 3:
                    slices_co.append(x[:, 0, :, y - 1:y + 2, :].movedim(2, 1))
                else:
                    slices_co.append(x[:, :, :, y, :])
            batch_co = self.extractor.normalize(torch.cat(slices_co, dim=0))

            slices_sa = []
            for xi in range(1, W - 1):
                if self.extractor.in_channels == 3:
                    slices_sa.append(x[:, 0, :, :, xi - 1:xi + 2].movedim(3, 1))
                else:
                    slices_sa.append(x[:, :, :, :, xi])
            batch_sa = self.extractor.normalize(torch.cat(slices_sa, dim=0))

            vols = []
            for feat_ax, feat_co, feat_sa in zip(self.extractor.extract(batch_ax),
                                                  self.extractor.extract(batch_co),
                                                  self.extractor.extract(batch_sa)):
                vols.append((
                    feat_ax.view(D - 2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4),
                    feat_co.view(H - 2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4),
                    feat_sa.view(W - 2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0),
                ))
            return vols

        from .syn import local_ncc_loss_nd
        loss = 0.0
        for vols_in, vols_tg in zip(reconstruct_3d_features(input_nd), reconstruct_3d_features(target_nd)):
            for v_in, v_tg in zip(vols_in, vols_tg):
                loss = loss + local_ncc_loss_nd(v_in, v_tg, window_size=self.lncc_window)
        return loss
