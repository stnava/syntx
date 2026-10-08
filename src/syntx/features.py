"""
Frozen pretrained feature extractors and an LNCC loss computed on their feature maps.

Extractors (all parameters frozen, ``requires_grad=False``; gradients still flow to the input):

- ``VGG19Extractor``: torchvision ImageNet VGG19 ``features``, truncated after the last
  requested layer. 2-D, 3-channel input.
- ``DINOv2Extractor``: DINOv2 ViT from ``torch.hub`` (network download on first use),
  transformer blocks truncated after the last requested block. 2-D, 3-channel input.
- ``ResNet10Extractor``: ``syntx.resnet`` ResNet-10, 2-D or 3-D, 1-channel input. Random
  weights only when asked (``weights_path="random"``); the MedicalNet checkpoint comes
  from antsxdata (``syntx_features/resnet_10_23iseg``) once it is registered
  (3-D only).
- ``SwinUNETRExtractor``: MONAI SwinUNETR Swin-ViT encoder, 3-D, 1-channel input; downloads the
  MONAI self-supervised weights through antsxdata (``syntx_features/model_swinvit``).

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

    For ``dim=3``, weights come from antsxdata ``syntx_features/resnet_10_23iseg`` (MedicalNet)
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

    def __init__(self, dim=3, feature_layers=[4], weights_path=None):
        super().__init__()
        self._is_3d = (dim == 3)
        self.feature_layers = feature_layers

        self.weights_report = None
        if self._is_3d:
            self.model = resnet10_3d()
            self._in_channels = 1
            # MedicalNet weights through antsxdata (collection syntx_features). The
            # checkpoint is not registered yet, so the 3-D extractor cannot be built with
            # pretrained weights; say so instead of silently using random weights.
            if weights_path == "random":
                self.weights_report = "random (untrained; no registered MedicalNet checkpoint)"
            elif weights_path is not None:
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"ResNet-10 weights not found at {weights_path!r}")
                state = torch.load(weights_path, map_location='cpu', weights_only=False)
                self.weights_report = _load_state_dict_checked(
                    self.model, state.get('state_dict', state), rename={"downsample": "shortcut"})
            else:
                import antsxdata

                try:
                    weights_file = antsxdata.fetch("syntx_features/resnet_10_23iseg")
                except antsxdata.RegistryError as exc:
                    raise RuntimeError(
                        "ResNet10Extractor(dim=3) needs the MedicalNet checkpoint "
                        "syntx_features/resnet_10_23iseg, which is not in the antsxdata registry; "
                        "pass weights_path='random' to use an untrained extractor explicitly."
                    ) from exc
                state = torch.load(str(weights_file), map_location='cpu', weights_only=False)
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
        'random': keep the random initialisation. None: antsxdata ``syntx_features/model_swinvit``.
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
                # MONAI self-supervised Swin ViT weights, sha256-verified by antsxdata
                # (collection syntx_features); unavailable weights raise, never random.
                import antsxdata

                weights_path = str(antsxdata.fetch("syntx_features/model_swinvit"))
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Swin ViT weights not found at {weights_path!r}")

            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu', weights_only=False)
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


# Re-export FeatureSpaceLoss from canonical core losses (eliminates circular import with syn)
from .core.losses import FeatureSpaceLoss

__all__ = [
    "DINOv2Extractor",
    "FeatureExtractor",
    "FeatureSpaceLoss",
    "ResNet10Extractor",
    "SwinUNETRExtractor",
    "VGG19Extractor",
]

