import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import resnet10_2d, resnet10_3d

class FeatureExtractor(nn.Module):
    """Abstract base for all feature extractors."""
    @property
    def is_3d(self) -> bool:
        raise NotImplementedError

    @property
    def in_channels(self) -> int:
        raise NotImplementedError

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract(self, x: torch.Tensor) -> list:
        raise NotImplementedError


class VGG19Extractor(FeatureExtractor):
    """VGG19 feature extractor with memory truncation."""
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

    def normalize(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features


class DINOv2Extractor(FeatureExtractor):
    """DINOv2 feature extractor supporting both ViT-S and ViT-B with sub-network pruning."""
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

    def normalize(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

    def extract(self, x):
        B, C, H, W = x.shape
        # Pad to patch_size-divisible dimensions
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))

        # We step through the model blocks to collect intermediate features
        # Matching DINOv2's forward pass but stopping at the max block
        x_tokens = self.model.prepare_tokens_with_masks(x)
        
        features = []
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)
            if i in self.feature_layers:
                # Patch tokens shape: (B, H_p * W_p, C_feat)
                patch_tokens = x_tokens[:, 1:]  # skip class token
                hp = (H + ph) // self.patch_size
                wp = (W + pw) // self.patch_size
                feat_grid = patch_tokens.reshape(B, hp, wp, -1).permute(0, 3, 1, 2)
                features.append(feat_grid)
        return features


class ResNet10Extractor(FeatureExtractor):
    """Unified ResNet-10 extractor supporting 2D or 3D."""
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
        return self._is_3d

    @property
    def in_channels(self) -> int:
        return self._in_channels

    def normalize(self, x):
        # Grayscale volumes are already in [0, 1]
        return x

    def extract(self, x):
        # Extract features at layers
        out = self.model.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.maxpool(out)
        
        features = []
        # Layer 1
        out = self.model.layer1(out)
        if 1 in self.feature_layers:
            features.append(out)
        # Layer 2
        out = self.model.layer2(out)
        if 2 in self.feature_layers:
            features.append(out)
        # Layer 3
        out = self.model.layer3(out)
        if 3 in self.feature_layers:
            features.append(out)
        # Layer 4
        out = self.model.layer4(out)
        if 4 in self.feature_layers:
            features.append(out)
            
        return features


class FeatureSpaceLoss(nn.Module):
    """Dimension-agnostic loss using modular feature extractors."""
    def __init__(self, extractor: FeatureExtractor, mode='lncc_3d', num_slices=4, lncc_window=9):
        super().__init__()
        self.extractor = extractor
        self.mode = mode
        self.num_slices = num_slices
        self.lncc_window = lncc_window

    def forward(self, input_nd, target_nd):
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

    def _forward_3d(self, input_nd, target_nd):
        # Native 3D pass
        feats_in = self.extractor.extract(self.extractor.normalize(input_nd))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_nd))
        
        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            # Compute 3D LNCC
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=5)
        return loss

    def _forward_2d_direct(self, input_nd, target_nd):
        # Direct 2D pass for 2D registration using 2D networks
        # Ensure 3-channel input if extractor expects RGB
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

    def _forward_2d_triplanar(self, input_nd, target_nd):
        # Extract orthogonal 2D slices and run 2D LNCC on 2D feature maps
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
            # Handle RGB channel stacking if needed
            if self.extractor.in_channels == 3:
                slices_in.append(F.interpolate(input_nd[:, 0, z-1:z+2], size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0, z-1:z+2], size=(target_size, target_size), mode='bilinear', align_corners=True))
            else:
                slices_in.append(F.interpolate(input_nd[:, 0:1, z:z+1], size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0:1, z:z+1], size=(target_size, target_size), mode='bilinear', align_corners=True))
                
        # Coronal
        for y in y_indices:
            if self.extractor.in_channels == 3:
                slices_in.append(F.interpolate(input_nd[:, 0, :, y-1:y+2, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0, :, y-1:y+2, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
            else:
                slices_in.append(F.interpolate(input_nd[:, 0:1, :, y:y+1, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0:1, :, y:y+1, :].movedim(2, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                
        # Sagittal
        for xi in x_indices:
            if self.extractor.in_channels == 3:
                slices_in.append(F.interpolate(input_nd[:, 0, :, :, xi-1:xi+2].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0, :, :, xi-1:xi+2].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
            else:
                slices_in.append(F.interpolate(input_nd[:, 0:1, :, :, xi:xi+1].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))
                slices_tg.append(F.interpolate(target_nd[:, 0:1, :, :, xi:xi+1].movedim(3, 1), size=(target_size, target_size), mode='bilinear', align_corners=True))

        input_batch = torch.cat(slices_in, dim=0)
        target_batch = torch.cat(slices_tg, dim=0)
        
        feats_in = self.extractor.extract(self.extractor.normalize(input_batch))
        feats_tg = self.extractor.extract(self.extractor.normalize(target_batch))
        
        loss = 0.0
        from .syn import local_ncc_loss_nd
        for f_in, f_tg in zip(feats_in, feats_tg):
            loss += local_ncc_loss_nd(f_in, f_tg, window_size=self.lncc_window)
        return loss

    def _forward_2d_reconstruct_3d(self, input_nd, target_nd):
        # 3D Feature LNCC on reconstructed feature volume
        D, H, W = input_nd.shape[2:]
        B = input_nd.shape[0]
        
        def reconstruct_3d_features(x):
            # 1. Axial
            slices_ax = []
            for z in range(1, D - 1):
                if self.extractor.in_channels == 3:
                    slices_ax.append(x[:, 0, z-1:z+2])
                else:
                    slices_ax.append(x[:, 0:1, z:z+1])
            batch_ax = self.extractor.normalize(torch.cat(slices_ax, dim=0))
            
            # 2. Coronal
            slices_co = []
            for y in range(1, H - 1):
                if self.extractor.in_channels == 3:
                    slices_co.append(x[:, 0, :, y-1:y+2, :].movedim(2, 1))
                else:
                    slices_co.append(x[:, 0:1, :, y:y+1, :].movedim(2, 1))
            batch_co = self.extractor.normalize(torch.cat(slices_co, dim=0))
            
            # 3. Sagittal
            slices_sa = []
            for xi in range(1, W - 1):
                if self.extractor.in_channels == 3:
                    slices_sa.append(x[:, 0, :, :, xi-1:xi+2].movedim(3, 1))
                else:
                    slices_sa.append(x[:, 0:1, :, :, xi:xi+1].movedim(3, 1))
            batch_sa = self.extractor.normalize(torch.cat(slices_sa, dim=0))
            
            # Run through extractor
            # We assume extract returns a single feature map for simplicity or we use the last one
            feat_ax = self.extractor.extract(batch_ax)[-1]
            feat_co = self.extractor.extract(batch_co)[-1]
            feat_sa = self.extractor.extract(batch_sa)[-1]
            
            # Reconstruct volumes
            vol_ax = feat_ax.view(D-2, B, -1, feat_ax.shape[2], feat_ax.shape[3]).permute(1, 2, 0, 3, 4)
            vol_co = feat_co.view(H-2, B, -1, feat_co.shape[2], feat_co.shape[3]).permute(1, 2, 3, 0, 4)
            vol_sa = feat_sa.view(W-2, B, -1, feat_sa.shape[2], feat_sa.shape[3]).permute(1, 2, 3, 4, 0)
            
            return vol_ax, vol_co, vol_sa
            
        vol_in_ax, vol_in_co, vol_in_sa = reconstruct_3d_features(input_nd)
        vol_tg_ax, vol_tg_co, vol_tg_sa = reconstruct_3d_features(target_nd)
        
        from .syn import local_ncc_loss_nd
        loss_ax = local_ncc_loss_nd(vol_in_ax, vol_tg_ax, window_size=5)
        loss_co = local_ncc_loss_nd(vol_in_co, vol_tg_co, window_size=5)
        loss_sa = local_ncc_loss_nd(vol_in_sa, vol_tg_sa, window_size=5)
        
        return loss_ax + loss_co + loss_sa
