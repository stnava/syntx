"""
syntx.resnet — small 2-D / 3-D ResNet-10 backbones (single-channel input, no classifier head).

Used by ``syntx.features.ResNet10Extractor`` (deep-feature similarity) and
``syntx.classifier.DiagnosticClassifier3D``.

- ``BasicBlock2D`` / ``BasicBlock3D``: two 3x3(x3) conv + batch-norm residual block, with a 1x1
  conv + batch-norm projection shortcut when the stride or channel count changes.
- ``ResNet10``: 7x7(x7) stride-2 stem, 3x3 stride-2 max-pool, then four one-block stages with
  64 / 128 / 256 / 512 channels (stages 2-4 stride 2); output is the stage-4 feature map
  (overall downsampling factor 32).
- ``resnet10_2d`` / ``resnet10_3d``: factories (randomly initialised).

Parameter names follow the MedicalNet layout for ``conv1``, ``bn1`` and ``layerN.0.*``; the
projection shortcut is called ``shortcut`` (MedicalNet: ``downsample``) --
``syntx.features.ResNet10Extractor`` renames those keys when loading a MedicalNet checkpoint.
"""

import torch
import torch.nn as nn


class BasicBlock2D(nn.Module):
    """
    Standard 2D Residual Block with 3x3 Convolutions and Batch Normalization.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of output channels for the convolutional layers.
    stride : int, default=1
        Stride for the first 3x3 convolution layer. Stride > 1 performs spatial downsampling.

    Attributes
    ----------
    expansion : int
        Channel expansion multiplier (1 for basic residual blocks).
    conv1 : nn.Conv2d
        First 3x3 convolutional layer.
    bn1 : nn.BatchNorm2d
        Batch normalization after conv1.
    relu : nn.ReLU
        Non-inplace ReLU activation function.
    conv2 : nn.Conv2d
        Second 3x3 convolutional layer.
    bn2 : nn.BatchNorm2d
        Batch normalization after conv2.
    shortcut : nn.Sequential
        1x1 conv + batch norm when stride != 1 or in_planes != planes, else empty (identity).
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input tensor through the 2D residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, in_planes, height, width)`.

        Returns
        -------
        torch.Tensor
            Shape `(batch, planes, ceil(height / stride), ceil(width / stride))`.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class BasicBlock3D(nn.Module):
    """
    Standard 3D Residual Block with 3x3x3 Convolutions and Batch Normalization.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of output channels for the convolutional layers.
    stride : int, default=1
        Stride for the first 3x3x3 convolution layer. Stride > 1 performs spatial downsampling.

    Attributes
    ----------
    expansion : int
        Channel expansion multiplier (1 for basic residual blocks).
    conv1 : nn.Conv3d
        First 3x3x3 convolutional layer.
    bn1 : nn.BatchNorm3d
        Batch normalization after conv1.
    relu : nn.ReLU
        Non-inplace ReLU activation function.
    conv2 : nn.Conv3d
        Second 3x3x3 convolutional layer.
    bn2 : nn.BatchNorm3d
        Batch normalization after conv2.
    shortcut : nn.Sequential
        1x1x1 conv + batch norm when stride != 1 or in_planes != planes, else empty (identity).
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input tensor through the 3D residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, in_planes, depth, height, width)`.

        Returns
        -------
        torch.Tensor
            Shape `(batch, planes, ceil(depth / stride), ceil(height / stride),
            ceil(width / stride))`.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    """
    ResNet-10 backbone for 2-D or 3-D single-channel input; returns stage-4 features.

    Parameters
    ----------
    block : type
        Residual block class (`BasicBlock2D` or `BasicBlock3D`).
    num_blocks : list of int
        Number of blocks in each of the 4 residual layers (e.g. `[1, 1, 1, 1]` for ResNet-10).
    dim : int, default=3
        2 builds 2-D layers; any other value builds 3-D layers.

    Attributes
    ----------
    in_planes : int
        Current internal channel width during construction.
    dim : int
        Spatial dimensionality.
    conv1 : nn.Module
        Initial 7x7 (2D) or 7x7x7 (3D) conv layer with stride 2.
    bn1 : nn.Module
        Batch normalization layer after conv1.
    relu : nn.ReLU
        ReLU activation function.
    maxpool : nn.Module
        Max pooling layer with kernel size 3 and stride 2.
    layer1, layer2, layer3, layer4 : nn.Sequential
        Sequential residual layers producing channel depths of 64, 128, 256, and 512 respectively.
    """

    def __init__(self, block, num_blocks: list, dim: int = 3):
        super().__init__()
        self.in_planes = 64
        self.dim = dim

        if dim == 2:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm3d(64)
            self.relu = nn.ReLU(inplace=False)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Helper method to construct a sequential residual layer consisting of `num_blocks` blocks.

        Parameters
        ----------
        block : type
            Residual block class (`BasicBlock2D` or `BasicBlock3D`).
        planes : int
            Target output channel width.
        num_blocks : int
            Number of residual blocks in this stage.
        stride : int
            Stride for the first block in this stage.

        Returns
        -------
        nn.Sequential
            Sequential block containing configured residual stages.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass through ResNet-10 backbone up to Layer 4.

        Parameters
        ----------
        x : torch.Tensor
            Single-channel input, shape `(batch, 1, H, W)` or `(batch, 1, D, H, W)`.

        Returns
        -------
        torch.Tensor
            Stage-4 features, `(batch, 512, H_4, W_4)` or `(batch, 512, D_4, H_4, W_4)`, each
            spatial size about 1/32 of the input (e.g. 64 -> 2).
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4


def resnet10_2d() -> ResNet10:
    """
    Build a randomly initialised 2-D ResNet-10 (``BasicBlock2D``, one block per stage).

    Returns
    -------
    ResNet10
        Configured 2D ResNet-10 model.
    """
    return ResNet10(BasicBlock2D, [1, 1, 1, 1], dim=2)


def resnet10_3d() -> ResNet10:
    """
    Build a randomly initialised 3-D ResNet-10 (``BasicBlock3D``, one block per stage).

    Returns
    -------
    ResNet10
        Same stage layout as MedicalNet's ResNet-10; the shortcut parameters are named
        ``shortcut`` rather than MedicalNet's ``downsample`` (renamed on load by
        ``ResNet10Extractor``).
    """
    return ResNet10(BasicBlock3D, [1, 1, 1, 1], dim=3)
