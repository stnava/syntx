"""
Metric registry, unified alias resolver, and canonical similarity loss dispatcher.

Unifies all similarity loss functionals under a single configuration dataclass
``SimilarityLossConfig`` and factory ``get_similarity_loss``.
Standardizes all returned callables to accept ``(I, J, mask=None)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .cross_correlation import BoxLNCCLoss, local_ncc_loss_nd
from .deep_features import FeatureSpaceLoss
from .mutual_information import mattes_mi_loss_nd
from .pointwise import l2_loss_nd, mae_loss_nd, mse_loss_nd
from .structural import distance_transform_loss, soft_dice_loss_nd


@dataclass(frozen=True)
class SimilarityLossConfig:
    """
    Configuration specification for canonical image similarity losses.
    """
    metric: str = "cc2"
    window_size: int = 5
    num_bins: int = 32
    squared: bool = True
    use_analytical_gradients: bool = False
    fixed_range: tuple[float, float] | None = (0.0, 1.0)
    sampling_percentage: float | None = None
    smooth_nr: float = 1e-5
    smooth_dr: float = 1e-5
    # Optional parameters for specialized families:
    mode: str | None = None
    tau: float = 0.10
    auto_mask: bool | None = None
    feature_layers: Sequence[int] | None = None


def parse_similarity_metric(
    metric_spec: str | SimilarityLossConfig,
    **kwargs,
) -> SimilarityLossConfig:
    """
    Parse a metric specification (string or config) and kwargs into a unified SimilarityLossConfig.

    Resolves all aliases across registration drivers:
    - Cross-correlation: 'cc2', 'lncc', 'cc', 'box_lncc', 'box_cc2', 'analytical_lncc', 'pseudo_lncc'
    - Mutual information: 'mattes', 'mattes_mi', 'mi', 'mmi', 'mattes_32', 'mattes_64' (extracts bin suffix)
    - Pointwise: 'mse', 'l2', 'mae'
    - Structural: 'soft_dice', 'dice', 'dt', 'sdf', 'distance_transform'
    - Deep features: 'vgg*', 'dino*', 'resnet*', 'swin*'
    """
    if isinstance(metric_spec, SimilarityLossConfig):
        # Update config with any explicitly provided kwargs
        current_dict = {
            "metric": metric_spec.metric,
            "window_size": metric_spec.window_size,
            "num_bins": metric_spec.num_bins,
            "squared": metric_spec.squared,
            "use_analytical_gradients": metric_spec.use_analytical_gradients,
            "fixed_range": metric_spec.fixed_range,
            "sampling_percentage": metric_spec.sampling_percentage,
            "smooth_nr": metric_spec.smooth_nr,
            "smooth_dr": metric_spec.smooth_dr,
            "mode": metric_spec.mode,
            "tau": metric_spec.tau,
            "auto_mask": metric_spec.auto_mask,
            "feature_layers": metric_spec.feature_layers,
        }
        for k, v in kwargs.items():
            if k in current_dict and v is not None:
                current_dict[k] = v
        return SimilarityLossConfig(**current_dict)

    if not isinstance(metric_spec, str):
        raise TypeError(f"Expected str or SimilarityLossConfig, got {type(metric_spec).__name__}")

    raw_str = metric_spec.strip()
    m_lower = raw_str.lower()

    # Window size resolution (window_size > kernel_size > lncc_window_size > lncc_radius)
    window_size = kwargs.get("window_size")
    if window_size is None:
        window_size = kwargs.get("kernel_size")
    if window_size is None:
        window_size = kwargs.get("lncc_window_size")
    if window_size is None and "lncc_radius" in kwargs and kwargs["lncc_radius"] is not None:
        window_size = 2 * int(kwargs["lncc_radius"]) + 1
    if window_size is None:
        window_size = 5

    # Defaults
    squared = kwargs.get("squared", True)
    use_analytical_gradients = kwargs.get(
        "use_analytical_gradients", kwargs.get("use_ants_pseudo_gradient", False)
    )
    fixed_range = kwargs.get("fixed_range", (0.0, 1.0))
    sampling_percentage = kwargs.get("sampling_percentage", None)
    smooth_nr = kwargs.get("smooth_nr", 1e-5)
    smooth_dr = kwargs.get("smooth_dr", 1e-5)
    mode = kwargs.get("mode", None)
    tau = kwargs.get("tau", 0.10)
    auto_mask = kwargs.get("auto_mask", None)
    feature_layers = kwargs.get("feature_layers", None)
    num_bins = kwargs.get("num_bins", kwargs.get("mattes_bins", 32))

    canonical_metric = m_lower

    # 1. Mutual Information Family
    if m_lower in ("mattes", "mattes_mi", "mi", "mmi") or m_lower.startswith(("mattes_", "mmi_", "mi_")):
        canonical_metric = "mattes_mi"
        # Extract bin count from suffix if present (e.g. 'mattes_32', 'mattes_64', 'mmi_48')
        parts = m_lower.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            extracted_bins = int(parts[-1])
            if kwargs.get("num_bins") is not None:
                num_bins = kwargs["num_bins"]
            else:
                num_bins = extracted_bins

    # 2. Cross-Correlation Family
    elif m_lower in ("cc2", "lncc2", "cc_squared"):
        canonical_metric = "cc2"
        squared = True
    elif m_lower in ("lncc", "cc", "ncc"):
        canonical_metric = "lncc"
        if "squared" not in kwargs:
            squared = False
    elif m_lower in ("box_lncc", "box_cc", "fireants_lncc"):
        canonical_metric = "box_lncc"
        if "squared" not in kwargs:
            squared = False
    elif m_lower in ("box_cc2", "box_lncc2"):
        canonical_metric = "box_cc2"
        squared = True
    elif m_lower in ("analytical_lncc", "pseudo_lncc", "ants_lncc"):
        canonical_metric = "lncc"
        use_analytical_gradients = True
        if "squared" not in kwargs:
            squared = False

    # 3. Pointwise Family
    elif m_lower in ("mse",):
        canonical_metric = "mse"
    elif m_lower in ("l2",):
        canonical_metric = "l2"
        squared = kwargs.get("squared", True)
    elif m_lower in ("mae", "l1"):
        canonical_metric = "mae"

    # 4. Structural & Label Family
    elif m_lower in ("soft_dice", "dice"):
        canonical_metric = "soft_dice"
    elif m_lower in ("dt", "distance_transform"):
        canonical_metric = "distance_transform"
        if mode is None:
            mode = "potential_lncc"
    elif m_lower in ("sdf",):
        canonical_metric = "distance_transform"
        if mode is None:
            mode = "sdf_mse"

    # 5. Deep Features Family
    elif m_lower.startswith(("vgg", "dino", "resnet", "swin")):
        canonical_metric = m_lower

    return SimilarityLossConfig(
        metric=canonical_metric,
        window_size=window_size,
        num_bins=num_bins,
        squared=squared,
        use_analytical_gradients=use_analytical_gradients,
        fixed_range=fixed_range,
        sampling_percentage=sampling_percentage,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        mode=mode,
        tau=tau,
        auto_mask=auto_mask,
        feature_layers=feature_layers,
    )


class SimilarityLossWrapper(nn.Module):
    """
    Standardized callable wrapper implementing ``(I, J, mask=None)`` interface.
    """

    def __init__(self, config: SimilarityLossConfig, **kwargs):
        super().__init__()
        self.config = config
        self._fn = None

        m = config.metric

        if m in ("box_lncc", "box_cc2"):
            self._box_loss = BoxLNCCLoss(
                kernel_size=config.window_size,
                smooth_nr=config.smooth_nr,
                smooth_dr=config.smooth_dr,
                squared=config.squared,
            )
        elif m.startswith(("vgg", "dino", "resnet", "swin")):
            self._feature_loss = self._build_feature_loss(config, **kwargs)
        else:
            self._box_loss = None
            self._feature_loss = None

    def _build_feature_loss(self, config: SimilarityLossConfig, **kwargs) -> nn.Module:
        m = config.metric
        mode = config.mode or "lncc_3d"
        from ...features import (
            DINOv2Extractor,
            ResNet10Extractor,
            SwinUNETRExtractor,
            VGG19Extractor,
        )

        extractor = None
        if m.startswith("vgg"):
            layers = config.feature_layers or [8]
            extractor = VGG19Extractor(feature_layers=list(layers))
        elif m.startswith("dino"):
            layers = config.feature_layers or [2]
            extractor = DINOv2Extractor(feature_layers=list(layers))
        elif m.startswith("resnet"):
            extractor = ResNet10Extractor(dim=kwargs.get("dim", 3), feature_layers=list(config.feature_layers or [4]))
        elif m.startswith("swin"):
            extractor = SwinUNETRExtractor()
        else:
            raise ValueError(f"Unknown deep feature metric: {m!r}")

        return FeatureSpaceLoss(
            extractor=extractor,
            mode=mode,
            lncc_window=config.window_size,
        )

    def forward(
        self,
        I: torch.Tensor,
        J: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = self.config.metric

        if m in ("box_lncc", "box_cc2"):
            return self._box_loss(I, J, mask=mask)

        elif m in ("cc2", "lncc"):
            return local_ncc_loss_nd(
                I,
                J,
                mask=mask,
                window_size=self.config.window_size,
                squared=self.config.squared,
                use_ants_pseudo_gradient=self.config.use_analytical_gradients,
            )

        elif m == "mattes_mi":
            auto_m = self.config.auto_mask
            if auto_m is None:
                auto_m = self.config.fixed_range is None
            return mattes_mi_loss_nd(
                I,
                J,
                mask=mask,
                num_bins=self.config.num_bins,
                sampling_percentage=self.config.sampling_percentage,
                auto_mask=auto_m,
                fixed_range=self.config.fixed_range,
            )

        elif m == "mse":
            return mse_loss_nd(I, J, mask=mask)

        elif m == "l2":
            return l2_loss_nd(I, J, mask=mask, squared=self.config.squared)

        elif m == "mae":
            return mae_loss_nd(I, J, mask=mask)

        elif m == "soft_dice":
            return soft_dice_loss_nd(I, J, mask=mask)

        elif m == "distance_transform":
            return distance_transform_loss(
                I,
                J,
                mode=self.config.mode or "potential_lncc",
                tau=self.config.tau,
                window_size=self.config.window_size,
                mask=mask,
            )

        elif m.startswith(("vgg", "dino", "resnet", "swin")):
            return self._feature_loss(I, J, mask=mask)

        else:
            raise ValueError(f"Unknown similarity metric: {m!r}")


def get_similarity_loss(
    metric_spec: str | SimilarityLossConfig,
    **kwargs,
) -> SimilarityLossWrapper:
    """
    Canonical loss factory.

    Resolves all aliases, extracts hyperparameters, and returns a standardized
    callable / nn.Module accepting ``(I, J, mask=None)``.
    """
    config = parse_similarity_metric(metric_spec, **kwargs)
    return SimilarityLossWrapper(config, **kwargs)
