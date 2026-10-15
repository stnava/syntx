# syntx 6.11.1

* antsxdata is no longer a hard dependency. It is the `syntx[features]` extra: only the
  pretrained deep-feature extractors (`ResNet10Extractor(dim=3)` MedicalNet weights,
  `SwinUNETRExtractor` MONAI Swin ViT weights) fetch through it, and only when no
  `weights_path` is given. Without the extra those constructors raise an `ImportError` naming
  the extra (`syntx.features._import_antsxdata`); nothing in the registration engine, spatial
  code or data fetchers uses antsxdata. The two tests that exercise the antsxdata path skip
  when it is not installed.
