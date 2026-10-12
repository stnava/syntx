# syntx 6.6.1

Fix: the device guard (`tests/test_device_guard.py`, added in 6.5.0) did not detect `device = torch.device("cpu")`. The scan tokenizes the source and re-joins tokens with spaces, so the pattern `torch\.device\(` never matched `torch . device ( "cpu" )`. The pattern now tolerates whitespace and a regression test pins it.

Making the pattern correct exposed 8 CPU-device sites the guard had not counted. Each was inspected and allow-listed with its reason (SciPy exact distance transform, JAX path accepts only CPU, `SyNToTransform` grids stored on CPU, defaults when no tensor input carries a device). None changes behaviour; no library code changed.
