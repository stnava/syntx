"""Central compute-device resolver for the antsx ecosystem (one implementation).

Every package that needs a torch device (syntx itself, antsxmm, antsxfunctional, antsxdwi,
antsxstructural, antsxslowflow) resolves it here; nothing else may hold ``cuda > mps > cpu``
logic or default a device to ``"cpu"``.

Precedence of :func:`resolve_device`
------------------------------------
1. ``requested`` (an explicit argument other than ``None`` / ``"auto"``).
2. Environment: for registration tasks ``ANTSXMM_REG_DEVICE`` and then ``ANTSXMM_DEVICE``; for
   every other task ``ANTSXMM_DEVICE``. A value of ``"auto"`` (or empty) falls through.
3. Automatic: ``cuda`` > ``mps`` > ``cpu``.

An explicit or environment device that is not available **raises**
:class:`DeviceUnavailableError`; only the automatic path falls through to the next device. An
unknown device type raises ``ValueError``. When ``mps`` is chosen, ``PYTORCH_ENABLE_MPS_FALLBACK=1``
is set unless the user already set that variable.

The ``source`` label returned by :func:`resolve_device_with_source` is one of ``"explicit"``,
``"env:ANTSXMM_REG_DEVICE"``, ``"env:ANTSXMM_DEVICE"``, ``"auto:cuda"``, ``"auto:mps"``,
``"auto:cpu"``. Callers that layer their own tier (e.g. antsxmm's per-task policy) pass it as
``requested`` and relabel the source.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DeviceUnavailableError",
    "REGISTRATION_TASKS",
    "available_devices",
    "is_available",
    "auto_device",
    "resolve_device",
    "resolve_device_with_source",
]

#: Tasks that honour ``ANTSXMM_REG_DEVICE`` (in addition to ``ANTSXMM_DEVICE``).
REGISTRATION_TASKS = frozenset({"reg", "registration", "rigid_motion", "rigid", "affine", "syn"})

_VALID_TYPES = ("cpu", "cuda", "mps")


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested compute device is not available on this machine."""


def _mps_available() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _split(device: str) -> tuple[str, str | None]:
    kind, _, index = device.partition(":")
    return kind, (index or None)


def is_available(device: Any) -> bool:
    """Whether ``device`` ('cpu', 'cuda', 'cuda:1', 'mps' or a ``torch.device``) can be used here."""
    import torch

    kind, index = _split(str(device).strip().lower())
    if kind == "cpu":
        return True
    if kind == "mps":
        return _mps_available()
    if kind == "cuda":
        if not torch.cuda.is_available():
            return False
        return index is None or (index.isdigit() and int(index) < torch.cuda.device_count())
    return False


def auto_device() -> str:
    """``'cuda'`` if available, else ``'mps'``, else ``'cpu'`` (no env, no validation of requests)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def available_devices() -> list[str]:
    """Usable device types on this machine, best first (always ends with 'cpu')."""
    import torch

    devs: list[str] = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if _mps_available():
        devs.append("mps")
    devs.append("cpu")
    return devs


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    return text


def _validated(device: str, source: str) -> str:
    kind, _ = _split(device)
    if kind not in _VALID_TYPES:
        raise ValueError(
            f"Unknown compute device {device!r} (from {source}); expected 'auto', 'cpu', 'cuda[:N]' or 'mps'."
        )
    if not is_available(device):
        raise DeviceUnavailableError(
            f"Compute device {device!r} was requested ({source}) but is not available on this machine "
            f"(available: {', '.join(available_devices())}). Request an available device or 'auto'; "
            "there is no silent downgrade."
        )
    return device


def resolve_device_with_source(requested: Any = None, task: str | None = None) -> tuple[str, str]:
    """Resolve a compute device and say how it was chosen; see the module docstring."""
    explicit = _clean(requested)
    if explicit is not None:
        dev, source = _validated(explicit, "explicit"), "explicit"
    else:
        env_names = ["ANTSXMM_DEVICE"]
        if task is not None and str(task).lower() in REGISTRATION_TASKS:
            env_names.insert(0, "ANTSXMM_REG_DEVICE")
        dev = source = None  # type: ignore[assignment]
        for name in env_names:
            value = _clean(os.environ.get(name))
            if value is not None:
                source = f"env:{name}"
                dev = _validated(value, source)
                break
        if dev is None:
            dev = auto_device()
            source = f"auto:{dev}"
    if dev.startswith("mps"):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return dev, source  # type: ignore[return-value]


def resolve_device(requested: Any = None, task: str | None = None) -> str:
    """Resolve a compute device string ('cpu', 'cuda[:N]' or 'mps').

    ``requested`` None / 'auto' => environment, then cuda > mps > cpu. An explicit device that is not
    available raises :class:`DeviceUnavailableError`. ``task`` selects the registration env var
    (``ANTSXMM_REG_DEVICE``) for registration tasks; see :data:`REGISTRATION_TASKS`.
    """
    return resolve_device_with_source(requested, task)[0]
