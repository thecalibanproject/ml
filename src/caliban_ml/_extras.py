"""Helpers for optional heavy dependencies (``[train]``, ``[export]``, ``[presidio]``)."""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtraError(ImportError):
    """Raised when a code path needs an optional extra that is not installed."""


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise a helpful error naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingExtraError(
            f"'{module}' is not installed. Install the extra with: "
            f"pip install 'caliban-ml[{extra}]'"
        ) from exc
