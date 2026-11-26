"""Public, de-identified ReCoRe integration boundary.

This package is intentionally independent from the historical research
implementations. It exposes a small protocol for wrapping any of the six
TFB backbones and keeps the geometry module separate from TFB scheduling.
"""

from .protocol import BackboneForecastAdapter, ReCoReModule
from .model import RECORE_MODE, ROLE_GEOMETRY, VARIANT

__all__ = ["BackboneForecastAdapter", "ReCoReModule"]
"""Independent public ReCoRe module and TFB adapters."""

from .adapter import (
    AmplifierReCoRe,
    BACKBONE_ADAPTERS,
    DLinearReCoRe,
    MICNReCoRe,
    PatchMLPReCoRe,
    iTransformerReCoRe,
    xPatchReCoRe,
)
from .protocol import BackboneForecastAdapter, ReCoReModule

__all__ = [
    "BackboneForecastAdapter",
    "ReCoReModule",
    "BACKBONE_ADAPTERS",
    "DLinearReCoRe",
    "PatchMLPReCoRe",
    "iTransformerReCoRe",
    "xPatchReCoRe",
    "MICNReCoRe",
    "AmplifierReCoRe",
    "RECORE_MODE",
    "ROLE_GEOMETRY",
    "VARIANT",
]
