"""Public, de-identified CoReG integration boundary.

This package is intentionally independent from the historical research
implementations. It exposes a small protocol for wrapping any of the six
TFB backbones and keeps the geometry module separate from TFB scheduling.
"""

from .protocol import BackboneForecastAdapter, CoReGModule
from .model import COREG_MODE, ROLE_GEOMETRY, VARIANT

__all__ = ["BackboneForecastAdapter", "CoReGModule"]
"""Independent public CoReG module and TFB adapters."""

from .adapter import (
    AmplifierCoReG,
    BACKBONE_ADAPTERS,
    DLinearCoReG,
    MICNCoReG,
    PatchMLPCoReG,
    iTransformerCoReG,
    xPatchCoReG,
)
from .protocol import BackboneForecastAdapter, CoReGModule

__all__ = [
    "BackboneForecastAdapter",
    "CoReGModule",
    "BACKBONE_ADAPTERS",
    "DLinearCoReG",
    "PatchMLPCoReG",
    "iTransformerCoReG",
    "xPatchCoReG",
    "MICNCoReG",
    "AmplifierCoReG",
    "COREG_MODE",
    "ROLE_GEOMETRY",
    "VARIANT",
]
