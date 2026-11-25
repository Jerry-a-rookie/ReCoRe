"""Core CoReG model primitives exposed independently from TFB adapters."""
from __future__ import annotations

from ._engine.geometry_model import GeometryLearnedFusion, GeometryModel
from ._engine.time_geometry import MultiscaleRetrievalGeometryModel

COREG_MODE = "cross_variable_dual_path_reproduced"
ROLE_GEOMETRY = "aligned"
VARIANT = "full_aligned"

__all__ = [
    "COREG_MODE",
    "ROLE_GEOMETRY",
    "VARIANT",
    "GeometryModel",
    "GeometryLearnedFusion",
    "MultiscaleRetrievalGeometryModel",
]
