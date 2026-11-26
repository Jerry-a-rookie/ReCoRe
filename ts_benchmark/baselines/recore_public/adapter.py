"""Public TFB adapters for the six supported forecasting backbones.

The adapter names are intentionally stable and contain no historical research
aliases.  Each class keeps the original backbone implementation and adds the
same-channel ReCoRe retrieval branch through the private engine package.
"""
from __future__ import annotations

from ._engine.memory_adapter import (
    AmplifierSameChannelMemoryFrozenBase,
    DLinearSameChannelMemoryFrozenBase,
    MICNSameChannelMemoryFrozenBase,
    PatchMLPSameChannelMemoryFrozenBase,
    iTransformerSameChannelMemoryFrozenBase,
    xPatchSameChannelMemoryFrozenBase,
)
from ._engine.time_geometry import MultiscaleRetrievalGeometryModel


class _AlignedReCoReMixin:
    MODEL_CLASS = MultiscaleRetrievalGeometryModel

    def __init__(self, **kwargs):
        kwargs["Geometry_multiscale_mode"] = "cross_variable_dual_path_reproduced"
        kwargs.setdefault("recore_reliability_enabled", True)
        kwargs.setdefault("recore_moving_average_kernel", 25)
        if int(kwargs["recore_moving_average_kernel"]) < 1 or int(kwargs["recore_moving_average_kernel"]) % 2 == 0:
            raise ValueError("moving average kernel must be positive and odd")
        kwargs.setdefault("memory_variant", "same_channel")
        super().__init__(**kwargs)


class DLinearReCoRe(_AlignedReCoReMixin, DLinearSameChannelMemoryFrozenBase):
    MODEL_NAME = "DLinearReCoRe"


class PatchMLPReCoRe(_AlignedReCoReMixin, PatchMLPSameChannelMemoryFrozenBase):
    MODEL_NAME = "PatchMLPReCoRe"


class iTransformerReCoRe(_AlignedReCoReMixin, iTransformerSameChannelMemoryFrozenBase):
    MODEL_NAME = "iTransformerReCoRe"


class xPatchReCoRe(_AlignedReCoReMixin, xPatchSameChannelMemoryFrozenBase):
    MODEL_NAME = "xPatchReCoRe"


class MICNReCoRe(_AlignedReCoReMixin, MICNSameChannelMemoryFrozenBase):
    MODEL_NAME = "MICNReCoRe"


class AmplifierReCoRe(_AlignedReCoReMixin, AmplifierSameChannelMemoryFrozenBase):
    MODEL_NAME = "AmplifierReCoRe"


BACKBONE_ADAPTERS = {
    cls.MODEL_NAME: cls
    for cls in (
        DLinearReCoRe,
        PatchMLPReCoRe,
        iTransformerReCoRe,
        xPatchReCoRe,
        MICNReCoRe,
        AmplifierReCoRe,
    )
}

__all__ = [*BACKBONE_ADAPTERS, "BACKBONE_ADAPTERS"]
