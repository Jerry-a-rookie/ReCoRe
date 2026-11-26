"""Public TFB adapters for the six supported forecasting backbones.

The adapter names are intentionally stable and contain no historical research
aliases.  Each class keeps the original backbone implementation and adds the
same-channel CoReG retrieval branch through the private engine package.
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


class _AlignedCoReGMixin:
    MODEL_CLASS = MultiscaleRetrievalGeometryModel

    def __init__(self, **kwargs):
        kwargs["Geometry_multiscale_mode"] = "cross_variable_dual_path_reproduced"
        kwargs.setdefault("coreg_reliability_enabled", True)
        kwargs.setdefault("coreg_moving_average_kernel", 25)
        if int(kwargs["coreg_moving_average_kernel"]) < 1 or int(kwargs["coreg_moving_average_kernel"]) % 2 == 0:
            raise ValueError("moving average kernel must be positive and odd")
        kwargs.setdefault("memory_variant", "same_channel")
        super().__init__(**kwargs)


class DLinearCoReG(_AlignedCoReGMixin, DLinearSameChannelMemoryFrozenBase):
    MODEL_NAME = "DLinearCoReG"


class PatchMLPCoReG(_AlignedCoReGMixin, PatchMLPSameChannelMemoryFrozenBase):
    MODEL_NAME = "PatchMLPCoReG"


class iTransformerCoReG(_AlignedCoReGMixin, iTransformerSameChannelMemoryFrozenBase):
    MODEL_NAME = "iTransformerCoReG"


class xPatchCoReG(_AlignedCoReGMixin, xPatchSameChannelMemoryFrozenBase):
    MODEL_NAME = "xPatchCoReG"


class MICNCoReG(_AlignedCoReGMixin, MICNSameChannelMemoryFrozenBase):
    MODEL_NAME = "MICNCoReG"


class AmplifierCoReG(_AlignedCoReGMixin, AmplifierSameChannelMemoryFrozenBase):
    MODEL_NAME = "AmplifierCoReG"


BACKBONE_ADAPTERS = {
    cls.MODEL_NAME: cls
    for cls in (
        DLinearCoReG,
        PatchMLPCoReG,
        iTransformerCoReG,
        xPatchCoReG,
        MICNCoReG,
        AmplifierCoReG,
    )
}

__all__ = [*BACKBONE_ADAPTERS, "BACKBONE_ADAPTERS"]
