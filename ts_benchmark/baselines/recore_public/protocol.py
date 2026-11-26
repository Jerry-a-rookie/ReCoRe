"""Stable public contracts for the ReCoRe module."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class BackboneForecastAdapter(Protocol):
    """Minimal interface ReCoRe needs from a forecasting Backbone."""

    def fit(self, train_valid_data, train_ratio_in_tv: float = 1.0) -> None:
        ...

    def predict_tensor(self, history: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, horizon, channels]`` Anchor predictions."""
        ...

    def forecast(
        self,
        horizon: int,
        series,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        ...


class ReCoReModule(nn.Module):
    """Public module shell for geometry-enhanced forecasting.

    The concrete implementation will own TimeGeo, StableRel, RoleGeo,
    geometry-specific readouts and fixed CoMerge. It receives an injected
    Backbone adapter and never relies on a Backbone's private layers.
    """

    def __init__(self, backbone: BackboneForecastAdapter):
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module adapter")
        self.backbone = backbone

    def predict_tensor(self, history: torch.Tensor) -> torch.Tensor:
        """Run the injected Anchor and reserve the public output contract."""
        prediction = self.backbone.predict_tensor(history)
        if prediction.ndim != 3:
            raise ValueError(
                "Backbone predictions must have shape [batch, horizon, channels]"
            )
        return prediction
