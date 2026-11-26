from __future__ import annotations

import torch
import torch.nn as nn

from ts_benchmark.baselines.recore_public._engine.memory_model import DLinearMemoryModel
from ts_benchmark.baselines.recore_public._engine.geometry_branch import GeometryBranch


class GeometryLearnedFusion(nn.Module):
    """Per-sample, per-channel learned fusion for the final ReCoRe model."""

    def __init__(self, topk: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(topk), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
            nn.Softmax(dim=-1),
        )

    @property
    def context_mode(self) -> str:
        return "none"

    def forward(
        self,
        retrieval_pred: torch.Tensor,
        base_pred: torch.Tensor,
        retrieval_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, channels = base_pred.shape
        branch_weights = self.network(
            retrieval_scores.reshape(batch_size * channels, -1)
        ).reshape(batch_size, channels, 2)
        branches = torch.stack([retrieval_pred, base_pred], dim=-1)
        prediction = torch.sum(branches * branch_weights.unsqueeze(1), dim=-1)
        return prediction, branch_weights


class GeometryModel(DLinearMemoryModel):
    """Final Learned architecture with an interchangeable forecasting Backbone."""

    def __init__(self, configs, backbone_class: type[nn.Module]):
        nn.Module.__init__(self)
        self.pred_len = int(configs.pred_len)
        self.memory_variant = "same_channel"
        self.fusion_mode = "learned"
        self.fixed_gamma = 0.0
        self.backbone = backbone_class(configs)
        self.backbone_name = backbone_class.__name__
        self.memory_branch = GeometryBranch(configs)
        self.fusion = GeometryLearnedFusion(
            self.memory_branch.topk,
            int(getattr(configs, "memory_fusion_hidden", 128)),
        )
        self._debug = self._empty_debug()

    def get_debug_summary(self) -> dict:
        summary = super().get_debug_summary()
        summary["architecture"] = "ReCoRe"
        summary["recore_fusion"] = "learned"
        return summary


