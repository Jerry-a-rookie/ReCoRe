from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ts_benchmark.baselines.coreg_public._engine.memory_branch import (
    IndependentMemoryBranch,
    MultivariateMemoryBranch,
)
from ts_benchmark.baselines.time_series_library.models.DLinear import DLinear


class MemoryFusion(nn.Module):
    def __init__(self, topk: int, hidden_dim: int, context_mode: str = "none"):
        super().__init__()
        if context_mode not in {"none", "mean_other_scores"}:
            raise ValueError(
                "memory_fusion_context must be 'none' or 'mean_other_scores'."
            )
        self.topk = int(topk)
        self.context_mode = context_mode
        input_dim = self.topk * (2 if context_mode == "mean_other_scores" else 1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        if context_mode == "mean_other_scores":
            
            
            nn.init.zeros_(self.network[0].weight[:, self.topk :])

    def _features(self, retrieval_scores: torch.Tensor) -> torch.Tensor:
        if self.context_mode == "none":
            return retrieval_scores
        _, channels, _ = retrieval_scores.shape
        if channels <= 1:
            context = torch.zeros_like(retrieval_scores)
        else:
            context = (
                retrieval_scores.sum(dim=1, keepdim=True) - retrieval_scores
            ) / float(channels - 1)
        return torch.cat([retrieval_scores, context], dim=-1)

    def forward(
        self,
        retrieval_pred: torch.Tensor,
        base_pred: torch.Tensor,
        retrieval_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, pred_len, channels = base_pred.shape
        features = self._features(retrieval_scores)
        branch_weights = self.network(features.reshape(batch_size * channels, -1))
        branch_weights = branch_weights.reshape(batch_size, channels, 2)
        branches = torch.stack([retrieval_pred, base_pred], dim=-1)
        prediction = torch.sum(branches * branch_weights.unsqueeze(1), dim=-1)
        return prediction, branch_weights


class DLinearMemoryModel(nn.Module):
    """Forecasting backbone with a parallel multivariate Memory branch."""

    def __init__(self, configs, backbone_class: type[nn.Module] = DLinear):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.memory_variant = str(getattr(configs, "memory_variant", "global"))
        self.fusion_mode = str(getattr(configs, "memory_fusion_mode", "learned"))
        self.fixed_gamma = float(getattr(configs, "memory_fixed_gamma", 0.0))
        if self.fusion_mode not in {"learned", "fixed"}:
            raise ValueError("memory_fusion_mode must be 'learned' or 'fixed'.")
        self.backbone = backbone_class(configs)
        self.backbone_name = backbone_class.__name__
        if self.memory_variant == "independent":
            self.memory_branch = IndependentMemoryBranch(configs)
        elif self.memory_variant in {"global", "same_channel"}:
            configs.memory_retrieval_scope = (
                "same_channel" if self.memory_variant == "same_channel" else "global"
            )
            self.memory_branch = MultivariateMemoryBranch(configs)
        else:
            raise ValueError(
                "memory_variant must be 'global', 'same_channel', or 'independent'."
            )
        self.fusion = MemoryFusion(
            self.memory_branch.topk,
            int(getattr(configs, "memory_fusion_hidden", 128)),
            str(getattr(configs, "memory_fusion_context", "none")),
        )
        self._debug = self._empty_debug()

    @staticmethod
    def _empty_debug() -> dict[str, float]:
        return {
            "count": 0.0,
            "retrieval_weight_sum": 0.0,
            "base_weight_sum": 0.0,
            "confidence_sum": 0.0,
            "same_channel_ratio_sum": 0.0,
        }

    def prepare_memory(self, values: np.ndarray, device: torch.device) -> None:
        self.memory_branch.prepare_memory(values, device)

    def reset_debug_summary(self) -> None:
        self._debug = self._empty_debug()

    def set_fixed_gamma(self, gamma: float) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"fixed gamma must be in [0, 1], got {gamma}.")
        self.fixed_gamma = float(gamma)

    def get_debug_summary(self) -> dict:
        count = self._debug["count"]
        summary = self.memory_branch.get_memory_summary()
        summary["memory_variant"] = self.memory_variant
        summary["memory_backbone"] = self.backbone_name
        summary["memory_fusion_mode"] = self.fusion_mode
        summary["memory_fusion_context"] = self.fusion.context_mode
        if self.fusion.context_mode == "mean_other_scores":
            context_weight = self.fusion.network[0].weight[:, self.fusion.topk :]
            summary["memory_fusion_context_weight_l2"] = float(
                torch.linalg.vector_norm(context_weight.detach()).cpu().item()
            )
        if self.fusion_mode == "fixed":
            summary["memory_selected_gamma"] = self.fixed_gamma
        if count > 0:
            summary.update(
                {
                    "retrieval_branch_weight": self._debug["retrieval_weight_sum"] / count,
                    "base_branch_weight": self._debug["base_weight_sum"] / count,
                    "confidence_mean": self._debug["confidence_sum"] / count,
                    "same_channel_retrieval_ratio": self._debug[
                        "same_channel_ratio_sum"
                    ]
                    / count,
                }
            )
        return summary

    def forward(
        self,
        x_enc: torch.Tensor,
        index: torch.Tensor | None = None,
        mode: str = "train",
        return_branches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_pred = self.backbone(x_enc, None, None, None)
        retrieval_pred, retrieval_scores, diagnostics = self.memory_branch(
            x_enc, index, mode
        )
        if self.fusion_mode == "fixed":
            gamma = self.fixed_gamma
            prediction = base_pred + gamma * (retrieval_pred - base_pred)
            branch_weights = base_pred.new_empty(base_pred.shape[0], base_pred.shape[2], 2)
            branch_weights[..., 0] = gamma
            branch_weights[..., 1] = 1.0 - gamma
        else:
            prediction, branch_weights = self.fusion(
                retrieval_pred, base_pred, retrieval_scores
            )
        self._last_branch_weights = branch_weights

        if mode == "test":
            self._debug["count"] += 1.0
            self._debug["retrieval_weight_sum"] += float(
                branch_weights[..., 0].detach().mean().cpu().item()
            )
            self._debug["base_weight_sum"] += float(
                branch_weights[..., 1].detach().mean().cpu().item()
            )
            self._debug["confidence_sum"] += float(
                diagnostics["confidence_mean"].cpu().item()
            )
            self._debug["same_channel_ratio_sum"] += float(
                diagnostics["same_channel_ratio"].cpu().item()
            )
        prediction = prediction[:, -self.pred_len :, :]
        if return_branches:
            return prediction, base_pred, retrieval_pred
        return prediction


