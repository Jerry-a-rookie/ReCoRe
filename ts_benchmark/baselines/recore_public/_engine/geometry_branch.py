from __future__ import annotations

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.recore_public._engine.memory_branch import (
    MultivariateMemoryBranch,
    _normalize_series,
)
from ts_benchmark.baselines.recore_public._engine.stage_profile import stage


class GeometryBranch(MultivariateMemoryBranch):
    """Final intra-channel predictive-memory retrieval branch."""

    def __init__(self, configs):
        configs.memory_retrieval_scope = "same_channel"
        configs.memory_memory_selection = "stratified_medoid"
        configs.memory_confidence_execution = "factorized"
        configs.memory_temporal_coverage_anchors = 200
        super().__init__(configs)
        self.stream_rows = int(getattr(configs, "recore_stream_rows", 4096))
        if self.stream_rows <= 0:
            raise ValueError("recore_stream_rows must be positive")
        self._streaming_calls = 0
        self._streaming_max_rows = 0

    def forward(self, x, index, mode):
        if mode != "test":
            return super().forward(x, index, mode)
        if not self._prepared or self.memory_keys.numel() == 0:
            raise RuntimeError("Memory memory is not prepared. Call prepare_memory first.")
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                f"Expected input [B, {self.seq_len}, {self.channels}], got {tuple(x.shape)}."
            )

        with stage("rolegeo.normalize"):
            x_ci = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
            query_mean, query_std, x_norm = _normalize_series(x_ci)
        with stage("rolegeo.query_encode"):
            query_keys = self._truth_keys(x_norm)
            memory_indices = self._same_channel_memory_indices
            memory_keys = self.memory_keys[memory_indices]
            query_keys, memory_keys = self._transform_retrieval_keys(
                query_keys, memory_keys
            )
        with stage("rolegeo.similarity"):
            similarity = torch.bmm(
                query_keys.reshape(batch_size, channels, -1).permute(1, 0, 2),
                memory_keys.transpose(1, 2),
            )
            similarity = similarity.permute(1, 0, 2).reshape(
                batch_size * channels, -1
            )
        with stage("rolegeo.mask"):
            similarity = self._control_similarity(similarity)
        with stage("rolegeo.topk"):
            topk_similarity, local_indices = torch.topk(
                similarity, self.topk, dim=1
            )
            if not bool(torch.isfinite(topk_similarity).all()):
                raise RuntimeError("ReCoRe memory does not contain enough neighbors.")
            query_channels = torch.arange(channels, device=x.device)
            query_channels = query_channels.unsqueeze(0).expand(
                batch_size, -1
            ).reshape(-1)
            topk_indices = memory_indices[
                query_channels.unsqueeze(1), local_indices
            ]

        first_layer = self.confidence_gate[0]
        with stage("rolegeo.confidence"):
            query_hidden = F.linear(
                x_norm,
                first_layer.weight[:, : self.seq_len],
                first_layer.bias,
            )
        retrieved_parts = []
        score_parts = []
        reliability_parts = []
        confidence_sum = x_norm.new_zeros(())
        confidence_count = 0
        for start in range(0, x_norm.shape[0], self.stream_rows):
            end = min(start + self.stream_rows, x_norm.shape[0])
            with stage("rolegeo.gather"):
                futures = self.memory_future[topk_indices[start:end]]
            with stage("rolegeo.confidence"):
                future_hidden = F.linear(
                    futures,
                    first_layer.weight[:, self.seq_len :],
                    None,
                )
                confidence = self.confidence_gate[3](
                    self.confidence_gate[2](
                        self.confidence_gate[1](
                            query_hidden[start:end].unsqueeze(1) + future_hidden
                        )
                    )
                ).squeeze(-1)
                adjusted = self._adjust_retrieval_scores(
                    topk_similarity[start:end], confidence
                )
            with stage("rolegeo.aggregate"):
                weights = self._compute_neighbor_weights(adjusted)
                weights = self._refine_neighbor_weights(weights, futures)
                retrieved = torch.sum(weights.unsqueeze(-1) * futures, dim=1)
                if getattr(self, "reliability_enabled", False):
                    from .reliability import reliability
                    reliability_parts.append(reliability(futures, weights, self.reliability_strength))
                if self.retrieval_truth_control == "zero_retrieval":
                    retrieved = torch.zeros_like(retrieved)
                retrieved_parts.append(retrieved)
                fusion_scores = self._scores_for_fusion(
                    topk_similarity[start:end], confidence, adjusted
                )
                score_parts.append(
                    self._augment_fusion_scores(fusion_scores, futures, weights)
                )
                confidence_sum = confidence_sum + confidence.detach().sum()
                confidence_count += confidence.numel()
                self._streaming_max_rows = max(
                    self._streaming_max_rows, end - start
                )

        with stage("rolegeo.aggregate"):
            retrieved = torch.cat(retrieved_parts, dim=0)
            fusion_scores = torch.cat(score_parts, dim=0)
            if reliability_parts:
                self._last_candidate_reliability = torch.cat(reliability_parts).reshape(batch_size, channels)
        with stage("rolegeo.output_denorm"):
            output_scale, output_shift = self.output_gate(x_norm)
            retrieved = retrieved * output_scale + output_shift
            retrieved = retrieved * query_std + query_mean
            retrieval_pred = retrieved.reshape(
                batch_size, channels, self.pred_len
            ).permute(0, 2, 1).contiguous()
            if self.retrieval_truth_control == "zero_retrieval":
                retrieval_pred = torch.zeros_like(retrieval_pred)
        self._last_confidence_execution = "factorized"
        self._streaming_calls += 1
        diagnostics = {
            "confidence_mean": confidence_sum / max(confidence_count, 1),
            "same_channel_ratio": x_norm.new_ones(()),
        }
        return (
            retrieval_pred,
            fusion_scores.reshape(batch_size, channels, -1),
            diagnostics,
        )

    def get_memory_summary(self) -> dict:
        summary = super().get_memory_summary()
        summary.update(
            {
                "architecture": "ReCoRe",
                "recore_retrieval_scope": "intra_channel",
                "recore_memory_selection": "temporal_stratified_predictive_coreset",
                "recore_confidence_execution": "exact_factorized_full",
                "recore_future_execution": "exact_streaming_query_rows",
                "recore_stream_rows": self.stream_rows,
                "recore_streaming_calls": self._streaming_calls,
                "recore_streaming_max_rows": self._streaming_max_rows,
            }
        )
        return summary


