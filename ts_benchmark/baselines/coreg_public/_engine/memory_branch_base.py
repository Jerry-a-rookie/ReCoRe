from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.coreg_public._engine.window_sampling import (
    nested_window_indices,
    window_indices_sha256,
)


logger = logging.getLogger(__name__)


def _normalize_series(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return mean, std, (x - mean) / std


class PastSeriesEncoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int = 64, embedding_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OutputAffineGate(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        self.fc1 = nn.Linear(seq_len, 2 * seq_len)
        self.fc2 = nn.Linear(2 * seq_len, 2 * pred_len)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias[:pred_len].fill_(1.0)
            self.fc2.bias[pred_len:].zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.fc2(F.gelu(self.fc1(x)))
        return params[:, : self.pred_len], params[:, self.pred_len :]


class MultivariateMemoryBranch(nn.Module):
    """Channel-independent Memory branch with a shared predictive encoder and GMB."""

    CACHE_VERSION = 2

    def __init__(self, configs):
        super().__init__()
        self.retrieval_truth_control = str(
            getattr(configs, "memory_retrieval_truth_control", "full")
        )
        if self.retrieval_truth_control not in {
            "full", "random_neighbors", "shuffled_future", "zero_retrieval",
            "untrained_encoder", "raw_history_knn",
        }:
            raise ValueError(
                "memory_retrieval_truth_control must be one of full, "
                "random_neighbors, shuffled_future, zero_retrieval, "
                "untrained_encoder, raw_history_knn"
            )
        self.retrieval_truth_use_base_cache = bool(
            getattr(configs, "memory_retrieval_truth_use_base_cache", False)
        )
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.embedding_dim = int(getattr(configs, "memory_embedding_dim", 16))
        self.encoder_hidden = int(getattr(configs, "memory_encoder_hidden", 64))
        self.gate_hidden = int(getattr(configs, "memory_gate_hidden", 128))
        self.topk = int(getattr(configs, "memory_topk", 10))
        self.gmb_size = int(getattr(configs, "memory_gmb_size", 1000))
        self.pcl_epochs = int(getattr(configs, "memory_pcl_epochs", 80))
        self.pcl_batch_size = int(getattr(configs, "memory_pcl_batch_size", 256))
        self.pcl_lr = float(getattr(configs, "memory_pcl_lr", 1e-3))
        self.pcl_temperature = float(getattr(configs, "memory_pcl_temperature", 0.05))
        self.retrieval_temperature = float(
            getattr(configs, "memory_retrieval_temperature", 1.0)
        )
        self.confidence_execution = str(
            getattr(configs, "memory_confidence_execution", "factorized")
        )
        if self.confidence_execution not in {"reference", "factorized", "adaptive"}:
            raise ValueError(
                "memory_confidence_execution must be 'reference', 'factorized', or 'adaptive'."
            )
        self.factorized_confidence_min_elements = int(
            getattr(configs, "memory_factorized_confidence_min_elements", 25_000_000)
        )
        self.retrieval_scope = str(
            getattr(configs, "memory_retrieval_scope", "global")
        )
        self.encode_batch_size = int(getattr(configs, "memory_encode_batch_size", 1024))
        self.cluster_batch_size = int(getattr(configs, "memory_cluster_batch_size", 4096))
        self.cluster_iterations = int(getattr(configs, "memory_cluster_iterations", 100))
        self.temporal_coverage_anchors = int(
            getattr(configs, "memory_temporal_coverage_anchors", 0)
        )
        self.memory_selection_requested = str(
            getattr(configs, "memory_memory_selection", "medoid")
        )
        self.uniform_memory_min_channels = int(
            getattr(configs, "memory_uniform_memory_min_channels", 64)
        )
        self.memory_selection = (
            "uniform"
            if self.memory_selection_requested == "adaptive"
            and self.channels >= self.uniform_memory_min_channels
            else (
                "medoid"
                if self.memory_selection_requested == "adaptive"
                else self.memory_selection_requested
            )
        )
        self.seed = int(configs.memory_seed)
        
        
        
        self._control_rng = torch.Generator(device="cpu")
        self._control_rng.manual_seed(self.seed)
        self._retrieval_trace_dir = getattr(configs, "coreg_retrieval_trace_dir", "")
        self._retrieval_trace_batch = 0
        self.train_window_ratio = float(
            getattr(configs, "memory_train_window_ratio", 1.0)
        )
        self.cache_root = Path(getattr(configs, "memory_cache_root", ".memory_cache"))
        self.require_existing_cache = bool(
            getattr(configs, "memory_require_existing_cache", False)
        )
        encoder_cache_root = getattr(configs, "memory_encoder_cache_root", None)
        self.encoder_cache_root = (
            Path(encoder_cache_root) if encoder_cache_root is not None else None
        )

        if self.topk <= 0 or self.gmb_size <= 0:
            raise ValueError("memory_topk and memory_gmb_size must be positive.")
        if self.topk > self.gmb_size:
            raise ValueError("memory_topk cannot exceed memory_gmb_size.")
        if not 0 <= self.temporal_coverage_anchors <= self.gmb_size:
            raise ValueError(
                "memory_temporal_coverage_anchors must be between 0 and memory_gmb_size."
            )
        if self.retrieval_scope not in {"global", "same_channel"}:
            raise ValueError(
                "memory_retrieval_scope must be 'global' or 'same_channel'."
            )
        if self.memory_selection_requested not in {
            "medoid",
            "uniform",
            "adaptive",
            "stratified_medoid",
            "information_arc",
        }:
            raise ValueError(
                "memory_memory_selection must be 'medoid', 'uniform', "
                "'adaptive', 'stratified_medoid', or 'information_arc'."
            )
        if self.uniform_memory_min_channels <= 0:
            raise ValueError("memory_uniform_memory_min_channels must be positive.")
        if (
            self.memory_selection in {"stratified_medoid", "information_arc"}
            and self.retrieval_scope != "same_channel"
        ):
            raise ValueError(
                "stratified_medoid memory selection requires same_channel retrieval."
            )

        self.encoder = PastSeriesEncoder(
            self.seq_len,
            hidden_dim=self.encoder_hidden,
            embedding_dim=self.embedding_dim,
        )
        self.confidence_gate = nn.Sequential(
            nn.Linear(self.seq_len + self.pred_len, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.output_gate = OutputAffineGate(self.seq_len, self.pred_len)
        self._last_confidence_execution = "uninitialized"

        self.register_buffer("memory_keys", torch.empty(0, self.embedding_dim))
        self.register_buffer("memory_future", torch.empty(0, self.pred_len))
        self.register_buffer("memory_starts", torch.empty(0, dtype=torch.long))
        self.register_buffer("memory_channels", torch.empty(0, dtype=torch.long))
        self.register_buffer(
            "_same_channel_memory_indices",
            torch.empty(0, 0, dtype=torch.long),
            persistent=False,
        )

        self.cache_path: str | None = None
        self.encoder_cache_source: str | None = None
        self.encoder_cache_reused = False
        self.cache_used = False
        self._prepared = False

    def _truth_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Build retrieval keys for the requested truth control."""
        if self.retrieval_truth_control == "raw_history_knn":
            
            
            raw = F.adaptive_avg_pool1d(x.unsqueeze(1), self.embedding_dim).squeeze(1)
            return F.normalize(raw, dim=-1)
        return F.normalize(self.encoder(x), dim=-1)

    def _control_similarity(self, similarity: torch.Tensor) -> torch.Tensor:
        if self.retrieval_truth_control != "random_neighbors":
            return similarity
        noise = torch.rand(
            similarity.shape, generator=self._control_rng, device="cpu"
        )
        return noise.to(device=similarity.device, dtype=similarity.dtype)

    def _apply_future_control(self) -> None:
        if self.retrieval_truth_control != "shuffled_future":
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 7919)
        permutation = torch.randperm(self.memory_future.shape[0], generator=generator)
        self.memory_future = self.memory_future[permutation].contiguous()

    @staticmethod
    def _metadata_compatible(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        
        
        
        for key, value in expected.items():
            if key == "version" or key.startswith("relational_predictive_contrast_"):
                continue
            if actual.get(key) != value:
                return False
        return True

    @property
    def total_window(self) -> int:
        return self.seq_len + self.pred_len

    def _compute_neighbor_weights(
        self, adjusted_scores: torch.Tensor
    ) -> torch.Tensor:
        return F.softmax(adjusted_scores / self.retrieval_temperature, dim=1)

    def _adjust_retrieval_scores(
        self,
        similarity: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        return similarity * confidence

    def _scores_for_fusion(
        self,
        similarity: torch.Tensor,
        confidence: torch.Tensor,
        adjusted_scores: torch.Tensor,
    ) -> torch.Tensor:
        return adjusted_scores

    def _augment_fusion_scores(
        self,
        fusion_scores: torch.Tensor,
        topk_future: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return fusion_scores

    def _refine_neighbor_weights(
        self,
        weights: torch.Tensor,
        topk_future: torch.Tensor,
    ) -> torch.Tensor:
        return weights

    def _transform_retrieval_keys(
        self,
        query_keys: torch.Tensor,
        memory_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional shared metric transform; production retrieval is identity."""
        return query_keys, memory_keys

    def _project_topk_future(
        self,
        topk_future: torch.Tensor,
        topk_indices: torch.Tensor,
        first_layer: nn.Linear,
        mode: str,
    ) -> torch.Tensor:
        del topk_indices, mode
        return F.linear(
            topk_future,
            first_layer.weight[:, self.seq_len :],
            None,
        )

    def _gather_pairs(
        self,
        values: torch.Tensor,
        pair_ids: torch.Tensor,
        length: int,
        offset: int,
    ) -> torch.Tensor:
        pair_ids = pair_ids.to(dtype=torch.long, device=values.device)
        starts = torch.div(pair_ids, self.channels, rounding_mode="floor")
        channels = torch.remainder(pair_ids, self.channels)
        steps = torch.arange(length, device=values.device).unsqueeze(0)
        rows = starts.unsqueeze(1) + offset + steps
        return values[rows, channels.unsqueeze(1)]

    def _encode_pairs(
        self,
        values: torch.Tensor,
        pair_ids: torch.Tensor | int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(pair_ids, int):
            pair_ids = torch.arange(pair_ids, dtype=torch.long)
        outputs = []
        self.encoder.to(device)
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, pair_ids.numel(), self.encode_batch_size):
                current_ids = pair_ids[start : start + self.encode_batch_size]
                x = self._gather_pairs(
                    values, current_ids, self.seq_len, 0
                ).to(device)
                _, _, x_norm = _normalize_series(x)
                outputs.append(self._truth_keys(x_norm).cpu())
        self.encoder.cpu()
        return torch.cat(outputs, dim=0)

    def _train_predictive_encoder(
        self,
        values: torch.Tensor,
        pair_ids: torch.Tensor | int,
        device: torch.device,
    ) -> None:
        if isinstance(pair_ids, int):
            pair_ids = torch.arange(
                pair_ids * self.channels, dtype=torch.long
            )
        pair_count = int(pair_ids.numel())
        if pair_count < 2:
            raise ValueError("Memory requires at least two channel-window pairs.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        self.encoder.to(device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.pcl_lr)

        for epoch in range(self.pcl_epochs):
            self.encoder.train()
            order = torch.randperm(pair_count, generator=generator)
            losses = []
            for start in range(0, pair_count, self.pcl_batch_size):
                current_ids = pair_ids[order[start : start + self.pcl_batch_size]]
                if current_ids.numel() < 2:
                    continue

                starts = torch.div(
                    current_ids, self.channels, rounding_mode="floor"
                )
                x = self._gather_pairs(
                    values, current_ids, self.seq_len, 0
                ).to(device)
                y = self._gather_pairs(
                    values, current_ids, self.pred_len, self.seq_len
                ).to(device)
                _, _, x_norm = _normalize_series(x)
                _, _, y_norm = _normalize_series(y)

                features = F.normalize(self.encoder(x_norm), dim=-1)
                future_distance = torch.cdist(y_norm, y_norm, p=2)
                overlap = (
                    torch.abs(starts.unsqueeze(1) - starts.unsqueeze(0))
                    < self.total_window
                ).to(device)
                legal_rows = (~overlap).any(dim=1)
                if not bool(legal_rows.any()):
                    continue

                future_distance = future_distance.masked_fill(overlap, float("inf"))
                positive = torch.argmin(future_distance, dim=1)
                logits = torch.matmul(features, features.transpose(0, 1))
                logits = logits / self.pcl_temperature
                logits = logits.masked_fill(overlap, float("-inf"))
                loss = F.cross_entropy(logits[legal_rows], positive[legal_rows])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

            if not losses:
                raise RuntimeError(
                    "Memory PCL produced no legal positive pairs. Increase the data span "
                    "or memory_pcl_batch_size."
                )
            logger.info(
                "Memory PCL epoch %d/%d: loss=%.6f",
                epoch + 1,
                self.pcl_epochs,
                float(np.mean(losses)),
            )

        self.encoder.cpu()

    def _temporal_anchor_indices(
        self, pair_count: int, memory_size: int
    ) -> np.ndarray:
        anchor_count = min(self.temporal_coverage_anchors, memory_size, pair_count)
        if anchor_count == 0:
            return np.empty(0, dtype=np.int64)
        anchors = np.unique(
            np.rint(np.linspace(0, pair_count - 1, anchor_count)).astype(np.int64)
        )
        query_starts = np.arange(pair_count)
        overlap_left = np.searchsorted(
            anchors, query_starts - self.total_window + 1, side="left"
        )
        overlap_right = np.searchsorted(
            anchors, query_starts + self.total_window, side="left"
        )
        legal_counts = len(anchors) - (overlap_right - overlap_left)
        if int(legal_counts.min()) < self.topk:
            raise ValueError(
                f"{len(anchors)} temporal anchors provide only "
                f"{int(legal_counts.min())} non-overlapping neighbors; "
                f"memory_topk={self.topk}. Increase memory_temporal_coverage_anchors."
            )
        return anchors

    def _select_memory_pairs(
        self, features: torch.Tensor, *, temporal_coverage: bool = False
    ) -> torch.Tensor:
        pair_count = features.shape[0]
        memory_size = min(self.gmb_size, pair_count)
        if memory_size == pair_count:
            return torch.arange(pair_count, dtype=torch.long)

        anchors = (
            self._temporal_anchor_indices(pair_count, memory_size)
            if temporal_coverage
            else np.empty(0, dtype=np.int64)
        )

        from sklearn.cluster import MiniBatchKMeans

        feature_np = features.numpy().astype(np.float32, copy=False)
        cluster = MiniBatchKMeans(
            n_clusters=memory_size,
            batch_size=min(self.cluster_batch_size, pair_count),
            max_iter=self.cluster_iterations,
            n_init=3,
            random_state=self.seed,
        )
        labels = cluster.fit_predict(feature_np)
        centers = cluster.cluster_centers_.astype(np.float32, copy=False)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

        medoids = np.empty(memory_size, dtype=np.int64)
        used: set[int] = set()
        all_indices = np.arange(pair_count)
        for cluster_id in range(memory_size):
            members = all_indices[labels == cluster_id]
            if members.size == 0:
                ranking = np.argsort(-(feature_np @ centers[cluster_id]))
            else:
                member_scores = feature_np[members] @ centers[cluster_id]
                ranking = members[np.argsort(-member_scores)]
            selected = next((int(idx) for idx in ranking if int(idx) not in used), None)
            if selected is None:
                selected = next(int(idx) for idx in all_indices if int(idx) not in used)
            medoids[cluster_id] = selected
            used.add(selected)
        if anchors.size == 0:
            return torch.from_numpy(medoids)
        selected = list(map(int, anchors))
        used = set(selected)
        selected.extend(int(index) for index in medoids if int(index) not in used)
        if len(selected) < memory_size:
            used = set(selected)
            selected.extend(
                int(index)
                for index in all_indices
                if int(index) not in used
            )
        return torch.tensor(selected[:memory_size], dtype=torch.long)

    def _select_uniform_memory_pairs(self, pair_count: int) -> torch.Tensor:
        memory_size = min(self.gmb_size, pair_count)
        starts = np.rint(
            np.linspace(0, pair_count - 1, memory_size)
        ).astype(np.int64)
        if np.unique(starts).size != memory_size:
            raise RuntimeError("Uniform memory selection produced duplicate windows.")
        return torch.from_numpy(starts)

    def _select_stratified_medoid_memory_pairs(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Select one learned-representation medoid per temporal stratum."""
        pair_count = int(features.shape[0])
        memory_size = min(self.gmb_size, pair_count)
        if memory_size == pair_count:
            return torch.arange(pair_count, dtype=torch.long)

        feature_np = features.numpy().astype(np.float32, copy=False)
        bin_ids = np.minimum(
            np.arange(pair_count, dtype=np.int64) * memory_size // pair_count,
            memory_size - 1,
        )
        counts = np.bincount(bin_ids, minlength=memory_size).astype(np.float32)
        centroids = np.zeros((memory_size, feature_np.shape[1]), dtype=np.float32)
        np.add.at(centroids, bin_ids, feature_np)
        centroids /= counts[:, None]
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)

        scores = np.einsum("nd,nd->n", feature_np, centroids[bin_ids])
        best_scores = np.full(memory_size, -np.inf, dtype=np.float32)
        np.maximum.at(best_scores, bin_ids, scores)
        indices = np.arange(pair_count, dtype=np.int64)
        candidates = np.where(scores == best_scores[bin_ids], indices, pair_count)
        selected = np.full(memory_size, pair_count, dtype=np.int64)
        np.minimum.at(selected, bin_ids, candidates)
        if np.any(selected == pair_count):
            raise RuntimeError("Stratified medoid selection left an empty stratum.")
        return torch.from_numpy(selected)

    def _select_information_arc_memory_pairs(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Sample equal mass on a chronology plus learned-change information axis."""
        pair_count = int(features.shape[0])
        memory_size = min(self.gmb_size, pair_count)
        if memory_size == pair_count:
            return torch.arange(pair_count, dtype=torch.long)

        feature_np = features.numpy().astype(np.float32, copy=False)
        changes = np.zeros(pair_count, dtype=np.float32)
        if pair_count > 1:
            adjacent_cosine = np.einsum(
                "nd,nd->n", feature_np[1:], feature_np[:-1]
            )
            changes[1:] = 1.0 - np.clip(adjacent_cosine, -1.0, 1.0)
        scale = float(np.mean(changes))
        information_steps = 1.0 + changes / max(scale, 1e-6)
        arc = np.cumsum(information_steps, dtype=np.float64)
        targets = (np.arange(memory_size, dtype=np.float64) + 0.5) * (
            arc[-1] / memory_size
        )
        base = np.searchsorted(arc, targets, side="left").astype(np.int64)
        selected = np.empty(memory_size, dtype=np.int64)
        previous = -1
        for rank, index in enumerate(base):
            lower = previous + 1
            upper = pair_count - (memory_size - rank)
            chosen = min(max(int(index), lower), upper)
            selected[rank] = chosen
            previous = chosen
        return torch.from_numpy(selected)

    def _build_memory(self, values: torch.Tensor, device: torch.device) -> None:
        num_windows = values.shape[0] - self.total_window + 1
        if num_windows <= 0:
            raise ValueError(
                f"Training data length {values.shape[0]} is too short for "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}."
            )
        allowed_starts = nested_window_indices(
            num_windows, self.train_window_ratio, self.seed
        )
        pair_ids = torch.from_numpy(
            (
                allowed_starts[:, None] * self.channels
                + np.arange(self.channels, dtype=np.int64)[None, :]
            ).reshape(-1)
        )
        pair_count = int(pair_ids.numel())
        features = self._encode_pairs(values, pair_ids, device)
        if self.retrieval_scope == "same_channel":
            selected_parts = []
            for channel in range(self.channels):
                channel_feature_ids = torch.arange(
                    channel, pair_count, self.channels
                )
                if self.memory_selection == "uniform":
                    local_ids = self._select_uniform_memory_pairs(
                        channel_feature_ids.numel()
                    )
                elif self.memory_selection == "stratified_medoid":
                    local_ids = self._select_stratified_medoid_memory_pairs(
                        features[channel_feature_ids]
                    )
                elif self.memory_selection == "information_arc":
                    local_ids = self._select_information_arc_memory_pairs(
                        features[channel_feature_ids]
                    )
                else:
                    local_ids = self._select_memory_pairs(
                        features[channel_feature_ids], temporal_coverage=True
                    )
                selected_parts.append(channel_feature_ids[local_ids])
            selected_feature_ids = torch.cat(selected_parts)
        else:
            if self.memory_selection == "uniform":
                selected_feature_ids = self._select_uniform_memory_pairs(pair_count)
            else:
                selected_feature_ids = self._select_memory_pairs(features)

        selected_ids = pair_ids[selected_feature_ids]

        future = self._gather_pairs(
            values, selected_ids, self.pred_len, self.seq_len
        )
        _, _, future_norm = _normalize_series(future)
        self.memory_keys = features[selected_feature_ids].contiguous()
        self.memory_future = future_norm.contiguous()
        self.memory_starts = torch.div(
            selected_ids, self.channels, rounding_mode="floor"
        ).contiguous()
        self.memory_channels = torch.remainder(selected_ids, self.channels).contiguous()
        self._apply_future_control()
        if self.memory_keys.shape[0] < self.topk:
            raise ValueError(
                f"Built GMB has {self.memory_keys.shape[0]} items, below memory_topk={self.topk}."
            )

    def _cache_metadata(self, data_hash: str) -> dict[str, Any]:
        metadata = {
            "version": self.CACHE_VERSION,
            "data_hash": data_hash,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "channels": self.channels,
            "embedding_dim": self.embedding_dim,
            "encoder_hidden": self.encoder_hidden,
            "gmb_size": self.gmb_size,
            "pcl_epochs": self.pcl_epochs,
            "pcl_batch_size": self.pcl_batch_size,
            "pcl_lr": self.pcl_lr,
            "pcl_temperature": self.pcl_temperature,
            "retrieval_scope": self.retrieval_scope,
            "temporal_coverage_anchors": self.temporal_coverage_anchors,
            "seed": self.seed,
        }
        if (
            self.retrieval_truth_control != "full"
            and not self.retrieval_truth_use_base_cache
        ):
            metadata["retrieval_truth_control"] = self.retrieval_truth_control
        if self.memory_selection != "medoid":
            metadata["memory_selection"] = self.memory_selection
        return metadata

    def _validate_retrieval_scope(self) -> None:
        if self.retrieval_scope != "same_channel":
            self._same_channel_memory_indices = torch.empty(
                0, 0, dtype=torch.long, device=self.memory_channels.device
            )
            return
        counts = torch.bincount(self.memory_channels, minlength=self.channels)
        insufficient = torch.nonzero(counts < self.topk).flatten().tolist()
        if insufficient:
            details = {channel: int(counts[channel]) for channel in insufficient}
            raise ValueError(
                "Same-channel Memory memory has fewer entries than memory_topk for "
                f"channels {details}. Increase memory_gmb_size."
            )
        if not bool(torch.all(counts == counts[0])):
            details = {channel: int(counts[channel]) for channel in range(self.channels)}
            raise ValueError(
                "Same-channel Memory requires an equal memory size per channel, "
                f"got {details}."
            )
        self._same_channel_memory_indices = torch.stack(
            [
                torch.nonzero(self.memory_channels == channel).flatten()
                for channel in range(self.channels)
            ]
        ).contiguous()

    def prepare_memory(self, values: np.ndarray, device: torch.device) -> None:
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(
                f"Expected training values [T, {self.channels}], got {values.shape}."
            )
        digest = hashlib.sha256(values.view(np.uint8)).hexdigest()[:20]
        metadata = self._cache_metadata(digest)
        num_windows = values.shape[0] - self.total_window + 1
        allowed_starts = nested_window_indices(
            num_windows, self.train_window_ratio, self.seed
        )
        mask_sha = window_indices_sha256(allowed_starts)
        sampling_suffix = ""
        if self.train_window_ratio < 1.0:
            metadata.update(
                {
                    "train_window_ratio": self.train_window_ratio,
                    "train_window_count": int(allowed_starts.size),
                    "train_window_mask_sha256": mask_sha,
                }
            )
            sampling_suffix = (
                f"_ratio{self.train_window_ratio:g}_mask{mask_sha[:12]}"
            )
        medoid_cache_name = (
            f"{digest}_sl{self.seq_len}_pl{self.pred_len}_c{self.channels}_"
            f"d{self.embedding_dim}_m{self.gmb_size}_scope{self.retrieval_scope}_"
            f"cov{self.temporal_coverage_anchors}{sampling_suffix}_seed{self.seed}.pt"
        )
        cache_name = medoid_cache_name
        if self.memory_selection != "medoid":
            cache_name = cache_name.replace(
                f"_seed{self.seed}.pt",
                f"_sel{self.memory_selection}_seed{self.seed}.pt",
            )
        use_base_cache = bool(
            getattr(self, "retrieval_truth_use_base_cache", False)
        ) and self.retrieval_truth_control != "untrained_encoder"
        if self.retrieval_truth_control == "full" or use_base_cache:
            cache_path = self.cache_root / cache_name
        else:
            cache_path = self.cache_root / self.retrieval_truth_control / cache_name
        medoid_cache_path = self.cache_root / medoid_cache_name
        external_stratified_path = None
        if self.encoder_cache_root is not None:
            external_stratified_path = self.encoder_cache_root / medoid_cache_name.replace(
                f"_seed{self.seed}.pt",
                f"_selstratified_medoid_seed{self.seed}.pt",
            )
        self.cache_path = str(cache_path.resolve())

        legacy_metadata = dict(metadata)
        legacy_metadata.pop("temporal_coverage_anchors")
        medoid_metadata = dict(metadata)
        medoid_metadata.pop("memory_selection", None)
        medoid_legacy_metadata = dict(medoid_metadata)
        medoid_legacy_metadata.pop("temporal_coverage_anchors")
        legacy_name = (
            f"{digest}_sl{self.seq_len}_pl{self.pred_len}_c{self.channels}_"
            f"d{self.embedding_dim}_m{self.gmb_size}_scope{self.retrieval_scope}_"
            f"seed{self.seed}.pt"
        )
        legacy_path = self.cache_root / legacy_name

        if cache_path.exists() and self.retrieval_truth_control != "untrained_encoder":
            payload = torch.load(cache_path, map_location="cpu")
            if not self._metadata_compatible(payload.get("metadata", {}), metadata):
                raise ValueError(f"Memory cache metadata mismatch: {cache_path}")
            self.encoder.load_state_dict(payload["encoder_state"])
            self.memory_keys = payload["memory_keys"].float().contiguous()
            self.memory_future = payload["memory_future"].float().contiguous()
            self.memory_starts = payload["memory_starts"].long().contiguous()
            self.memory_channels = payload["memory_channels"].long().contiguous()
            self._apply_future_control()
            self.cache_used = True
            logger.info("Loaded Memory encoder and GMB cache: %s", cache_path)
        else:
            if self.require_existing_cache and self.retrieval_truth_control != "untrained_encoder":
                raise FileNotFoundError(
                    "Required Memory cache was not found; cache construction is "
                    f"disabled: {cache_path}"
                )
            values_tensor = torch.from_numpy(values)
            encoder_cache_path = None
            encoder_cache_metadata = None
            if self.retrieval_truth_control == "untrained_encoder":
                
                
                generator_state = torch.random.get_rng_state()
                torch.manual_seed(self.seed)
                for module in self.encoder.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
                torch.random.set_rng_state(generator_state)
            elif (
                self.memory_selection
                in {"uniform", "stratified_medoid", "information_arc"}
                and medoid_cache_path.exists()
            ):
                encoder_cache_path = medoid_cache_path
                encoder_cache_metadata = medoid_metadata
            elif (
                external_stratified_path is not None
                and external_stratified_path.exists()
            ):
                encoder_cache_path = external_stratified_path
                encoder_cache_metadata = dict(metadata)
                encoder_cache_metadata["memory_selection"] = "stratified_medoid"
            elif (
                self.train_window_ratio == 1.0
                and self.temporal_coverage_anchors > 0
                and legacy_path.exists()
            ):
                encoder_cache_path = legacy_path
                encoder_cache_metadata = (
                    medoid_legacy_metadata
                    if self.memory_selection
                    in {"uniform", "stratified_medoid", "information_arc"}
                    else legacy_metadata
                )
            if encoder_cache_path is not None:
                encoder_payload = torch.load(encoder_cache_path, map_location="cpu")
                payload_metadata = encoder_payload.get("metadata", {})
                expected_metadata = dict(encoder_cache_metadata)
                if self.retrieval_truth_control != "full":
                    payload_metadata = dict(payload_metadata)
                    payload_metadata.pop("retrieval_truth_control", None)
                    expected_metadata.pop("retrieval_truth_control", None)
                if not self._metadata_compatible(payload_metadata, expected_metadata):
                    raise ValueError(
                        f"Memory encoder cache metadata mismatch: {encoder_cache_path}"
                    )
                self.encoder.load_state_dict(encoder_payload["encoder_state"])
                self.encoder_cache_source = str(encoder_cache_path.resolve())
                self.encoder_cache_reused = True
                logger.info(
                    "Reusing Memory encoder for %s memory selection: %s",
                    self.memory_selection,
                    encoder_cache_path,
                )
            else:
                pair_ids = torch.from_numpy(
                    (
                        allowed_starts[:, None] * self.channels
                        + np.arange(self.channels, dtype=np.int64)[None, :]
                    ).reshape(-1)
                )
                training_window_spec = (
                    values.shape[0] - self.total_window + 1
                    if self.train_window_ratio == 1.0
                    else pair_ids
                )
                self._train_predictive_encoder(
                    values_tensor, training_window_spec, device
                )
            self._build_memory(values_tensor, device)
            if self.retrieval_truth_control != "untrained_encoder":
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                {
                    "metadata": metadata,
                    "encoder_state": copy.deepcopy(self.encoder.state_dict()),
                    "memory_keys": self.memory_keys.cpu(),
                    "memory_future": self.memory_future.cpu(),
                    "memory_starts": self.memory_starts.cpu(),
                    "memory_channels": self.memory_channels.cpu(),
                },
                    cache_path,
                )
            self.cache_used = False
            logger.info("Built and cached Memory encoder/GMB: %s", cache_path)

        self._validate_retrieval_scope()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self._prepared = True

    def _save_retrieval_trace(self, x_enc, scale_indices, scale_scores, query_starts=None) -> None:
        """Persist exact Top-K identities/scores for a bounded test batch."""
        if not self._retrieval_trace_dir or not scale_indices or self._retrieval_trace_batch >= 4:
            return
        indices = scale_indices[-1]
        scores = scale_scores[-1] if scale_scores else None
        rows = min(int(indices.shape[0]), 256)
        query_matrix = x_enc.permute(0, 2, 1).reshape(-1, x_enc.shape[1])
        trace_dir = Path(self._retrieval_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_history": query_matrix[:rows].detach().to(dtype=torch.float32).cpu().numpy(),
            "query_channels": (torch.arange(self.channels, device=x_enc.device)
                               .unsqueeze(0).expand(x_enc.shape[0], -1)
                               .reshape(-1)[:rows].cpu().numpy()),
            "neighbor_indices": indices[:rows].detach().cpu().numpy(),
            "neighbor_starts": self.memory_starts[indices[:rows]].detach().cpu().numpy(),
            "neighbor_channels": self.memory_channels[indices[:rows]].detach().cpu().numpy(),
            "neighbor_future": self.memory_future[indices[:rows]].detach().to(dtype=torch.float32).cpu().numpy(),
        }
        if query_starts is not None:
            payload["query_starts"] = query_starts.unsqueeze(1).expand(-1, self.channels).reshape(-1)[:rows].detach().cpu().numpy()
        if scores is not None:
            payload["neighbor_scores"] = scores[:rows].detach().to(dtype=torch.float32).cpu().numpy()
        np.savez_compressed(trace_dir / f"retrieval_trace_{self._retrieval_trace_batch:06d}.npz", **payload)
        self._retrieval_trace_batch += 1

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not self._prepared or self.memory_keys.numel() == 0:
            raise RuntimeError("Memory memory is not prepared. Call prepare_memory first.")
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                f"Expected input [B, {self.seq_len}, {self.channels}], got {tuple(x.shape)}."
            )

        x_ci = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
        query_mean, query_std, x_norm = _normalize_series(x_ci)
        query_keys = self._truth_keys(x_norm)

        query_channels = torch.arange(channels, device=x.device)
        query_channels = query_channels.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        if self.retrieval_scope == "same_channel":
            memory_indices = self._same_channel_memory_indices
            memory_keys = self.memory_keys[memory_indices]
            query_keys, memory_keys = self._transform_retrieval_keys(
                query_keys, memory_keys
            )
            similarity = torch.bmm(
                query_keys.reshape(batch_size, channels, -1).permute(1, 0, 2),
                memory_keys.transpose(1, 2),
            )
            similarity = similarity.permute(1, 0, 2).reshape(
                batch_size * channels, -1
            )
        else:
            memory_indices = None
            query_keys, memory_keys = self._transform_retrieval_keys(
                query_keys, self.memory_keys
            )
            similarity = torch.matmul(query_keys, memory_keys.transpose(0, 1))
        similarity = self._control_similarity(similarity)
        if mode == "train":
            if index is None:
                raise ValueError("Training Memory retrieval requires window indices.")
            if memory_indices is not None:
                query_starts = index.to(x.device).reshape(batch_size, 1, 1)
                memory_starts = self.memory_starts[memory_indices].unsqueeze(0)
                overlap = torch.abs(query_starts - memory_starts) < self.total_window
                overlap = overlap.reshape(batch_size * channels, -1)
            else:
                query_starts = (
                    index.to(x.device).unsqueeze(1).expand(-1, channels).reshape(-1)
                )
                overlap = (
                    torch.abs(
                        query_starts.unsqueeze(1) - self.memory_starts.unsqueeze(0)
                    )
                    < self.total_window
                )
            similarity = similarity.masked_fill(overlap, float("-inf"))

        topk_similarity, topk_indices = torch.topk(similarity, self.topk, dim=1)
        if not bool(torch.isfinite(topk_similarity).all()):
            raise RuntimeError(
                "Memory GMB does not contain enough non-overlapping neighbors for a training query."
            )
        if memory_indices is not None:
            topk_indices = memory_indices[
                query_channels.unsqueeze(1), topk_indices
            ]
        topk_future = self.memory_future[topk_indices]

        confidence_elements = (
            topk_future.shape[0]
            * topk_future.shape[1]
            * (self.seq_len + self.pred_len)
        )
        use_factorized = self.confidence_execution == "factorized" or (
            self.confidence_execution == "adaptive"
            and confidence_elements >= self.factorized_confidence_min_elements
        )
        if use_factorized:
            first_layer = self.confidence_gate[0]
            query_hidden = F.linear(
                x_norm,
                first_layer.weight[:, : self.seq_len],
                first_layer.bias,
            ).unsqueeze(1)
            future_hidden = self._project_topk_future(
                topk_future, topk_indices, first_layer, mode
            )
            confidence = self.confidence_gate[3](
                self.confidence_gate[2](
                    self.confidence_gate[1](query_hidden + future_hidden)
                )
            ).squeeze(-1)
            self._last_confidence_execution = "factorized"
        else:
            query_expanded = x_norm.unsqueeze(1).expand(-1, self.topk, -1)
            confidence = self.confidence_gate(
                torch.cat([query_expanded, topk_future], dim=-1)
            ).squeeze(-1)
            self._last_confidence_execution = "reference"
        adjusted_scores = self._adjust_retrieval_scores(
            topk_similarity, confidence
        )
        weights = self._compute_neighbor_weights(adjusted_scores)
        weights = self._refine_neighbor_weights(weights, topk_future)
        retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
        if getattr(self, "reliability_enabled", False):
            from .reliability import reliability
            self._last_candidate_reliability = reliability(topk_future, weights, self.reliability_strength).reshape(batch_size, channels)

        output_scale, output_shift = self.output_gate(x_norm)
        retrieved = retrieved * output_scale + output_shift
        retrieved = retrieved * query_std + query_mean
        retrieval_pred = retrieved.reshape(batch_size, channels, self.pred_len)
        retrieval_pred = retrieval_pred.permute(0, 2, 1).contiguous()
        if self.retrieval_truth_control == "zero_retrieval":
            retrieval_pred = torch.zeros_like(retrieval_pred)

        source_channels = self.memory_channels[topk_indices]
        diagnostics = {
            "confidence_mean": confidence.detach().mean(),
            "same_channel_ratio": (
                source_channels == query_channels.unsqueeze(1)
            ).float().detach().mean(),
        }
        fusion_scores = self._scores_for_fusion(
            topk_similarity, confidence, adjusted_scores
        )
        if self.retrieval_truth_control == "zero_retrieval":
            
            
            fusion_scores = torch.zeros_like(fusion_scores)
        fusion_scores = self._augment_fusion_scores(
            fusion_scores, topk_future, weights
        )
        self._last_scale_indices = [topk_indices]
        self._last_scale_scores = [fusion_scores]
        return (
            retrieval_pred,
            fusion_scores.reshape(batch_size, channels, -1),
            diagnostics,
        )

    def get_memory_summary(self) -> dict[str, Any]:
        summary = {
            "memory_cache_path": self.cache_path,
            "memory_cache_used": self.cache_used,
            "memory_gmb_size": int(self.memory_keys.shape[0]),
            "memory_topk": self.topk,
            "memory_retrieval_scope": self.retrieval_scope,
            "memory_memory_selection": self.memory_selection_requested,
            "memory_memory_selection_effective": self.memory_selection,
            "memory_uniform_memory_min_channels": self.uniform_memory_min_channels,
            "memory_confidence_execution": self.confidence_execution,
            "memory_factorized_confidence_min_elements": self.factorized_confidence_min_elements,
            "memory_confidence_last_execution": self._last_confidence_execution,
            "memory_encoder_cache_reused": self.encoder_cache_reused,
            "memory_retrieval_truth_control": self.retrieval_truth_control,
        }
        if self.encoder_cache_source is not None:
            summary["memory_encoder_cache_source"] = self.encoder_cache_source
        if self.retrieval_scope == "same_channel":
            counts = torch.bincount(
                self.memory_channels,
                minlength=self.channels,
            )
            summary["memory_gmb_size_per_channel"] = counts.tolist()
        return summary


class IndependentMemoryBranch(nn.Module):
    """One complete Memory encoder and memory bank per target channel."""

    def __init__(self, configs):
        super().__init__()
        self.channels = int(configs.enc_in)
        self.topk = int(getattr(configs, "memory_topk", 10))
        branches = []
        for _ in range(self.channels):
            channel_config = copy.copy(configs)
            channel_config.enc_in = 1
            channel_config.dec_in = 1
            channel_config.c_out = 1
            channel_config.memory_retrieval_scope = "global"
            branches.append(MultivariateMemoryBranch(channel_config))
        self.branches = nn.ModuleList(branches)

    def prepare_memory(self, values: np.ndarray, device: torch.device) -> None:
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(
                f"Expected training values [T, {self.channels}], got {values.shape}."
            )
        for channel, branch in enumerate(self.branches):
            logger.info(
                "Preparing independent Memory memory for channel %d/%d.",
                channel + 1,
                self.channels,
            )
            branch.prepare_memory(values[:, channel : channel + 1], device)

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        predictions = []
        scores = []
        confidences = []
        same_channel_ratios = []
        for channel, branch in enumerate(self.branches):
            prediction, channel_scores, diagnostics = branch(
                x[:, :, channel : channel + 1],
                index,
                mode,
            )
            predictions.append(prediction)
            scores.append(channel_scores)
            confidences.append(diagnostics["confidence_mean"])
            same_channel_ratios.append(diagnostics["same_channel_ratio"])

        return (
            torch.cat(predictions, dim=2),
            torch.cat(scores, dim=1),
            {
                "confidence_mean": torch.stack(confidences).mean(),
                "same_channel_ratio": torch.stack(same_channel_ratios).mean(),
            },
        )

    def get_memory_summary(self) -> dict[str, Any]:
        return {
            "memory_cache_path": [branch.cache_path for branch in self.branches],
            "memory_cache_used": all(branch.cache_used for branch in self.branches),
            "memory_gmb_size": sum(
                int(branch.memory_keys.shape[0]) for branch in self.branches
            ),
            "memory_gmb_size_per_channel": [
                int(branch.memory_keys.shape[0]) for branch in self.branches
            ],
            "memory_topk": self.topk,
            "memory_retrieval_scope": "independent",
        }

