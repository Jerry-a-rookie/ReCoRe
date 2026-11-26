from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.recore_public._engine.memory_branch import _normalize_series
from ts_benchmark.baselines.recore_public._engine.geometry_branch import GeometryBranch
from ts_benchmark.baselines.recore_public._engine.window_sampling import nested_window_indices
from ts_benchmark.baselines.recore_public._engine.geometry_model import GeometryModel, GeometryLearnedFusion
from ts_benchmark.baselines.recore_public._engine.stablerel_controls import CONTROLS as RELATION_CONTROLS, build_relation


logger = logging.getLogger(__name__)


class RelationalPredictiveContrastGeometryBranch(GeometryBranch):
    """Train Geometry keys from target futures and synchronized peer-relative roles."""

    CACHE_VERSION = 6
    CONTROLS = {"zero_peer", "aligned", "time_shuffled"} | RELATION_CONTROLS

    def __init__(self, configs):
        super().__init__(configs)
        self.relational_control = str(
            getattr(configs, "memory_relational_contrast_control", "aligned")
        )
        self.relational_control_seed = int(
            getattr(configs, "memory_relational_contrast_control_seed", self.seed)
        )
        self.relational_max_peers = int(
            getattr(configs, "memory_relational_max_peers", 0)
        )
        if not 0 <= self.relational_max_peers <= max(self.channels - 1, 0):
            raise ValueError(
                "memory_relational_max_peers must be 0 (all peers) or between "
                f"1 and {max(self.channels - 1, 0)}"
            )
        self.require_existing_cache = bool(
            getattr(
                configs,
                "memory_relational_require_existing_cache",
                self.require_existing_cache,
            )
        )
        artifact = getattr(configs, "memory_relational_artifact_path", None)
        self.relation_artifact_path = Path(artifact) if artifact else None
        self.require_existing_relation_artifact = bool(
            getattr(configs, "memory_require_existing_relation_artifact", False)
        )
        if self.relational_control not in self.CONTROLS:
            raise ValueError(
                "memory_relational_contrast_control must be one of "
                f"{sorted(self.CONTROLS)}"
            )
        self.cache_lock_timeout_seconds = float(
            getattr(configs, "memory_relational_cache_lock_timeout_seconds", 6 * 60 * 60)
        )
        if self.cache_lock_timeout_seconds <= 0:
            raise ValueError("memory_relational_cache_lock_timeout_seconds must be positive")
        self._relation_summary: dict[str, Any] = {
            "coherence_mean": 0.0,
            "coherence_min": 0.0,
            "coherence_max": 0.0,
            "nonzero_fraction": 0.0,
            "edge_density": 0.0,
            "positive_edges": 0,
            "negative_edges": 0,
            "effective_peers_mean": 0.0,
        }
        self._prepared_relation_weights: torch.Tensor | None = None
        self._prepared_coherence: torch.Tensor | None = None
        self._prepared_relative_values: torch.Tensor | None = None
        
        
        
        
        
        
        
        self._prepared_values_device: torch.Tensor | None = None
        self._prepared_relative_values_device: torch.Tensor | None = None
        self._prepared_coherence_device: torch.Tensor | None = None

    @staticmethod
    def _correlation_matrix(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        scaled = centered / centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        return scaled.transpose(0, 1).matmul(scaled) / max(int(values.shape[0]), 1)

    @classmethod
    def _relation_signal(cls, values: torch.Tensor) -> torch.Tensor:
        """Signal used to estimate the static relation graph.

        Experimental RoleGeo studies override this hook.  The canonical branch
        keeps its historical first-difference definition exactly.
        """
        return values[1:] - values[:-1]

    @classmethod
    def _role_signal(cls, values: torch.Tensor) -> torch.Tensor:
        """Signal used by the relational teacher before optional left padding."""
        return cls._relation_signal(values)

    @classmethod
    def _stable_relation_weights(
        cls, values: torch.Tensor, max_peers: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return signed peer weights and split-half-consistent coherence."""
        differences = cls._relation_signal(values)
        midpoint = int(differences.shape[0]) // 2
        if midpoint < 2 or differences.shape[0] - midpoint < 2:
            raise ValueError("Relational contrast requires two non-trivial training halves.")
        first = cls._correlation_matrix(differences[:midpoint])
        second = cls._correlation_matrix(differences[midpoint:])
        stable = (first * second).clamp_min(0.0).sqrt()
        signed = torch.sign(first + second) * stable
        signed.fill_diagonal_(0.0)
        stable.fill_diagonal_(0.0)
        max_peers = int(max_peers)
        if max_peers < 0 or max_peers > max(values.shape[1] - 1, 0):
            raise ValueError("max_peers is outside the available peer range")
        if 0 < max_peers < values.shape[1] - 1:
            selected = torch.topk(stable, max_peers, dim=1).indices
            active = torch.zeros_like(stable, dtype=torch.bool)
            active.scatter_(1, selected, True)
            stable = stable * active
            signed = signed * active
        strength_sum = stable.sum(dim=1)
        weights = signed / strength_sum.clamp_min(1e-8).unsqueeze(1)
        coherence = stable.square().sum(dim=1) / strength_sum.clamp_min(1e-8)
        coherence = coherence.clamp(0.0, 1.0)
        empty = strength_sum <= 1e-8
        weights[empty] = 0.0
        coherence[empty] = 0.0
        return weights, coherence

    def _cache_metadata(self, data_hash: str) -> dict[str, Any]:
        metadata = super()._cache_metadata(data_hash)
        metadata.update(
            {
                "relational_predictive_contrast_version": self.CACHE_VERSION,
                "relational_predictive_contrast_control": self.relational_control,
                "relational_predictive_contrast_control_seed": self.relational_control_seed,
                "relational_predictive_contrast_relation": (
                    "split_half_consistent_first_difference_correlation"
                ),
                "relational_predictive_contrast_response": (
                    "target_minus_synchronized_peer_dynamics_composite"
                ),
            }
        )
        
        
        if self.relational_max_peers:
            metadata["relational_predictive_contrast_max_peers"] = (
                self.relational_max_peers
            )
        return metadata

    def _relation_key_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Deterministic signed relation signal injected into retrieval keys."""
        weights = self._prepared_relation_weights
        if weights is None:
            return torch.zeros(self.channels, device=device, dtype=dtype)
        indices = torch.arange(1, self.channels + 1, device=weights.device, dtype=weights.dtype)
        signal = (weights * indices.unsqueeze(0)).sum(dim=1)
        scale = signal.abs().mean().clamp_min(1e-6)
        return (0.05 * signal / scale).to(device=device, dtype=dtype)

    def _truth_keys(self, x: torch.Tensor) -> torch.Tensor:
        keys = super()._truth_keys(x)
        bias = self._relation_key_bias(keys.device, keys.dtype)
        if bias.numel() and keys.ndim == 2 and keys.shape[0] % self.channels == 0:
            keys = keys.clone()
            keys[:, 0] = keys[:, 0] + bias.repeat(keys.shape[0] // self.channels)
            keys = F.normalize(keys, dim=-1)
        return keys

    def _encode_pairs(self, values: torch.Tensor, pair_ids: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        """Encode memory rows with the actual channel of each window pair."""
        if isinstance(pair_ids, int):
            pair_ids = torch.arange(pair_ids, dtype=torch.long)
        outputs = []
        self.encoder.to(device)
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, pair_ids.numel(), self.encode_batch_size):
                current = pair_ids[start : start + self.encode_batch_size]
                history = self._gather_pairs(values, current, self.seq_len, 0).to(device)
                _, _, history = _normalize_series(history)
                keys = super()._truth_keys(history)
                channels = torch.remainder(current.to(device), self.channels)
                bias = self._relation_key_bias(device, keys.dtype)
                keys = keys.clone()
                keys[:, 0] += bias[channels]
                outputs.append(F.normalize(keys, dim=-1).cpu())
        self.encoder.cpu()
        return torch.cat(outputs, dim=0)

    @contextlib.contextmanager
    def _cache_build_lock(self, values: np.ndarray):
        """Serialize only duplicate relational-cache builders.

        Cache contents are already validated by the parent implementation.  The
        lock merely gives identical concurrent jobs a single builder: waiters
        re-enter ``super().prepare_memory`` after the owner has atomically
        finished writing the usual cache artifact.  Distinct metadata (notably
        different PLs) receive distinct locks and still build in parallel.
        """
        digest = hashlib.sha256(values.view(np.uint8)).hexdigest()[:20]
        identity = json.dumps(self._cache_metadata(digest), sort_keys=True)
        token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        lock_path = Path(self.cache_root) / f".{token}.relational-build.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_seconds = self.cache_lock_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        descriptor: int | None = None
        try:
            while descriptor is None:
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    os.write(
                        descriptor,
                        f"pid={os.getpid()}\n".encode("ascii"),
                    )
                except FileExistsError:
                    
                    
                    
                    try:
                        owner = lock_path.read_text(encoding="ascii").strip()
                        owner_pid = int(owner.split("=", 1)[1])
                        os.kill(owner_pid, 0)
                    except (OSError, ValueError, IndexError):
                        try:
                            lock_path.unlink()
                            continue
                        except FileNotFoundError:
                            continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for relational Geometry cache builder: "
                            f"{lock_path}"
                        )
                    time.sleep(0.2)
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def prepare_memory(self, values: np.ndarray, device: torch.device) -> None:
        values = np.ascontiguousarray(values, dtype=np.float32)
        values_tensor = torch.from_numpy(values)
        try:
            if self.relational_control == "zero_peer":
                relation_weights = torch.zeros(self.channels, self.channels)
                coherence = torch.zeros(self.channels)
                relative_values = torch.zeros_like(values_tensor)
            else:
                values_device = values_tensor.to(device)
                relation_values_device = values_device
                if self.train_window_ratio < 1.0:
                    num_windows = values.shape[0] - self.total_window + 1
                    allowed_starts = nested_window_indices(
                        num_windows, self.train_window_ratio, self.seed
                    )
                    starts_tensor = torch.from_numpy(allowed_starts).to(device)
                    offsets = torch.arange(self.total_window, device=device)
                    relation_values_device = values_device[
                        starts_tensor.unsqueeze(1) + offsets.unsqueeze(0)
                    ].reshape(-1, self.channels)
                if self.relational_control in RELATION_CONTROLS:
                    relation_weights_device, coherence_device = build_relation(
                        relation_values_device,
                        self.relational_control,
                        self.relational_control_seed,
                    )
                else:
                    relation_weights_device, coherence_device = self._stable_relation_weights(
                        relation_values_device, self.relational_max_peers
                    )
                role_signal = self._role_signal(values_device)
                normalized_signal = role_signal / role_signal.std(
                    dim=0, keepdim=True
                ).clamp_min(1e-6)
                peer_signal = normalized_signal
                if self.relational_control == "time_shuffled" and self.channels > 1:
                    num_windows = values_tensor.shape[0] - self.total_window + 1
                    stride = max(self.total_window + 1, num_windows // self.channels)
                    
                    
                    
                    
                    length = normalized_signal.shape[0]
                    shifts = (
                        torch.arange(self.channels, device=device) * stride
                    ) % length
                    rows = (
                        torch.arange(length, device=device).unsqueeze(1)
                        - shifts.unsqueeze(0)
                    ) % length
                    columns = torch.arange(self.channels, device=device)
                    peer_signal = normalized_signal[rows, columns]
                relative_values_device = normalized_signal - peer_signal.matmul(
                    relation_weights_device.transpose(0, 1)
                )
                if relative_values_device.shape[0] == values_device.shape[0] - 1:
                    relative_values_device = torch.cat(
                        (
                            relative_values_device.new_zeros(1, self.channels),
                            relative_values_device,
                        ),
                        dim=0,
                    )
                elif relative_values_device.shape[0] != values_device.shape[0]:
                    raise ValueError(
                        "Role signal must have length T or T-1 relative to training values."
                    )
                relative_values = relative_values_device.cpu()
                relation_weights = relation_weights_device.cpu()
                coherence = coherence_device.cpu()
                self._prepared_values_device = values_device
                self._prepared_relative_values_device = relative_values_device
                self._prepared_coherence_device = coherence_device
            self._prepared_relation_weights = relation_weights
            self._prepared_coherence = coherence
            self._prepared_relative_values = relative_values
            if self.relation_artifact_path is not None:
                payload = {
                    "version": 1,
                    "control": self.relational_control,
                    "seed": self.relational_control_seed,
                    "weights": relation_weights,
                    "coherence": coherence,
                }
                if self.require_existing_relation_artifact:
                    if not self.relation_artifact_path.is_file():
                        raise FileNotFoundError(self.relation_artifact_path)
                    existing = torch.load(
                        self.relation_artifact_path, map_location="cpu", weights_only=True
                    )
                    if existing.get("control") != self.relational_control or int(existing.get("seed", -1)) != self.relational_control_seed:
                        raise ValueError("StableRel artifact identity mismatch")
                    if not torch.equal(existing["weights"], relation_weights) or not torch.equal(existing["coherence"], coherence):
                        raise ValueError("StableRel artifact tensor mismatch")
                else:
                    self.relation_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.relation_artifact_path.with_suffix(
                        self.relation_artifact_path.suffix + f".{os.getpid()}.tmp"
                    )
                    torch.save(payload, temporary)
                    os.replace(temporary, self.relation_artifact_path)
            nonzero = coherence > 1e-8
            self._relation_summary = {
                "coherence_mean": float(coherence.mean()),
                "coherence_min": float(coherence.min()),
                "coherence_max": float(coherence.max()),
                "nonzero_fraction": float(nonzero.float().mean()),
                "edge_density": float((relation_weights != 0).float().mean()),
                "positive_edges": int((relation_weights > 0).sum()),
                "negative_edges": int((relation_weights < 0).sum()),
                "effective_peers_mean": float(
                    (relation_weights != 0).sum(dim=1).float().mean()
                ),
            }
            if self.require_existing_cache:
                super().prepare_memory(values, device)
            else:
                with self._cache_build_lock(values):
                    super().prepare_memory(values, device)
            
            
            
            if self.cache_used and self.memory_keys.numel():
                pair_ids = self.memory_starts.cpu() * self.channels + self.memory_channels.cpu()
                self.memory_keys = self._encode_pairs(values_tensor, pair_ids, device)
        finally:
            self._prepared_values_device = None
            self._prepared_relative_values_device = None
            self._prepared_coherence_device = None

    def _train_predictive_encoder(
        self,
        values: torch.Tensor,
        num_windows: int | torch.Tensor,
        device: torch.device,
    ) -> None:
        if self.relational_control == "zero_peer":
            super()._train_predictive_encoder(values, num_windows, device)
            return

        if isinstance(num_windows, torch.Tensor):
            eligible_pair_ids = num_windows.long().cpu()
        else:
            eligible_pair_ids = torch.arange(
                int(num_windows) * self.channels, dtype=torch.long
            )
        pair_count = int(eligible_pair_ids.numel())
        if pair_count < 2:
            raise ValueError("Relational predictive contrast requires two pairs.")
        coherence = self._prepared_coherence
        relative_values = self._prepared_relative_values
        if coherence is None or relative_values is None:
            raise RuntimeError("Relational peer dynamics were not prepared.")

        
        
        values_for_gather = self._prepared_values_device
        if values_for_gather is None:
            values_for_gather = values.to(device)
        relative_for_gather = self._prepared_relative_values_device
        if relative_for_gather is None:
            relative_for_gather = relative_values.to(device)
        coherence_for_gather = self._prepared_coherence_device
        if coherence_for_gather is None:
            coherence_for_gather = coherence.to(device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        self.encoder.to(device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.pcl_lr)

        for epoch in range(self.pcl_epochs):
            self.encoder.train()
            order = torch.randperm(pair_count, generator=generator)
            losses = []
            for start in range(0, pair_count, self.pcl_batch_size):
                pair_ids = eligible_pair_ids[
                    order[start : start + self.pcl_batch_size]
                ]
                if pair_ids.numel() < 2:
                    continue

                pair_ids_device = pair_ids.to(device)
                starts = torch.div(
                    pair_ids_device, self.channels, rounding_mode="floor"
                )
                channels = torch.remainder(pair_ids_device, self.channels)
                x = self._gather_pairs(
                    values_for_gather, pair_ids_device, self.seq_len, 0
                )
                target_future = self._gather_pairs(
                    values_for_gather, pair_ids_device, self.pred_len, self.seq_len
                )
                relative_future = self._gather_pairs(
                    relative_for_gather,
                    pair_ids_device,
                    self.pred_len,
                    self.seq_len,
                )
                _, _, x_norm = _normalize_series(x)
                _, _, target_future = _normalize_series(target_future)
                _, _, relative_future = _normalize_series(relative_future)

                features = F.normalize(self.encoder(x_norm), dim=-1)
                pair_coherence = coherence_for_gather[channels]
                signature = torch.cat(
                    (
                        torch.sqrt(1.0 - pair_coherence).unsqueeze(1)
                        * target_future,
                        torch.sqrt(pair_coherence).unsqueeze(1) * relative_future,
                    ),
                    dim=1,
                )
                overlap = (
                    torch.abs(starts.unsqueeze(1) - starts.unsqueeze(0))
                    < self.total_window
                )
                legal_rows = (~overlap).any(dim=1)
                if not bool(legal_rows.any()):
                    continue

                future_distance = torch.cdist(signature, signature, p=2)
                future_distance = future_distance.masked_fill(overlap, float("inf"))
                positive = torch.argmin(future_distance, dim=1)
                logits = torch.matmul(features, features.transpose(0, 1))
                logits = (logits / self.pcl_temperature).masked_fill(
                    overlap, float("-inf")
                )
                loss = F.cross_entropy(logits[legal_rows], positive[legal_rows])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            if not losses:
                raise RuntimeError(
                    "Relational predictive contrast produced no legal batches."
                )
            logger.info(
                "Relational predictive contrast epoch %d/%d control=%s loss=%.6f",
                epoch + 1,
                self.pcl_epochs,
                self.relational_control,
                float(np.mean(losses)),
            )
        self.encoder.cpu()

    def get_memory_summary(self) -> dict[str, Any]:
        summary = super().get_memory_summary()
        summary.update(
            {
                "architecture": "RelationalPredictiveContrastGeometry",
                "Geometry_relational_contrast_control": self.relational_control,
                "Geometry_relational_contrast_control_seed": self.relational_control_seed,
                "Geometry_relational_contrast_max_peers": self.relational_max_peers,
                "Geometry_relational_contrast_artifact": str(self.relation_artifact_path) if self.relation_artifact_path else None,
                "Geometry_relational_contrast_artifact_verified": bool(self.relation_artifact_path and self.require_existing_relation_artifact),
                "Geometry_relational_contrast_positive_scope": "original_global_pairs",
                "Geometry_relational_contrast_relation": (
                    "split_half_consistent_first_difference_correlation"
                ),
                "Geometry_relational_contrast_response": (
                    "target_minus_synchronized_peer_dynamics_composite"
                ),
                "Geometry_relational_contrast_coherence_mean": self._relation_summary[
                    "coherence_mean"
                ],
                "Geometry_relational_contrast_coherence_min": self._relation_summary[
                    "coherence_min"
                ],
                "Geometry_relational_contrast_coherence_max": self._relation_summary[
                    "coherence_max"
                ],
                "Geometry_relational_contrast_nonzero_fraction": self._relation_summary[
                    "nonzero_fraction"
                ],
                "Geometry_relational_contrast_edge_density": self._relation_summary["edge_density"],
                "Geometry_relational_contrast_positive_edges": self._relation_summary["positive_edges"],
                "Geometry_relational_contrast_negative_edges": self._relation_summary["negative_edges"],
                "Geometry_relational_contrast_effective_peers_mean": self._relation_summary["effective_peers_mean"],
                "Geometry_relational_contrast_inference_overhead": "none",
                "Geometry_target_specific_topk": True,
                "Geometry_target_future_only": True,
            }
        )
        return summary


class RelationalPredictiveContrastGeometryModel(GeometryModel):
    def __init__(self, configs, backbone_class: type[nn.Module]):
        nn.Module.__init__(self)
        self.pred_len = int(configs.pred_len)
        self.memory_variant = "same_channel"
        self.fusion_mode = "learned"
        self.fixed_gamma = 0.0
        self.backbone = backbone_class(configs)
        self.backbone_name = backbone_class.__name__
        self.memory_branch = RelationalPredictiveContrastGeometryBranch(configs)
        self.fusion = GeometryLearnedFusion(
            self.memory_branch.topk,
            int(getattr(configs, "memory_fusion_hidden", 128)),
        )
        self._debug = self._empty_debug()

    def get_debug_summary(self) -> dict:
        summary = super().get_debug_summary()
        summary["architecture"] = "RelationalPredictiveContrastGeometry"
        return summary



