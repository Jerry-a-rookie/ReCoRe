from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.coreg_public._engine.memory_branch import _normalize_series
from ts_benchmark.baselines.coreg_public._engine.geometry_branch import GeometryBranch
from ts_benchmark.baselines.coreg_public._engine.geometry_model import GeometryModel
from ts_benchmark.baselines.coreg_public._engine.stage_profile import stage
from .reliability import moving_average_components, reliability as candidate_reliability


def block_mean_repeat(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Return a non-overlapping block average expanded to the input length."""
    factor = int(factor)
    if factor <= 0 or x.shape[-1] % factor:
        raise ValueError(
            f"factor={factor} must divide temporal length {x.shape[-1]}"
        )
    if factor == 1:
        return x
    shape = (*x.shape[:-1], x.shape[-1] // factor, factor)
    return x.reshape(shape).mean(dim=-1).repeat_interleave(factor, dim=-1)


def log_rms_energy(x: torch.Tensor) -> torch.Tensor:
    """Return a stable logarithmic RMS for the last dimension."""
    return 0.5 * x.square().mean(dim=-1).clamp_min(1e-12).log()


def temporal_laplacian(x: torch.Tensor, factors: tuple[int, ...]) -> torch.Tensor:
    """Split a trajectory into additive coarse-to-fine temporal components."""
    factors = tuple(sorted({int(value) for value in factors}, reverse=True))
    if not factors or factors[-1] != 1:
        raise ValueError("multiscale factors must contain 1")
    lowpasses = [block_mean_repeat(x, factor) for factor in factors]
    components = [lowpasses[0]]
    components.extend(
        lowpasses[index] - lowpasses[index - 1]
        for index in range(1, len(lowpasses))
    )
    return torch.stack(components, dim=0)


def temporal_laplacian_component(
    x: torch.Tensor, factors: tuple[int, ...], index: int
) -> torch.Tensor:
    """Return one Laplacian component without materializing the other bands."""
    factors = tuple(sorted({int(value) for value in factors}, reverse=True))
    if not factors or factors[-1] != 1:
        raise ValueError("multiscale factors must contain 1")
    if not 0 <= int(index) < len(factors):
        raise IndexError("Laplacian component index is out of range")
    factor = factors[int(index)]
    lowpass = block_mean_repeat(x, factor)
    if index == 0:
        return lowpass
    return lowpass - block_mean_repeat(x, factors[int(index) - 1])


def anchored_laplacian_reconstruction(
    anchor: torch.Tensor,
    replacement_components: torch.Tensor,
    factors: tuple[int, ...],
) -> torch.Tensor:
    """Replace coarse components while retaining an anchor's finest detail."""
    anchor_components = temporal_laplacian(anchor, factors)
    if replacement_components.shape != anchor_components[:-1].shape:
        raise ValueError("replacement components do not match the coarse scales")
    return replacement_components.sum(dim=0) + anchor_components[-1]


def coherent_multiscale_similarity(similarities: torch.Tensor) -> torch.Tensor:
    """Combine scale evidence without allowing one scale's spread to dominate."""
    if similarities.ndim != 3:
        raise ValueError("expected similarities [scale, query, memory]")
    finite = torch.isfinite(similarities)
    count = finite.sum(dim=-1, keepdim=True).clamp_min(1)
    safe = torch.where(finite, similarities, torch.zeros_like(similarities))
    mean = safe.sum(dim=-1, keepdim=True) / count
    centered = torch.where(finite, similarities - mean, torch.zeros_like(safe))
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    
    
    scale = variance.clamp_min(1e-12).sqrt()
    standardized = torch.where(
        finite,
        centered / scale,
        torch.full_like(centered, float("-inf")),
    )
    return standardized.mean(dim=0)


def exact_candidate_consensus_scores(
    exact_scores: torch.Tensor, scale_scores: torch.Tensor
) -> torch.Tensor:
    """Calibrate an Exact TopK set by agreement across temporal scales."""
    if exact_scores.ndim != 2 or scale_scores.ndim != 3:
        raise ValueError("expected exact [query, neighbor] and scale [scale, query, neighbor]")
    if scale_scores.shape[1:] != exact_scores.shape:
        raise ValueError("scale evidence must describe the same Exact candidates")
    return coherent_multiscale_similarity(
        torch.cat([exact_scores.unsqueeze(0), scale_scores], dim=0)
    )


def moment_anchored_consensus_scores(
    exact_scores: torch.Tensor, scale_scores: torch.Tensor
) -> torch.Tensor:
    """Change consensus rank geometry while preserving Exact score moments."""
    consensus = exact_candidate_consensus_scores(exact_scores, scale_scores)
    consensus_mean = consensus.mean(dim=-1, keepdim=True)
    consensus_std = consensus.square().mean(dim=-1, keepdim=True).sqrt()
    normalized = (consensus - consensus_mean) / consensus_std.clamp_min(1e-6)
    exact_mean = exact_scores.mean(dim=-1, keepdim=True)
    exact_std = (
        (exact_scores - exact_mean).square().mean(dim=-1, keepdim=True).sqrt()
    )
    return exact_mean + exact_std * normalized


def learned_candidate_consensus_scores(
    exact_scores: torch.Tensor,
    scale_scores: torch.Tensor,
    evidence_logits: torch.Tensor,
) -> torch.Tensor:
    """Combine standardized Exact and scale evidence on a learned simplex."""
    if evidence_logits.shape != (scale_scores.shape[0] + 1,):
        raise ValueError("evidence logits must contain Exact plus every scale")
    evidence = torch.cat([exact_scores.unsqueeze(0), scale_scores], dim=0)
    centered = evidence - evidence.mean(dim=-1, keepdim=True)
    standardized = centered / centered.square().mean(
        dim=-1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    weights = torch.softmax(evidence_logits, dim=0)
    return torch.sum(weights[:, None, None] * standardized, dim=0)


def predictive_fidelity_weights(
    exact_memory_keys: torch.Tensor, scale_memory_keys: torch.Tensor
) -> torch.Tensor:
    """Estimate scale trust from preservation of frozen predictive semantics."""
    if exact_memory_keys.ndim != 2 or scale_memory_keys.ndim != 3:
        raise ValueError("expected Exact [memory, dim] and scale [scale, memory, dim]")
    if scale_memory_keys.shape[1:] != exact_memory_keys.shape:
        raise ValueError("Exact and scale memory keys must align")
    fidelity = torch.sum(
        scale_memory_keys * exact_memory_keys.unsqueeze(0), dim=-1
    ).mean(dim=-1).clamp_min(0.0)
    raw = torch.cat([fidelity.new_ones(1), fidelity])
    return raw / raw.sum().clamp_min(1e-6)


def negative_consensus_veto_scores(
    exact_scores: torch.Tensor, scale_scores: torch.Tensor
) -> torch.Tensor:
    """Use multiscale evidence only to demote candidates it jointly rejects."""
    if scale_scores.ndim != 3 or scale_scores.shape[1:] != exact_scores.shape:
        raise ValueError("scale evidence must describe the same Exact candidates")
    scale_consensus = coherent_multiscale_similarity(scale_scores)
    exact_mean = exact_scores.mean(dim=-1, keepdim=True)
    exact_std = (
        (exact_scores - exact_mean).square().mean(dim=-1, keepdim=True).sqrt()
    )
    return exact_scores - exact_std * F.relu(-scale_consensus)


def standardized_negative_veto_scores(
    exact_scores: torch.Tensor, scale_scores: torch.Tensor
) -> torch.Tensor:
    """Apply a one-sided multiscale veto in a shared standardized space."""
    if scale_scores.ndim != 3 or scale_scores.shape[1:] != exact_scores.shape:
        raise ValueError("scale evidence must describe the same Exact candidates")
    exact_mean = exact_scores.mean(dim=-1, keepdim=True)
    exact_z = (exact_scores - exact_mean) / (
        (exact_scores - exact_mean)
        .square()
        .mean(dim=-1, keepdim=True)
        .sqrt()
        .clamp_min(1e-6)
    )
    scale_consensus = coherent_multiscale_similarity(scale_scores)
    return exact_z - F.relu(-scale_consensus)


def posterior_mean_reliability(
    futures: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a weighted mean and its scale-free posterior consistency."""
    if futures.ndim != 3 or weights.shape != futures.shape[:2]:
        raise ValueError("expected futures [query, neighbor, time] and matching weights")
    mean = torch.sum(weights.unsqueeze(-1) * futures, dim=1)
    residual_energy = torch.sum(
        weights * (futures - mean.unsqueeze(1)).square().mean(dim=-1), dim=1
    )
    mean_energy = mean.square().mean(dim=-1)
    total_energy = mean_energy + residual_energy
    safe_total = torch.where(
        total_energy > 0, total_energy, torch.ones_like(total_energy)
    )
    reliability = mean_energy / safe_total
    return mean, reliability


def dominant_spectral_signature(
    futures: torch.Tensor, dimensions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent a future by its training-set dominant complex frequencies."""
    if futures.ndim != 2 or dimensions <= 0 or dimensions % 2:
        raise ValueError("expected [sample, time] futures and an even dimension")
    spectrum = torch.fft.rfft(futures, dim=-1, norm="ortho")
    frequency_count = dimensions // 2
    if spectrum.shape[-1] < frequency_count:
        raise ValueError("future is too short for the requested signature")
    energy = spectrum.abs().square().mean(dim=0)
    frequencies = torch.topk(energy, frequency_count).indices.sort().values
    selected = spectrum[:, frequencies]
    signature = torch.cat([selected.real, selected.imag], dim=-1)
    return signature, frequencies


def cross_variable_event_signatures(
    histories: torch.Tensor,
    factors: tuple[int, ...],
    segments: int = 8,
) -> torch.Tensor:
    """Return target-excluded, permutation-invariant multiscale histories.

    ``histories`` is ``[item, channel, time]``.  Every output descriptor owns
    one target channel and summarizes only its synchronized peers.  It contains
    observed history statistics only, never peer futures or channel identities.
    """
    if histories.ndim != 3 or histories.shape[1] < 2:
        raise ValueError("cross-variable event signatures require at least two channels")
    if segments <= 0:
        raise ValueError("segments must be positive")
    items, channels, length = histories.shape
    if any(length % int(factor) for factor in factors):
        raise ValueError("every multiscale factor must divide history length")
    centered = histories - histories.mean(dim=-1, keepdim=True)
    normalized = centered / centered.square().mean(
        dim=-1, keepdim=True
    ).sqrt().clamp_min(1e-6)
    descriptors = []
    for factor in factors:
        factor = int(factor)
        view = normalized.reshape(items, channels, length // factor, factor).mean(dim=-1)
        view = F.adaptive_avg_pool1d(
            view.reshape(items * channels, 1, view.shape[-1]), segments
        ).reshape(items, channels, segments)
        peer_mean = (view.sum(dim=1, keepdim=True) - view) / float(channels - 1)
        peer_square_mean = (
            view.square().sum(dim=1, keepdim=True) - view.square()
        ) / float(channels - 1)
        peer_std = (peer_square_mean - peer_mean.square()).clamp_min(0.0).sqrt()
        if segments == 1:
            peer_delta = torch.zeros_like(peer_mean)
        else:
            delta = view.diff(dim=-1)
            delta_square_mean = (
                delta.square().sum(dim=1, keepdim=True) - delta.square()
            ) / float(channels - 1)
            peer_delta = F.pad(delta_square_mean.sqrt(), (1, 0), mode="replicate")
        descriptors.extend([peer_mean, peer_std, peer_delta])
    return F.normalize(torch.cat(descriptors, dim=-1), dim=-1)


def cross_variable_event_coupled_weights(
    scale_scores: torch.Tensor,
    query_signatures: torch.Tensor,
    candidate_signatures: torch.Tensor,
    iterations: int = 3,
) -> torch.Tensor:
    """Couple independent scale Top-K marginals through system-event agreement."""
    if scale_scores.ndim != 3 or candidate_signatures.ndim != 4:
        raise ValueError("invalid cross-variable event coupling shapes")
    scales, queries, _ = scale_scores.shape
    if scales < 2 or query_signatures.shape[0] != queries:
        raise ValueError("event coupling requires aligned multiscale query signatures")
    if candidate_signatures.shape[:3] != scale_scores.shape:
        raise ValueError("candidate signatures must describe every scale candidate")
    query_signatures = F.normalize(query_signatures, dim=-1)
    candidate_signatures = F.normalize(candidate_signatures, dim=-1)
    base_log_weights = F.log_softmax(scale_scores, dim=-1)
    weights = base_log_weights.exp()
    query_agreement = torch.einsum(
        "qf,sqkf->sqk", query_signatures, candidate_signatures
    )
    pair_agreement = torch.einsum(
        "sqkf,tqlf->stqkl", candidate_signatures, candidate_signatures
    )
    for _ in range(int(iterations)):
        compatibility = query_agreement.clone()
        for scale in range(scales):
            support = 0.0
            for other in range(scales):
                if other != scale:
                    support = support + torch.einsum(
                        "qkl,ql->qk", pair_agreement[scale, other], weights[other]
                    )
            compatibility[scale] = compatibility[scale] + support / float(scales - 1)
        weights = torch.softmax(base_log_weights + compatibility, dim=-1)
    return weights


def cross_variable_event_clock_weights(
    scale_scores: torch.Tensor,
    scale_starts: torch.Tensor,
    total_window: int,
    shuffled_peers: bool = False,
) -> torch.Tensor:
    """Reweight scale candidates by cross-variable retrieval-time consensus.

    Each peer channel contributes only the score-weighted temporal center and
    spread of *its own* target-local candidate set.  A target candidate is
    preferred when its historical time is plausible under the other channels'
    retrieval clocks at the other temporal scales.  No peer history/key/future
    is read by this operator.
    """
    if scale_scores.ndim != 4 or scale_starts.shape != scale_scores.shape:
        raise ValueError("expected scores and starts [scale, batch, channel, neighbor]")
    scales, batches, channels, _ = scale_scores.shape
    if scales < 2 or channels < 2 or total_window <= 0:
        raise ValueError("event-clock coupling requires multiple scales/channels")
    base = torch.softmax(scale_scores, dim=-1)
    centers = torch.sum(base * scale_starts, dim=-1)
    spreads = torch.sum(base * (scale_starts - centers.unsqueeze(-1)).square(), dim=-1)
    peer_centers = (centers.sum(dim=2, keepdim=True) - centers) / float(channels - 1)
    peer_spreads = (spreads.sum(dim=2, keepdim=True) - spreads) / float(channels - 1)
    if shuffled_peers and batches > 1:
        peer_centers = torch.roll(peer_centers, shifts=1, dims=1)
        peer_spreads = torch.roll(peer_spreads, shifts=1, dims=1)
    clock_evidence = torch.zeros_like(scale_scores)
    for scale in range(scales):
        for other in range(scales):
            if other == scale:
                continue
            variance = peer_spreads[other].unsqueeze(-1) + float(total_window**2)
            clock_evidence[scale] = clock_evidence[scale] - (
                scale_starts[scale] - peer_centers[other].unsqueeze(-1)
            ).square() / variance.clamp_min(1.0)
    return torch.softmax(
        torch.log_softmax(scale_scores, dim=-1) + clock_evidence / float(scales - 1),
        dim=-1,
    )


def cross_variable_pareto_mask(
    target_scores: torch.Tensor,
    query_signatures: torch.Tensor,
    candidate_signatures: torch.Tensor,
) -> torch.Tensor:
    """Keep candidates non-dominated in target fit and peer-context fit."""
    if target_scores.ndim != 3 or candidate_signatures.ndim != 4:
        raise ValueError("invalid Pareto screening shapes")
    if candidate_signatures.shape[:3] != target_scores.shape:
        raise ValueError("candidate signatures must align with target scores")
    if query_signatures.shape != (
        target_scores.shape[1],
        candidate_signatures.shape[-1],
    ):
        raise ValueError("query signatures must align with candidate signatures")
    peer_scores = -torch.mean(
        (candidate_signatures - query_signatures[None, :, None, :]).square(),
        dim=-1,
    )
    target_i = target_scores.unsqueeze(-1)
    target_j = target_scores.unsqueeze(-2)
    peer_i = peer_scores.unsqueeze(-1)
    peer_j = peer_scores.unsqueeze(-2)
    dominates = (target_i >= target_j) & (peer_i >= peer_j)
    dominates &= (target_i > target_j) | (peer_i > peer_j)
    keep = ~dominates.any(dim=-2)
    if not bool(keep.any(dim=-1).all()):
        raise RuntimeError("Pareto screening removed every candidate")
    return keep


class SharedScaleEvidenceRouter(nn.Module):
    """Route additive scale components with one query-dependent shared rule."""

    def __init__(self, topk: int):
        super().__init__()
        self.projection = nn.Linear(int(topk), 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        scale_scores: torch.Tensor,
        *,
        batch_size: int,
        channels: int,
        shuffle_queries: bool = False,
    ) -> torch.Tensor:
        if scale_scores.ndim != 3:
            raise ValueError("expected scale scores [scale, query, neighbor]")
        centered = scale_scores - scale_scores.mean(dim=-1, keepdim=True)
        standardized = centered / centered.square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).sqrt()
        if shuffle_queries and batch_size > 1:
            standardized = standardized.reshape(
                scale_scores.shape[0], batch_size, channels, scale_scores.shape[2]
            ).roll(shifts=1, dims=1).reshape_as(standardized)
        return 2.0 * torch.sigmoid(self.projection(standardized).squeeze(-1))


class SharedResidualLaplacianRouter(nn.Module):
    """Predict a bounded residual coefficient from Exact and scale evidence."""

    def __init__(self, topk: int):
        super().__init__()
        self.projection = nn.Linear(2 * int(topk), 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _standardize(scores: torch.Tensor) -> torch.Tensor:
        centered = scores - scores.mean(dim=-1, keepdim=True)
        return centered / centered.square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).sqrt()

    def forward(
        self,
        exact_scores: torch.Tensor,
        scale_scores: torch.Tensor,
        *,
        batch_size: int,
        channels: int,
        shuffle_queries: bool = False,
    ) -> torch.Tensor:
        if exact_scores.ndim != 2 or scale_scores.ndim != 3:
            raise ValueError("expected Exact [query, neighbor] and scale evidence")
        if scale_scores.shape[1:] != exact_scores.shape:
            raise ValueError("Exact and scale evidence shapes must align")
        exact = self._standardize(exact_scores).unsqueeze(0).expand(
            scale_scores.shape[0], -1, -1
        )
        scale = self._standardize(scale_scores)
        evidence = torch.cat([exact, scale], dim=-1)
        if shuffle_queries and batch_size > 1:
            evidence = evidence.reshape(
                scale_scores.shape[0], batch_size, channels, evidence.shape[-1]
            ).roll(shifts=1, dims=1).reshape_as(evidence)
        return torch.tanh(self.projection(evidence).squeeze(-1))


def scale_resolved_blend(
    retrieval_pred: torch.Tensor,
    base_pred: torch.Tensor,
    scale_weights: torch.Tensor,
    factors: tuple[int, ...],
) -> torch.Tensor:
    """Blend matching retrieval and Backbone components at each scale."""
    if retrieval_pred.shape != base_pred.shape or retrieval_pred.ndim != 3:
        raise ValueError("predictions must share shape [batch, time, channel]")
    batch_size, pred_len, channels = retrieval_pred.shape
    retrieval_ci = retrieval_pred.permute(0, 2, 1).reshape(-1, pred_len)
    base_ci = base_pred.permute(0, 2, 1).reshape(-1, pred_len)
    retrieval_components = temporal_laplacian(retrieval_ci, factors).reshape(
        len(factors), batch_size, channels, pred_len
    ).permute(0, 1, 3, 2)
    base_components = temporal_laplacian(base_ci, factors).reshape(
        len(factors), batch_size, channels, pred_len
    ).permute(0, 1, 3, 2)
    if scale_weights.shape != (len(factors), batch_size, channels, 2):
        raise ValueError("scale weights do not match predictions and factors")
    branches = torch.stack([retrieval_components, base_components], dim=-1)
    weights = scale_weights.unsqueeze(2)
    return torch.sum(branches * weights, dim=-1).sum(dim=0)


class RelationalTriFusion(nn.Module):
    """Learned simplex over original retrieval, relational retrieval and Backbone."""

    def __init__(self, topk: int, hidden_dim: int):
        super().__init__()
        self.topk = int(topk)
        self.network = nn.Sequential(
            nn.Linear(2 * self.topk, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
            nn.Softmax(dim=-1),
        )

    @property
    def context_mode(self) -> str:
        return "none"

    def forward(
        self,
        original_pred: torch.Tensor,
        relational_pred: torch.Tensor,
        base_pred: torch.Tensor,
        original_scores: torch.Tensor,
        relational_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, channels = base_pred.shape
        features = torch.cat((original_scores, relational_scores), dim=-1)
        weights = self.network(features.reshape(batch_size * channels, -1))
        weights = weights.reshape(batch_size, channels, 3)
        branches = torch.stack((original_pred, relational_pred, base_pred), dim=-1)
        return torch.sum(branches * weights.unsqueeze(1), dim=-1), weights

    def initialize_from_two_branch(self, fusion: nn.Module) -> None:
        """Embed the original retrieval/Backbone fusion as the initial solution."""
        with torch.no_grad():
            self.network[0].weight.zero_()
            self.network[0].weight[:, : self.topk].copy_(fusion.network[0].weight)
            self.network[0].bias.copy_(fusion.network[0].bias)
            self.network[2].weight.zero_()
            self.network[2].bias.zero_()
            self.network[2].weight[0].copy_(fusion.network[2].weight[0])
            self.network[2].bias[0].copy_(fusion.network[2].bias[0])
            self.network[2].weight[2].copy_(fusion.network[2].weight[1])
            self.network[2].bias[2].copy_(fusion.network[2].bias[1])
            self.network[2].bias[1] = -10.0


class DualPathFusion(nn.Module):
    """Independent Learned Fusion heads for multiscale and relational paths."""

    def __init__(self, topk: int, hidden_dim: int, pred_len: int):
        super().__init__()
        from ts_benchmark.baselines.coreg_public._engine.geometry_model import GeometryLearnedFusion

        self.multiscale = GeometryLearnedFusion(topk, hidden_dim)
        self.relational = GeometryLearnedFusion(topk, hidden_dim)
        self.horizon_logits = nn.Parameter(torch.zeros(int(pred_len)))

    @property
    def context_mode(self) -> str:
        return "none"


class MultiscaleRetrievalGeometryBranch(GeometryBranch):
    """Independent same-channel retrieval over additive temporal scales."""

    def __init__(self, configs):
        super().__init__(configs)
        self.moving_average_kernel = int(getattr(configs, "coreg_moving_average_kernel", 25))
        self.reliability_enabled = bool(getattr(configs, "coreg_reliability_enabled", False))
        self.reliability_strength = float(getattr(configs, "coreg_reliability_strength", 1.0))
        if self.reliability_strength < 0:
            raise ValueError("reliability strength must be nonnegative")
        if self.moving_average_kernel < 1 or self.moving_average_kernel % 2 == 0:
            raise ValueError("moving average kernel must be positive and odd")
        factors = (2, 1)
        if isinstance(factors, str):
            factors = tuple(int(value) for value in factors.split(","))
        self.scale_factors = tuple(
            sorted({int(value) for value in factors}, reverse=True)
        )
        if not self.scale_factors or self.scale_factors[-1] != 1:
            raise ValueError("Geometry_multiscale_factors must contain 1")
        for factor in self.scale_factors:
            if not self.moving_average_kernel and (self.seq_len % factor or self.pred_len % factor):
                raise ValueError(
                    f"scale factor {factor} must divide seq_len and pred_len"
                )
        self.multiscale_mode = str(
            getattr(configs, "Geometry_multiscale_mode", "laplacian")
        )
        if (self.reliability_enabled or self.moving_average_kernel) and not self.multiscale_mode.startswith("cross_variable_dual_path"):
            raise ValueError("reliability and moving-average options require the dual-path mode")
        if self.multiscale_mode not in {
            "laplacian",
            "laplacian_shuffled",
            "parallel_mean",
            "hierarchical",
            "hierarchical_shuffled",
            "coherent_joint",
            "coherent_joint_shuffled",
            "posterior_shrink",
            "posterior_shrink_shuffled",
            "predictive_spectral",
            "predictive_spectral_shuffled",
            "anchored_laplacian",
            "anchored_laplacian_shuffled",
            "energy_aligned",
            "energy_aligned_shuffled",
            "evidence_routed",
            "evidence_routed_shuffled",
            "scale_resolved_fusion",
            "scale_resolved_fusion_shuffled",
            "exact_candidate_consensus",
            "exact_candidate_consensus_shuffled",
            "moment_anchored_consensus",
            "moment_anchored_consensus_shuffled",
            "learned_scale_consensus",
            "learned_scale_consensus_shuffled",
            "predictive_fidelity_consensus",
            "predictive_fidelity_consensus_shuffled",
            "negative_consensus_veto",
            "negative_consensus_veto_shuffled",
            "standardized_negative_veto",
            "standardized_negative_veto_shuffled",
            "boundary_aligned",
            "boundary_aligned_shuffled",
            "residual_laplacian_adapter",
            "residual_laplacian_adapter_shuffled",
            "residual_laplacian_router",
            "residual_laplacian_router_shuffled",
            "cross_variable_event_coupling",
            "cross_variable_event_coupling_shuffled",
            "cross_variable_event_clock",
            "cross_variable_event_clock_shuffled",
            "cross_variable_pareto_screen",
            "cross_variable_pareto_screen_shuffled",
            "cross_variable_dual_path",
            "cross_variable_dual_path_shuffled",
            "cross_variable_dual_path_reproduced",
            "cross_variable_dual_path_reproduced_shuffled",
            "scale_predictive_encoder",
            "scale_predictive_encoder_shuffled",
            "relational_multiscale",
            "relational_multiscale_shuffled",
            "dual_relational_multiscale",
            "dual_relational_multiscale_shuffled",
            "tri_relational_multiscale",
            "tri_relational_multiscale_shuffled",
            "tri_relational_multiscale_nested",
            "tri_relational_multiscale_nested_shuffled",
        }:
            raise ValueError("invalid Geometry_multiscale_mode")
        self.hierarchical_pool = int(
            getattr(configs, "Geometry_multiscale_pool", max(96, self.topk))
        )
        if not self.topk <= self.hierarchical_pool <= self.gmb_size:
            raise ValueError("Geometry_multiscale_pool must be between topk and GMB size")
        self.multiscale_stream_rows = int(
            getattr(configs, "Geometry_multiscale_stream_rows", 2048)
        )
        if self.multiscale_stream_rows <= 0:
            raise ValueError("Geometry_multiscale_stream_rows must be positive")
        self.register_buffer(
            "multiscale_memory_keys",
            torch.empty(0, 0, self.embedding_dim),
            persistent=False,
        )
        self.register_buffer(
            "predictive_scale_maps",
            torch.empty(0, 0, 0),
            persistent=False,
        )
        self.register_buffer(
            "multiscale_memory_log_energy",
            torch.empty(0, 0),
            persistent=False,
        )
        self.register_buffer(
            "cross_variable_event_memory_signatures",
            torch.empty(0, 0),
            persistent=False,
        )
        self.register_buffer(
            "original_multiscale_memory_keys",
            torch.empty(0, 0, self.embedding_dim),
            persistent=False,
        )
        self._scale_key_build_seconds = 0.0
        self._scale_candidate_overlap_sum = 0.0
        self._scale_candidate_overlap_count = 0
        self._posterior_reliability_sum = 0.0
        self._posterior_reliability_count = 0
        self.scale_evidence_router = SharedScaleEvidenceRouter(self.topk)
        self._scale_gate_sum = 0.0
        self._scale_gate_square_sum = 0.0
        self._scale_gate_count = 0
        self._last_scale_scores: torch.Tensor | None = None
        self._candidate_consensus_weight_delta_sum = 0.0
        self._candidate_consensus_weight_delta_count = 0
        self.scale_evidence_logits = nn.Parameter(
            torch.zeros(len(self.scale_factors) + 1)
        )
        self.residual_laplacian_logits = nn.Parameter(
            torch.zeros(len(self.scale_factors))
        )
        self.residual_laplacian_router = SharedResidualLaplacianRouter(self.topk)
        self._residual_coefficient_sum = 0.0
        self._residual_coefficient_square_sum = 0.0
        self._residual_coefficient_count = 0
        self.register_buffer(
            "scale_predictive_fidelity_weights",
            torch.empty(0),
            persistent=False,
        )
        
        
        
        
        
        scale_specific = self.multiscale_mode.startswith("scale_predictive_encoder")
        relational_multiscale = "relational_multiscale" in self.multiscale_mode
        self.scale_predictive_encoders = nn.ModuleList(
            [copy.deepcopy(self.encoder) for _ in self.scale_factors]
            if scale_specific
            else []
        )
        self.relational_multiscale_encoder = (
            copy.deepcopy(self.encoder) if relational_multiscale else None
        )
        self._relational_multiscale_config = (
            copy.deepcopy(configs) if relational_multiscale else None
        )
        self._relational_teacher_cache_used = False
        self.relational_exact_branch = None
        if (
            self.multiscale_mode.startswith("cross_variable_dual_path")
            and not self.multiscale_mode.startswith(
                "cross_variable_dual_path_reproduced"
            )
        ):
            from ts_benchmark.baselines.coreg_public._engine.role_geometry import (
                RelationalPredictiveContrastGeometryBranch,
            )

            relational_config = copy.deepcopy(configs)
            relational_control = (
                "zero_peer"
                if self.multiscale_mode.endswith("_zero")
                else (
                    "time_shuffled"
                    if self.multiscale_mode.endswith("_shuffled")
                    else "aligned"
                )
            )
            setattr(
                relational_config,
                "memory_relational_contrast_control",
                relational_control,
            )
            setattr(
                relational_config,
                "memory_cache_root",
                str(
                    Path(self.cache_root)
                    / f"dual_path_relational_{relational_control}"
                ),
            )
            self.relational_exact_branch = RelationalPredictiveContrastGeometryBranch(
                relational_config
            )
        self._scale_encoder_cache_used = False
        self._scale_encoder_losses: list[float] = []
        self._retrieval_trace_dir = getattr(configs, "coreg_retrieval_trace_dir", "")
        self._retrieval_trace_batch = 0

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
        if scores is not None:
            payload["neighbor_scores"] = scores[:rows].detach().to(dtype=torch.float32).cpu().numpy()
        if query_starts is not None:
            payload["query_starts"] = query_starts.unsqueeze(1).expand(-1, self.channels).reshape(-1)[:rows].detach().cpu().numpy()
        np.savez_compressed(trace_dir / f"retrieval_trace_{self._retrieval_trace_batch:06d}.npz", **payload)
        self._retrieval_trace_batch += 1

    def _query_views(self, x_norm: torch.Tensor) -> torch.Tensor:
        return moving_average_components(x_norm, self.moving_average_kernel)

    def _future_views(self, future: torch.Tensor) -> torch.Tensor:
        return moving_average_components(future, self.moving_average_kernel)

    def _future_view_at(self, future: torch.Tensor, scale: int) -> torch.Tensor:
        return moving_average_components(future, self.moving_average_kernel)[scale]

    @staticmethod
    def _roll_keys_within_channel(keys: torch.Tensor) -> torch.Tensor:
        if keys.shape[2] <= 1:
            return keys
        rolled = keys.clone()
        
        for scale in range(0, keys.shape[0] - 1):
            shift = max(1, keys.shape[2] // (scale + 2))
            rolled[scale] = torch.roll(keys[scale], shifts=shift, dims=1)
        return rolled

    def _encode_multiscale_memory(
        self, values: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        values_tensor = torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32)
        )
        pair_ids = self.memory_starts.cpu() * self.channels + self.memory_channels.cpu()
        outputs = [[] for _ in self.scale_factors]
        energy_outputs = [[] for _ in self.scale_factors]
        scale_specific = self.multiscale_mode.startswith("scale_predictive_encoder")
        relational = "relational_multiscale" in self.multiscale_mode
        dual_relational = self.multiscale_mode.startswith(
            "dual_relational_multiscale"
        )
        tri_relational = self.multiscale_mode.startswith(
            "tri_relational_multiscale"
        )
        original_outputs = [[] for _ in self.scale_factors]
        active_encoders = (
            list(self.scale_predictive_encoders)
            if scale_specific
            else (
                [self.relational_multiscale_encoder for _ in self.scale_factors]
                if relational
                else [self.encoder for _ in self.scale_factors]
            )
        )
        
        
        
        unique_active_encoders = tuple(dict.fromkeys(active_encoders))
        for encoder in unique_active_encoders:
            encoder.to(device)
            encoder.eval()
        if dual_relational or tri_relational:
            self.encoder.to(device)
            self.encoder.eval()
        with torch.no_grad():
            for start in range(0, pair_ids.numel(), self.encode_batch_size):
                current = pair_ids[start : start + self.encode_batch_size]
                history = self._gather_pairs(
                    values_tensor, current, self.seq_len, 0
                ).to(device)
                _, _, history_norm = _normalize_series(history)
                views = self._query_views(history_norm)
                for scale, view in enumerate(views):
                    energy_outputs[scale].append(log_rms_energy(view).cpu())
                    _, _, view_norm = _normalize_series(view)
                    active_key = F.normalize(active_encoders[scale](view_norm), dim=-1)
                    if dual_relational or tri_relational:
                        base_key = F.normalize(self.encoder(view_norm), dim=-1)
                        if tri_relational:
                            original_outputs[scale].append(base_key.cpu())
                    if dual_relational:
                        active_key = F.normalize(
                            torch.cat((base_key, active_key), dim=-1), dim=-1
                        )
                    outputs[scale].append(active_key.cpu())
        for encoder in unique_active_encoders:
            encoder.cpu()
        if dual_relational or tri_relational:
            self.encoder.cpu()
        self.multiscale_memory_log_energy = torch.stack(
            [torch.cat(scale_outputs, dim=0) for scale_outputs in energy_outputs]
        )
        if tri_relational:
            self.original_multiscale_memory_keys = torch.stack(
                [torch.cat(scale_outputs, dim=0) for scale_outputs in original_outputs]
            )
        return torch.stack(
            [torch.cat(scale_outputs, dim=0) for scale_outputs in outputs]
        )

    def _prepare_scale_predictive_encoders(
        self, values: np.ndarray, device: torch.device
    ) -> None:
        """Fit one predictive metric per Laplacian band on the selected GMB.

        The original encoder is trained against the complete future.  Reusing it
        for every Laplacian band is therefore a representation mismatch.  These
        encoders start from that exact checkpoint and refine only the metric used
        by one band.  The shuffled control preserves capacity and optimization
        while breaking the history/future pairing.
        """
        cache_path = Path(self.cache_path)
        suffix = "shuffled" if self.multiscale_mode.endswith("_shuffled") else "aligned"
        scale_cache = cache_path.with_name(
            f"{cache_path.stem}_scale_predictive_v1_{suffix}.pt"
        )
        metadata = {
            "version": 1,
            "mode": self.multiscale_mode,
            "factors": self.scale_factors,
            "epochs": self.pcl_epochs,
            "temperature": self.pcl_temperature,
            "lr": self.pcl_lr,
            "seed": self.seed,
            "base_cache": cache_path.name,
        }
        if scale_cache.exists():
            payload = torch.load(scale_cache, map_location="cpu")
            if payload.get("metadata") != metadata:
                raise ValueError(f"Scale predictive cache metadata mismatch: {scale_cache}")
            for encoder, state in zip(
                self.scale_predictive_encoders, payload["encoder_states"]
            ):
                encoder.load_state_dict(state)
            self._scale_encoder_losses = list(payload.get("losses", []))
            self._scale_encoder_cache_used = True
        else:
            for encoder in self.scale_predictive_encoders:
                encoder.load_state_dict(self.encoder.state_dict())

            values_tensor = torch.from_numpy(
                np.ascontiguousarray(values, dtype=np.float32)
            )
            pair_ids = (
                self.memory_starts.cpu() * self.channels
                + self.memory_channels.cpu()
            )
            histories = self._gather_pairs(
                values_tensor, pair_ids, self.seq_len, 0
            )
            futures = self.memory_future.float().cpu()
            _, _, histories = _normalize_series(histories)
            _, _, futures = _normalize_series(futures)
            history_views = self._query_views(histories)
            future_views = self._future_views(futures)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            self._scale_encoder_losses = []

            for scale, encoder in enumerate(self.scale_predictive_encoders):
                encoder.to(device)
                optimizer = torch.optim.Adam(encoder.parameters(), lr=self.pcl_lr)
                epoch_losses = []
                for _ in range(self.pcl_epochs):
                    encoder.train()
                    order = torch.randperm(pair_ids.numel(), generator=generator)
                    losses = []
                    for begin in range(0, pair_ids.numel(), self.pcl_batch_size):
                        selected = order[begin : begin + self.pcl_batch_size]
                        if selected.numel() < 2:
                            continue
                        x_view = history_views[scale, selected].to(device)
                        y_view = future_views[scale, selected].to(device)
                        _, _, x_view = _normalize_series(x_view)
                        _, _, y_view = _normalize_series(y_view)
                        starts = self.memory_starts[selected].to(device)
                        overlap = (
                            torch.abs(starts.unsqueeze(1) - starts.unsqueeze(0))
                            < self.total_window
                        )
                        legal = (~overlap).any(dim=1)
                        if not bool(legal.any()):
                            continue
                        if self.multiscale_mode.endswith("_shuffled"):
                            shift = max(1, selected.numel() // 2)
                            y_view = torch.roll(y_view, shifts=shift, dims=0)
                        distance = torch.cdist(y_view, y_view, p=2).masked_fill(
                            overlap, float("inf")
                        )
                        positive = torch.argmin(distance, dim=1)
                        features = F.normalize(encoder(x_view), dim=-1)
                        logits = features.matmul(features.transpose(0, 1))
                        logits = (logits / self.pcl_temperature).masked_fill(
                            overlap, float("-inf")
                        )
                        loss = F.cross_entropy(logits[legal], positive[legal])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                    if not losses:
                        raise RuntimeError("Scale predictive encoder found no legal pairs")
                    epoch_losses.append(float(np.mean(losses)))
                encoder.cpu()
                self._scale_encoder_losses.append(epoch_losses[-1])

            scale_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": metadata,
                    "encoder_states": [
                        copy.deepcopy(encoder.state_dict())
                        for encoder in self.scale_predictive_encoders
                    ],
                    "losses": self._scale_encoder_losses,
                },
                scale_cache,
            )
            self._scale_encoder_cache_used = False
        for encoder in self.scale_predictive_encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            encoder.eval()

    def _prepare_relational_multiscale_encoder(
        self, values: np.ndarray, device: torch.device
    ) -> None:
        """Train-time multivariate supervision with target-local inference."""
        from ts_benchmark.baselines.coreg_public._engine.role_geometry import (
            RelationalPredictiveContrastGeometryBranch,
        )

        control = (
            "time_shuffled"
            if self.multiscale_mode.endswith("_shuffled")
            else "aligned"
        )
        if (
            self.relational_multiscale_encoder is None
            or self._relational_multiscale_config is None
        ):
            raise RuntimeError("relational multiscale encoder was not configured")
        config = copy.deepcopy(self._relational_multiscale_config)
        setattr(config, "memory_relational_contrast_control", control)
        setattr(
            config,
            "memory_cache_root",
            str(Path(self.cache_root) / f"relational_multiscale_teacher_{control}"),
        )
        teacher = RelationalPredictiveContrastGeometryBranch(config)
        teacher.prepare_memory(values, device)
        self.relational_multiscale_encoder.load_state_dict(teacher.encoder.state_dict())
        self._relational_teacher_cache_used = bool(teacher.cache_used)
        for parameter in self.relational_multiscale_encoder.parameters():
            parameter.requires_grad = False
        self.relational_multiscale_encoder.eval()

    def _encode_cross_variable_event_memory(
        self, values: np.ndarray
    ) -> torch.Tensor:
        """Encode system contexts once per selected historical start on CPU."""
        if self.channels < 2:
            raise ValueError("cross-variable event coupling requires multivariate data")
        values_tensor = torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32)
        )
        starts, inverse = torch.unique(
            self.memory_starts.cpu(), sorted=True, return_inverse=True
        )
        offsets = torch.arange(self.seq_len, dtype=torch.long)
        outputs = []
        
        
        with torch.no_grad():
            for begin in range(0, starts.numel(), 16):
                current = starts[begin : begin + 16]
                histories = values_tensor[current.unsqueeze(1) + offsets]
                outputs.append(
                    cross_variable_event_signatures(
                        histories.permute(0, 2, 1).contiguous(),
                        self.scale_factors,
                    )
                )
        by_start = torch.cat(outputs, dim=0)
        memory_signatures = by_start[inverse, self.memory_channels.cpu()]
        if not bool(torch.isfinite(memory_signatures).all()):
            raise RuntimeError("cross-variable event memory signatures are non-finite")
        return memory_signatures.contiguous()

    def prepare_memory(self, values: np.ndarray, device: torch.device) -> None:
        import time

        super().prepare_memory(values, device)
        started = time.perf_counter()
        if self.relational_exact_branch is not None:
            self.relational_exact_branch.prepare_memory(values, device)
        if self.multiscale_mode.startswith("scale_predictive_encoder"):
            self._prepare_scale_predictive_encoders(values, device)
        if "relational_multiscale" in self.multiscale_mode:
            self._prepare_relational_multiscale_encoder(values, device)
        if self.multiscale_mode.startswith("predictive_spectral"):
            self.multiscale_memory_keys, self.predictive_scale_maps = (
                self._fit_predictive_spectral_keys()
            )
        else:
            self.multiscale_memory_keys = self._encode_multiscale_memory(values, device)
        if self.multiscale_mode.startswith(
            ("cross_variable_event_coupling", "cross_variable_pareto_screen")
        ):
            self.cross_variable_event_memory_signatures = (
                self._encode_cross_variable_event_memory(values)
            )
        if self.multiscale_memory_keys.shape[-1] == self.memory_keys.shape[-1]:
            self.scale_predictive_fidelity_weights = predictive_fidelity_weights(
                self.memory_keys.float().cpu(),
                self.multiscale_memory_keys.float().cpu(),
            )
        else:
            self.scale_predictive_fidelity_weights = torch.full(
                (len(self.scale_factors),), 1.0 / len(self.scale_factors)
            )
        self._scale_key_build_seconds = time.perf_counter() - started

    def _fit_predictive_spectral_keys(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_keys = self.memory_keys.float().cpu()
        design = torch.cat(
            [history_keys, torch.ones(history_keys.shape[0], 1)], dim=1
        )
        future_views = self._future_views(self.memory_future.float().cpu())
        memory_keys = []
        maps = []
        frequency_sets = []
        for future_view in future_views:
            signature, frequencies = dominant_spectral_signature(
                future_view, self.embedding_dim
            )
            solution = torch.linalg.lstsq(
                design.double(), signature.double()
            ).solution.float()
            memory_keys.append(F.normalize(signature, dim=-1))
            maps.append(solution)
            frequency_sets.append(tuple(int(value) for value in frequencies))
        self._predictive_spectral_frequencies = tuple(frequency_sets)
        return torch.stack(memory_keys), torch.stack(maps)

    def _mask_overlap(
        self,
        similarity: torch.Tensor,
        index: torch.Tensor | None,
        batch_size: int,
        mode: str,
        memory_indices: torch.Tensor,
    ) -> torch.Tensor:
        if mode != "train":
            return similarity
        if index is None:
            raise ValueError("Training Geometry retrieval requires window indices.")
        query_starts = index.to(similarity.device).reshape(batch_size, 1, 1)
        memory_starts = self.memory_starts[memory_indices].unsqueeze(0)
        overlap = torch.abs(query_starts - memory_starts) < self.total_window
        return similarity.masked_fill(
            overlap.reshape(batch_size * self.channels, -1), float("-inf")
        )

    def _similarity(
        self,
        query_keys: torch.Tensor,
        memory_keys: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        similarity = torch.bmm(
            query_keys.reshape(batch_size, self.channels, -1).permute(1, 0, 2),
            memory_keys.transpose(1, 2),
        )
        return similarity.permute(1, 0, 2).reshape(
            batch_size * self.channels, -1
        )

    def _retrieve_one_scale(
        self,
        query_view: torch.Tensor,
        memory_keys: torch.Tensor,
        future_scale: int | None,
        index: torch.Tensor | None,
        mode: str,
        batch_size: int,
        memory_indices: torch.Tensor,
        query_log_energy: torch.Tensor | None = None,
        memory_log_energy: torch.Tensor | None = None,
        return_candidate_futures: bool = False,
        compute_reliability: bool = True,
    ) -> tuple:
        factor = self.scale_factors[future_scale] if future_scale is not None else 1
        prefix = f"timegeo.scale{factor}"
        with stage(f"{prefix}.query_encode"):
            _, _, query_for_key = _normalize_series(query_view)
            encoder = (
                self.scale_predictive_encoders[future_scale]
                if future_scale is not None
                and self.multiscale_mode.startswith("scale_predictive_encoder")
                else (
                    self.relational_multiscale_encoder
                    if "relational_multiscale" in self.multiscale_mode
                    else self.encoder
                )
            )
            query_keys = self._truth_keys(query_for_key)
            if encoder is not self.encoder and self.retrieval_truth_control not in {
                "raw_history_knn", "untrained_encoder"
            }:
                query_keys = F.normalize(encoder(query_for_key), dim=-1)
            if self.multiscale_mode.startswith("dual_relational_multiscale"):
                base_query_keys = F.normalize(self.encoder(query_for_key), dim=-1)
                query_keys = F.normalize(
                    torch.cat((base_query_keys, query_keys), dim=-1), dim=-1
                )
        with stage(f"{prefix}.similarity"):
            similarity = self._similarity(query_keys, memory_keys, batch_size)
        with stage(f"{prefix}.mask"):
            similarity = self._control_similarity(similarity)
            similarity = self._mask_overlap(
                similarity, index, batch_size, mode, memory_indices
            )
            if query_log_energy is not None or memory_log_energy is not None:
                if query_log_energy is None or memory_log_energy is None:
                    raise ValueError("query and memory scale energy must be provided together")
                expanded_memory_energy = memory_log_energy.unsqueeze(0).expand(
                    batch_size, -1, -1
                ).reshape(similarity.shape)
                energy_similarity = -torch.abs(
                    query_log_energy.unsqueeze(1) - expanded_memory_energy
                )
                energy_similarity = self._mask_overlap(
                    energy_similarity, index, batch_size, mode, memory_indices
                )
                similarity = coherent_multiscale_similarity(
                    torch.stack([similarity, energy_similarity])
                )
        with stage(f"{prefix}.topk"):
            topk_similarity, local_indices = torch.topk(similarity, self.topk, dim=1)
            if not bool(torch.isfinite(topk_similarity).all()):
                raise RuntimeError("Multiscale Geometry has insufficient legal neighbors")
            query_channels = torch.arange(self.channels, device=query_view.device)
            query_channels = query_channels.unsqueeze(0).expand(
                batch_size, -1
            ).reshape(-1)
            topk_indices = memory_indices[
                query_channels.unsqueeze(1), local_indices
            ]
        first_layer = self.confidence_gate[0]
        query_count = query_for_key.shape[0]
        
        
        
        
        
        stream_rows = (
            query_count
            if return_candidate_futures
            else min(self.multiscale_stream_rows, query_count)
        )
        retrieved_parts = []
        score_parts = []
        reliability_parts = []
        candidate_parts = []
        for begin in range(0, query_count, stream_rows):
            end = min(begin + stream_rows, query_count)
            rows = slice(begin, end)
            with stage(f"{prefix}.gather"):
                full_future = self.memory_future[topk_indices[rows]]
                topk_future = (
                    full_future
                    if future_scale is None
                    else self._future_view_at(full_future, future_scale)
                )
            with stage(f"{prefix}.confidence"):
                query_hidden = F.linear(
                    query_for_key[rows],
                    first_layer.weight[:, : self.seq_len],
                    first_layer.bias,
                ).unsqueeze(1)
                future_hidden = F.linear(
                    topk_future,
                    first_layer.weight[:, self.seq_len :],
                    None,
                )
                confidence = self.confidence_gate[3](
                    self.confidence_gate[2](
                        self.confidence_gate[1](query_hidden + future_hidden)
                    )
                ).squeeze(-1)
                adjusted = self._adjust_retrieval_scores(
                    topk_similarity[rows], confidence
                )
                fusion_scores = adjusted
                if self.multiscale_mode.startswith("boundary_aligned"):
                    width = max(8, min(32, self.pred_len // 8))
                    query_tail = query_view[rows, -width:]
                    candidate_head = topk_future[..., :width]
                    query_tail = (
                        query_tail - query_tail.mean(dim=-1, keepdim=True)
                    ) / query_tail.std(
                        dim=-1, keepdim=True, unbiased=False
                    ).clamp_min(1e-6)
                    candidate_head = (
                        candidate_head - candidate_head.mean(dim=-1, keepdim=True)
                    ) / candidate_head.std(
                        dim=-1, keepdim=True, unbiased=False
                    ).clamp_min(1e-6)
                    continuity = torch.sum(
                        query_tail.unsqueeze(1) * candidate_head, dim=-1
                    ) / float(width)
                    if self.multiscale_mode.endswith("_shuffled"):
                        continuity = torch.roll(continuity, shifts=1, dims=1)
                    centered = continuity - continuity.mean(dim=1, keepdim=True)
                    adjusted = adjusted + centered / continuity.std(
                        dim=1, keepdim=True, unbiased=False
                    ).clamp_min(1e-6)
            with stage(f"{prefix}.aggregate"):
                weights = self._compute_neighbor_weights(adjusted)
                if self.reliability_enabled:
                    retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
                    reliability_parts.append(candidate_reliability(topk_future, weights, self.reliability_strength))
                elif compute_reliability:
                    retrieved, reliability = posterior_mean_reliability(
                        topk_future, weights
                    )
                    reliability_parts.append(reliability)
                else:
                    retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
                if self.retrieval_truth_control == "zero_retrieval":
                    retrieved = torch.zeros_like(retrieved)
                    
                    
                    
                    fusion_scores = torch.zeros_like(fusion_scores)
                retrieved_parts.append(retrieved)
                score_parts.append(fusion_scores)
                if return_candidate_futures:
                    candidate_parts.append(topk_future)
        with stage(f"{prefix}.aggregate"):
            retrieved = torch.cat(retrieved_parts, dim=0)
            fusion_scores = torch.cat(score_parts, dim=0)
            reliability = (
                torch.cat(reliability_parts, dim=0) if (compute_reliability or self.reliability_enabled) else None
            )
        if return_candidate_futures:
            return (
                retrieved,
                fusion_scores,
                topk_indices,
                reliability,
                torch.cat(candidate_parts, dim=0),
            )
        return retrieved, fusion_scores, topk_indices, reliability

    def _retrieve_predictive_scale(
        self,
        x_norm: torch.Tensor,
        base_query_keys: torch.Tensor,
        memory_keys: torch.Tensor,
        scale_map: torch.Tensor,
        future_scale: int,
        index: torch.Tensor | None,
        mode: str,
        batch_size: int,
        memory_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        design = torch.cat(
            [base_query_keys, torch.ones_like(base_query_keys[:, :1])], dim=1
        )
        query_keys = F.normalize(design @ scale_map, dim=-1)
        similarity = self._similarity(query_keys, memory_keys, batch_size)
        similarity = self._mask_overlap(
            similarity, index, batch_size, mode, memory_indices
        )
        topk_similarity, local_indices = torch.topk(similarity, self.topk, dim=1)
        if not bool(torch.isfinite(topk_similarity).all()):
            raise RuntimeError("Predictive spectral Geometry has insufficient neighbors")
        query_channels = torch.arange(self.channels, device=x_norm.device)
        query_channels = query_channels.unsqueeze(0).expand(
            batch_size, -1
        ).reshape(-1)
        topk_indices = memory_indices[
            query_channels.unsqueeze(1), local_indices
        ]
        topk_future = self._future_views(self.memory_future[topk_indices])[future_scale]

        first_layer = self.confidence_gate[0]
        query_hidden = F.linear(
            x_norm,
            first_layer.weight[:, : self.seq_len],
            first_layer.bias,
        ).unsqueeze(1)
        future_hidden = F.linear(
            topk_future,
            first_layer.weight[:, self.seq_len :],
            None,
        )
        confidence = self.confidence_gate[3](
            self.confidence_gate[2](
                self.confidence_gate[1](query_hidden + future_hidden)
            )
        ).squeeze(-1)
        adjusted = self._adjust_retrieval_scores(topk_similarity, confidence)
        weights = self._compute_neighbor_weights(adjusted)
        retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
        return retrieved, adjusted, topk_indices

    def _hierarchical_retrieval(
        self,
        x_norm: torch.Tensor,
        index: torch.Tensor | None,
        mode: str,
        batch_size: int,
        memory_indices: torch.Tensor,
        scale_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_view = block_mean_repeat(x_norm, self.scale_factors[0])
        _, _, coarse_norm = _normalize_series(coarse_view)
        coarse_query = F.normalize(self.encoder(coarse_norm), dim=-1)
        coarse_similarity = self._similarity(
            coarse_query, scale_keys[0], batch_size
        )
        coarse_similarity = self._mask_overlap(
            coarse_similarity, index, batch_size, mode, memory_indices
        )
        _, pool_local = torch.topk(
            coarse_similarity, self.hierarchical_pool, dim=1
        )

        fine_query = F.normalize(self.encoder(x_norm), dim=-1)
        query_channels = torch.arange(self.channels, device=x_norm.device)
        query_channels = query_channels.unsqueeze(0).expand(
            batch_size, -1
        ).reshape(-1)
        pool_global = memory_indices[query_channels.unsqueeze(1), pool_local]
        pool_keys = self.memory_keys[pool_global]
        fine_similarity = torch.sum(fine_query.unsqueeze(1) * pool_keys, dim=-1)
        topk_similarity, within_pool = torch.topk(
            fine_similarity, self.topk, dim=1
        )
        topk_indices = torch.gather(pool_global, 1, within_pool)
        topk_future = self.memory_future[topk_indices]

        first_layer = self.confidence_gate[0]
        query_hidden = F.linear(
            x_norm,
            first_layer.weight[:, : self.seq_len],
            first_layer.bias,
        ).unsqueeze(1)
        future_hidden = F.linear(
            topk_future,
            first_layer.weight[:, self.seq_len :],
            None,
        )
        confidence = self.confidence_gate[3](
            self.confidence_gate[2](
                self.confidence_gate[1](query_hidden + future_hidden)
            )
        ).squeeze(-1)
        adjusted = self._adjust_retrieval_scores(topk_similarity, confidence)
        weights = self._compute_neighbor_weights(adjusted)
        retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
        return retrieved, adjusted, topk_indices

    def _coherent_joint_retrieval(
        self,
        query_views: torch.Tensor,
        scale_keys: torch.Tensor,
        x_norm: torch.Tensor,
        index: torch.Tensor | None,
        mode: str,
        batch_size: int,
        memory_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_similarities = []
        for scale, query_view in enumerate(query_views):
            _, _, query_for_key = _normalize_series(query_view)
            query_key = F.normalize(self.encoder(query_for_key), dim=-1)
            similarity = self._similarity(
                query_key, scale_keys[scale], batch_size
            )
            scale_similarities.append(
                self._mask_overlap(
                    similarity, index, batch_size, mode, memory_indices
                )
            )
        joint_similarity = coherent_multiscale_similarity(
            torch.stack(scale_similarities)
        )
        topk_similarity, local_indices = torch.topk(
            joint_similarity, self.topk, dim=1
        )
        if not bool(torch.isfinite(topk_similarity).all()):
            raise RuntimeError("Coherent multiscale Geometry has insufficient neighbors")
        query_channels = torch.arange(self.channels, device=x_norm.device)
        query_channels = query_channels.unsqueeze(0).expand(
            batch_size, -1
        ).reshape(-1)
        topk_indices = memory_indices[
            query_channels.unsqueeze(1), local_indices
        ]

        first_layer = self.confidence_gate[0]
        query_hidden = F.linear(
            x_norm,
            first_layer.weight[:, : self.seq_len],
            first_layer.bias,
        )
        retrieved_parts = []
        score_parts = []
        for start in range(0, x_norm.shape[0], self.stream_rows):
            end = min(start + self.stream_rows, x_norm.shape[0])
            futures = self.memory_future[topk_indices[start:end]]
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
            weights = self._compute_neighbor_weights(adjusted)
            weights = self._refine_neighbor_weights(weights, futures)
            retrieved_parts.append(
                torch.sum(weights.unsqueeze(-1) * futures, dim=1)
            )
            fusion_scores = self._scores_for_fusion(
                topk_similarity[start:end], confidence, adjusted
            )
            score_parts.append(
                self._augment_fusion_scores(fusion_scores, futures, weights)
            )
        return (
            torch.cat(retrieved_parts, dim=0),
            torch.cat(score_parts, dim=0),
            topk_indices,
        )

    def _exact_candidate_consensus_retrieval(
        self,
        x_norm: torch.Tensor,
        index: torch.Tensor | None,
        mode: str,
        batch_size: int,
        memory_indices: torch.Tensor,
        scale_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep Exact candidates/futures and recalibrate only their weights."""
        def require_finite(name: str, value: torch.Tensor) -> None:
            finite = torch.isfinite(value)
            if not bool(finite.all()):
                invalid = int((~finite).sum().detach().cpu())
                raise RuntimeError(
                    f"Exact candidate consensus produced non-finite {name}: "
                    f"{invalid}/{value.numel()} values"
                )

        exact_query = F.normalize(self.encoder(x_norm), dim=-1)
        require_finite("exact query", exact_query)
        exact_similarity = self._similarity(
            exact_query, self.memory_keys[memory_indices], batch_size
        )
        exact_similarity = self._mask_overlap(
            exact_similarity, index, batch_size, mode, memory_indices
        )
        topk_similarity, local_indices = torch.topk(
            exact_similarity, self.topk, dim=1
        )
        if not bool(torch.isfinite(topk_similarity).all()):
            raise RuntimeError("Exact candidate consensus has insufficient neighbors")

        query_channels = torch.arange(self.channels, device=x_norm.device)
        query_channels = query_channels.unsqueeze(0).expand(
            batch_size, -1
        ).reshape(-1)
        topk_indices = memory_indices[
            query_channels.unsqueeze(1), local_indices
        ]
        topk_future = self.memory_future[topk_indices]

        scale_similarities = []
        for scale, query_view in enumerate(self._query_views(x_norm)):
            _, _, query_for_key = _normalize_series(query_view)
            query_key = F.normalize(self.encoder(query_for_key), dim=-1)
            candidate_keys = scale_keys[
                scale, query_channels.unsqueeze(1), local_indices
            ]
            require_finite(f"scale {scale} query key", query_key)
            require_finite(f"scale {scale} candidate keys", candidate_keys)
            scale_similarity = torch.sum(
                query_key.unsqueeze(1) * candidate_keys, dim=-1
            )
            require_finite(f"scale {scale} similarity", scale_similarity)
            scale_similarities.append(scale_similarity)

        first_layer = self.confidence_gate[0]
        query_hidden = F.linear(
            x_norm,
            first_layer.weight[:, : self.seq_len],
            first_layer.bias,
        ).unsqueeze(1)
        future_hidden = F.linear(
            topk_future,
            first_layer.weight[:, self.seq_len :],
            None,
        )
        confidence = self.confidence_gate[3](
            self.confidence_gate[2](
                self.confidence_gate[1](query_hidden + future_hidden)
            )
        ).squeeze(-1)
        require_finite("confidence", confidence)
        adjusted = self._adjust_retrieval_scores(topk_similarity, confidence)
        require_finite("adjusted Exact scores", adjusted)
        scale_score_tensor = torch.stack(scale_similarities)
        if self.multiscale_mode.startswith("moment_anchored_consensus"):
            consensus = moment_anchored_consensus_scores(
                adjusted, scale_score_tensor
            )
        elif self.multiscale_mode.startswith("learned_scale_consensus"):
            consensus = learned_candidate_consensus_scores(
                adjusted, scale_score_tensor, self.scale_evidence_logits
            )
        elif self.multiscale_mode.startswith("predictive_fidelity_consensus"):
            consensus = learned_candidate_consensus_scores(
                adjusted,
                scale_score_tensor,
                self.scale_predictive_fidelity_weights.to(adjusted.device)
                .clamp_min(1e-6)
                .log(),
            )
        elif self.multiscale_mode.startswith("negative_consensus_veto"):
            consensus = negative_consensus_veto_scores(
                adjusted, scale_score_tensor
            )
        elif self.multiscale_mode.startswith("standardized_negative_veto"):
            consensus = standardized_negative_veto_scores(
                adjusted, scale_score_tensor
            )
        else:
            consensus = exact_candidate_consensus_scores(
                adjusted, scale_score_tensor
            )
        require_finite("consensus scores", consensus)
        exact_weights = self._compute_neighbor_weights(adjusted)
        weights = self._compute_neighbor_weights(consensus)
        require_finite("consensus weights", weights)
        self._candidate_consensus_weight_delta_sum += float(
            torch.abs(weights.detach() - exact_weights.detach()).sum().cpu()
        )
        self._candidate_consensus_weight_delta_count += weights.numel()
        retrieved = torch.sum(weights.unsqueeze(-1) * topk_future, dim=1)
        require_finite("retrieved future", retrieved)
        return retrieved, adjusted, topk_indices, confidence.detach().mean()

    def forward(self, x, index, mode):
        if not self._prepared or self.multiscale_memory_keys.numel() == 0:
            raise RuntimeError("Multiscale Geometry memory is not prepared")
        batch_size, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError("Multiscale Geometry input shape mismatch")
        with stage("timegeo.normalize_views"):
            x_ci = x.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
            query_mean, query_std, x_norm = _normalize_series(x_ci)
            memory_indices = self._same_channel_memory_indices
            scale_keys = self.multiscale_memory_keys[:, memory_indices]
        if self.multiscale_mode in {
            "laplacian_shuffled",
            "hierarchical_shuffled",
            "coherent_joint_shuffled",
            "posterior_shrink_shuffled",
            "predictive_spectral_shuffled",
            "anchored_laplacian_shuffled",
            "energy_aligned_shuffled",
            "exact_candidate_consensus_shuffled",
            "moment_anchored_consensus_shuffled",
            "learned_scale_consensus_shuffled",
            "predictive_fidelity_consensus_shuffled",
            "negative_consensus_veto_shuffled",
            "standardized_negative_veto_shuffled",
            "residual_laplacian_adapter_shuffled",
            "residual_laplacian_router_shuffled",
        }:
            scale_keys = self._roll_keys_within_channel(scale_keys)
        scale_log_energy = None
        if self.multiscale_mode.startswith("energy_aligned"):
            scale_log_energy = self.multiscale_memory_log_energy[:, memory_indices]
            if self.multiscale_mode.endswith("_shuffled"):
                scale_log_energy = self._roll_keys_within_channel(scale_log_energy)

        if self.multiscale_mode.startswith(
            ("residual_laplacian_adapter", "residual_laplacian_router")
        ):
            anchor, fusion_scores, anchor_indices, _ = self._retrieve_one_scale(
                x_norm,
                self.memory_keys[memory_indices],
                None,
                index,
                mode,
                batch_size,
                memory_indices,
            )
            anchor_components = temporal_laplacian(anchor, self.scale_factors)
            residuals = []
            residual_scores = []
            scale_indices = [anchor_indices]
            for scale, query_view in enumerate(self._query_views(x_norm)):
                retrieved_scale, scores, selected, _ = self._retrieve_one_scale(
                    query_view,
                    scale_keys[scale],
                    scale,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                )
                residuals.append(retrieved_scale - anchor_components[scale])
                residual_scores.append(scores)
                scale_indices.append(selected)
            if self.multiscale_mode.startswith("residual_laplacian_router"):
                coefficients = self.residual_laplacian_router(
                    fusion_scores,
                    torch.stack(residual_scores),
                    batch_size=batch_size,
                    channels=self.channels,
                    shuffle_queries=self.multiscale_mode.endswith("_shuffled"),
                )
                self._residual_coefficient_sum += float(
                    coefficients.detach().sum().cpu()
                )
                self._residual_coefficient_square_sum += float(
                    coefficients.detach().square().sum().cpu()
                )
                self._residual_coefficient_count += coefficients.numel()
            else:
                coefficients = torch.tanh(self.residual_laplacian_logits)[:, None]
            retrieved = anchor + torch.sum(
                coefficients.unsqueeze(-1) * torch.stack(residuals), dim=0
            )
        elif self.multiscale_mode.startswith(
            (
                "exact_candidate_consensus",
                "moment_anchored_consensus",
                "learned_scale_consensus",
                "predictive_fidelity_consensus",
                "negative_consensus_veto",
                "standardized_negative_veto",
            )
        ):
            retrieved, fusion_scores, topk_indices, confidence_mean = (
                self._exact_candidate_consensus_retrieval(
                    x_norm,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                    scale_keys,
                )
            )
            scale_indices = [topk_indices]
        elif self.multiscale_mode.startswith("anchored_laplacian"):
            with stage("timegeo.normalize_views"):
                query_views = self._query_views(x_norm)
            retrieved_scales = []
            score_scales = []
            scale_indices = []
            for scale, query_view in enumerate(query_views[:-1]):
                retrieved_scale, scores, selected, _ = self._retrieve_one_scale(
                    query_view,
                    scale_keys[scale],
                    scale,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                )
                retrieved_scales.append(retrieved_scale)
                score_scales.append(scores)
                scale_indices.append(selected)
            anchor, anchor_scores, anchor_indices, _ = self._retrieve_one_scale(
                x_norm,
                self.memory_keys[memory_indices],
                None,
                index,
                mode,
                batch_size,
                memory_indices,
            )
            retrieved = anchored_laplacian_reconstruction(
                anchor, torch.stack(retrieved_scales), self.scale_factors
            )
            score_scales.append(anchor_scores)
            scale_indices.append(anchor_indices)
            fusion_scores = torch.stack(score_scales).mean(dim=0)
        elif self.multiscale_mode.startswith("predictive_spectral"):
            base_query_keys = F.normalize(self.encoder(x_norm), dim=-1)
            retrieved_scales = []
            score_scales = []
            scale_indices = []
            for scale in range(len(self.scale_factors)):
                retrieved, scores, selected = self._retrieve_predictive_scale(
                    x_norm,
                    base_query_keys,
                    scale_keys[scale],
                    self.predictive_scale_maps[scale],
                    scale,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                )
                retrieved_scales.append(retrieved)
                score_scales.append(scores)
                scale_indices.append(selected)
            retrieved = torch.stack(retrieved_scales).sum(dim=0)
            fusion_scores = torch.stack(score_scales).mean(dim=0)
        elif self.multiscale_mode.startswith("coherent_joint"):
            query_views = self._query_views(x_norm)
            retrieved, fusion_scores, topk_indices = (
                self._coherent_joint_retrieval(
                    query_views,
                    scale_keys,
                    x_norm,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                )
            )
            scale_indices = [topk_indices]
        elif self.multiscale_mode.startswith("hierarchical"):
            retrieved, fusion_scores, topk_indices = self._hierarchical_retrieval(
                x_norm, index, mode, batch_size, memory_indices, scale_keys
            )
            scale_indices = [topk_indices]
        else:
            query_views = self._query_views(x_norm)
            
            
            
            
            
            coupling_active = self.multiscale_mode.startswith(
                (
                    "cross_variable_event_coupling",
                    "cross_variable_event_clock",
                    "cross_variable_pareto_screen",
                )
            )
            requires_stacked_retrievals = coupling_active or self.multiscale_mode.startswith(
                "evidence_routed"
            )
            retrieved_scales = [] if requires_stacked_retrievals else None
            retrieved_total = None
            score_scales = []
            scale_reliabilities = []
            scale_predictions = []
            scale_indices = []
            candidate_future_scales = []
            for scale, query_view in enumerate(query_views):
                retrieval = self._retrieve_one_scale(
                    query_view,
                    scale_keys[scale],
                    scale,
                    index,
                    mode,
                    batch_size,
                    memory_indices,
                    (
                        log_rms_energy(query_view)
                        if scale_log_energy is not None
                        else None
                    ),
                    (
                        scale_log_energy[scale]
                        if scale_log_energy is not None
                        else None
                    ),
                    return_candidate_futures=coupling_active,
                    compute_reliability=self.multiscale_mode.startswith(
                        "posterior_shrink"
                    ),
                )
                if coupling_active:
                    retrieved, scores, selected, reliability, candidate_futures = retrieval
                    candidate_future_scales.append(candidate_futures)
                else:
                    retrieved, scores, selected, reliability = retrieval
                if self.reliability_enabled:
                    scale_reliabilities.append(reliability)
                    scale_predictions.append(retrieved)
                if self.multiscale_mode.startswith("posterior_shrink"):
                    retrieved = retrieved * reliability.unsqueeze(-1)
                    self._posterior_reliability_sum += float(
                        reliability.detach().sum().cpu()
                    )
                    self._posterior_reliability_count += reliability.numel()
                if retrieved_scales is None:
                    with stage("timegeo.cross_scale_accumulate"):
                        retrieved_total = (
                            retrieved
                            if retrieved_total is None
                            else retrieved_total + retrieved
                        )
                else:
                    retrieved_scales.append(retrieved)
                score_scales.append(scores)
                scale_indices.append(selected)
            if self.multiscale_mode.startswith("cross_variable_event_coupling"):
                query_signatures = cross_variable_event_signatures(
                    x.permute(0, 2, 1).contiguous(), self.scale_factors
                ).reshape(batch_size * self.channels, -1)
                candidate_signatures = torch.stack(
                    [
                        self.cross_variable_event_memory_signatures.to(x.device)[selected]
                        for selected in scale_indices
                    ]
                )
                if self.multiscale_mode.endswith("_shuffled"):
                    candidate_signatures = torch.roll(
                        candidate_signatures, shifts=1, dims=2
                    )
                coupled_weights = cross_variable_event_coupled_weights(
                    torch.stack(score_scales), query_signatures, candidate_signatures
                )
                stacked_futures = torch.stack(candidate_future_scales)
                retrieved = torch.sum(
                    coupled_weights.unsqueeze(-1) * stacked_futures, dim=2
                ).sum(dim=0)
            elif self.multiscale_mode.startswith("cross_variable_pareto_screen"):
                query_signatures = cross_variable_event_signatures(
                    x.permute(0, 2, 1).contiguous(), self.scale_factors
                ).reshape(batch_size * self.channels, -1)
                candidate_signatures = torch.stack(
                    [
                        self.cross_variable_event_memory_signatures.to(x.device)[selected]
                        for selected in scale_indices
                    ]
                )
                if self.multiscale_mode.endswith("_shuffled"):
                    candidate_signatures = torch.roll(
                        candidate_signatures, shifts=1, dims=2
                    )
                pareto = cross_variable_pareto_mask(
                    torch.stack(score_scales),
                    query_signatures,
                    candidate_signatures,
                )
                screened_scores = torch.stack(score_scales).masked_fill(
                    ~pareto, float("-inf")
                )
                screened_weights = torch.stack(
                    [
                        self._compute_neighbor_weights(scale_scores)
                        for scale_scores in screened_scores
                    ]
                )
                stacked_futures = torch.stack(candidate_future_scales)
                retrieved = torch.sum(
                    screened_weights.unsqueeze(-1) * stacked_futures, dim=2
                ).sum(dim=0)
            elif self.multiscale_mode.startswith("cross_variable_event_clock"):
                candidate_starts = torch.stack(
                    [self.memory_starts[selected] for selected in scale_indices]
                ).reshape(
                    len(self.scale_factors), batch_size, self.channels, self.topk
                ).to(dtype=stacked.dtype)
                clock_weights = cross_variable_event_clock_weights(
                    torch.stack(score_scales).reshape(
                        len(self.scale_factors), batch_size, self.channels, self.topk
                    ),
                    candidate_starts,
                    self.total_window,
                    shuffled_peers=self.multiscale_mode.endswith("_shuffled"),
                ).reshape(len(self.scale_factors), batch_size * self.channels, self.topk)
                stacked_futures = torch.stack(candidate_future_scales)
                retrieved = torch.sum(
                    clock_weights.unsqueeze(-1) * stacked_futures, dim=2
                ).sum(dim=0)
            elif self.multiscale_mode.startswith("evidence_routed"):
                stacked = torch.stack(retrieved_scales)
                gates = self.scale_evidence_router(
                    torch.stack(score_scales),
                    batch_size=batch_size,
                    channels=self.channels,
                    shuffle_queries=self.multiscale_mode.endswith("_shuffled"),
                )
                retrieved = torch.sum(stacked * gates.unsqueeze(-1), dim=0)
                self._scale_gate_sum += float(gates.detach().sum().cpu())
                self._scale_gate_square_sum += float(
                    gates.detach().square().sum().cpu()
                )
                self._scale_gate_count += gates.numel()
            else:
                if retrieved_total is None:
                    raise RuntimeError("multiscale retrieval accumulation failed")
                retrieved = (
                    retrieved_total / float(len(self.scale_factors))
                    if self.multiscale_mode == "parallel_mean"
                    else retrieved_total
                )
            with stage("timegeo.cross_scale_accumulate"):
                fusion_scores = torch.stack(score_scales).mean(dim=0)
            if self.multiscale_mode.startswith("scale_resolved_fusion"):
                self._last_scale_scores = torch.stack(score_scales).reshape(
                    len(self.scale_factors), batch_size, self.channels, self.topk
                )

        if len(scale_indices) > 1:
            reference = scale_indices[-1]
            overlaps = []
            for selected in scale_indices[:-1]:
                overlaps.append(
                    (selected.unsqueeze(-1) == reference.unsqueeze(-2))
                    .any(dim=-1)
                    .float()
                    .mean()
                )
            overlap = torch.stack(overlaps).mean()
            self._scale_candidate_overlap_sum += float(overlap.detach().cpu())
            self._scale_candidate_overlap_count += 1

        with stage("timegeo.output_denorm"):
            output_scale, output_shift = self.output_gate(x_norm)
            retrieved = retrieved * output_scale + output_shift
            retrieved = retrieved * query_std + query_mean
            retrieval_pred = retrieved.reshape(
                batch_size, channels, self.pred_len
            ).permute(0, 2, 1).contiguous()
        if self.reliability_enabled:
            self._last_component_reliabilities = torch.stack(scale_reliabilities).reshape(-1, batch_size, channels)
            components = torch.stack(scale_predictions) * output_scale.unsqueeze(0) * query_std.unsqueeze(0)
            components = torch.cat((components[:1] + (output_shift * query_std + query_mean).unsqueeze(0), components[1:]), dim=0)
            self._last_component_predictions = components.reshape(-1, batch_size, channels, self.pred_len).permute(0, 1, 3, 2)
        self._last_confidence_execution = "factorized"
        diagnostics = {
            "confidence_mean": (
                confidence_mean
                if self.multiscale_mode.startswith(
                    (
                        "exact_candidate_consensus",
                        "moment_anchored_consensus",
                        "learned_scale_consensus",
                        "predictive_fidelity_consensus",
                        "negative_consensus_veto",
                        "standardized_negative_veto",
                    )
                )
                else fusion_scores.detach().mean()
            ),
            "same_channel_ratio": x_norm.new_ones(()),
        }
        self._last_scale_indices = scale_indices
        self._last_scale_scores = score_scales if 'score_scales' in locals() else [fusion_scores]
        return (
            retrieval_pred,
            fusion_scores.reshape(batch_size, channels, -1),
            diagnostics,
        )

    def forward_original_multiscale(self, x, index, mode):
        """Evaluate the unchanged Laplacian path for tri-branch fusion."""
        if not self.multiscale_mode.startswith("tri_relational_multiscale"):
            raise RuntimeError("original multiscale path is only defined for tri mode")
        active_mode = self.multiscale_mode
        active_keys = self.multiscale_memory_keys
        try:
            self.multiscale_mode = "laplacian"
            self.multiscale_memory_keys = self.original_multiscale_memory_keys
            return self.forward(x, index, mode)
        finally:
            self.multiscale_memory_keys = active_keys
            self.multiscale_mode = active_mode

    def get_memory_summary(self) -> dict[str, Any]:
        summary = super().get_memory_summary()
        relational_summary = (
            self.relational_exact_branch.get_memory_summary()
            if self.relational_exact_branch is not None
            else {}
        )
        summary.update(
            {
                "architecture": "MultiscaleRetrievalGeometry",
                "Geometry_multiscale_mode": self.multiscale_mode,
                "Geometry_multiscale_factors": list(self.scale_factors),
                "Geometry_multiscale_pool": self.hierarchical_pool,
                "Geometry_multiscale_key_build_seconds": self._scale_key_build_seconds,
                "Geometry_multiscale_candidate_overlap": (
                    self._scale_candidate_overlap_sum
                    / max(self._scale_candidate_overlap_count, 1)
                ),
                "Geometry_multiscale_target_owned_future": True,
                "Geometry_relational_cache_path": relational_summary.get(
                    "memory_cache_path"
                ),
                "Geometry_relational_cache_used": bool(
                    relational_summary.get("memory_cache_used")
                ),
                "Geometry_relational_contrast_max_peers": relational_summary.get(
                    "Geometry_relational_contrast_max_peers", 0
                ),
                "Geometry_relational_contrast_effective_peers_mean": (
                    relational_summary.get(
                        "Geometry_relational_contrast_effective_peers_mean", 0.0
                    )
                ),
                "Geometry_relational_contrast_coherence_mean": relational_summary.get(
                    "Geometry_relational_contrast_coherence_mean", 0.0
                ),
                "Geometry_multiscale_posterior_reliability_mean": (
                    self._posterior_reliability_sum
                    / max(self._posterior_reliability_count, 1)
                ),
                "Geometry_multiscale_predictive_representation": (
                    "closed_form_history_to_scale_spectral_future"
                    if self.multiscale_mode.startswith("predictive_spectral")
                    else (
                        "scale_specific_future_contrastive_encoder"
                        if self.multiscale_mode.startswith("scale_predictive_encoder")
                        else (
                            "cross_variable_relational_predictive_encoder"
                            if "relational_multiscale" in self.multiscale_mode
                            else "none"
                        )
                    )
                ),
                "Geometry_multiscale_relational_teacher_cache_used": (
                    self._relational_teacher_cache_used
                    if "relational_multiscale" in self.multiscale_mode
                    else False
                ),
                "Geometry_multiscale_scale_encoder_cache_used": (
                    self._scale_encoder_cache_used
                    if self.multiscale_mode.startswith("scale_predictive_encoder")
                    else False
                ),
                "Geometry_multiscale_scale_encoder_final_losses": (
                    self._scale_encoder_losses
                    if self.multiscale_mode.startswith("scale_predictive_encoder")
                    else []
                ),
                "Geometry_multiscale_predictive_frequencies": (
                    [list(values) for values in self._predictive_spectral_frequencies]
                    if self.multiscale_mode.startswith("predictive_spectral")
                    else []
                ),
                "Geometry_multiscale_anchor": (
                    "exact_Geometry_finest_component"
                    if self.multiscale_mode.startswith("anchored_laplacian")
                    else "none"
                ),
                "Geometry_multiscale_retrieval_evidence": (
                    "standardized_shape_plus_log_rms_match"
                    if self.multiscale_mode.startswith("energy_aligned")
                    else "shape_only"
                ),
                "Geometry_multiscale_scale_router": (
                    "shared_query_topk_evidence"
                    if self.multiscale_mode.startswith("evidence_routed")
                    else "none"
                ),
                "Geometry_multiscale_scale_gate_mean": (
                    self._scale_gate_sum / max(self._scale_gate_count, 1)
                ),
                "Geometry_multiscale_scale_gate_std": (
                    max(
                        self._scale_gate_square_sum
                        / max(self._scale_gate_count, 1)
                        - (
                            self._scale_gate_sum
                            / max(self._scale_gate_count, 1)
                        )
                        ** 2,
                        0.0,
                    )
                    ** 0.5
                ),
                "Geometry_multiscale_scale_router_weight_l2": float(
                    torch.linalg.vector_norm(
                        self.scale_evidence_router.projection.weight.detach()
                    ).cpu()
                ),
                "Geometry_multiscale_scale_fusion": (
                    "shared_original_Geometry_fusion_per_scale"
                    if self.multiscale_mode.startswith("scale_resolved_fusion")
                    else "none"
                ),
                "Geometry_multiscale_candidate_calibration": (
                    "exact_topk_moment_anchored_multiscale_consensus"
                    if self.multiscale_mode.startswith("moment_anchored_consensus")
                    else "exact_topk_learned_scale_simplex"
                    if self.multiscale_mode.startswith("learned_scale_consensus")
                    else "exact_topk_predictive_fidelity_consensus"
                    if self.multiscale_mode.startswith("predictive_fidelity_consensus")
                    else "exact_topk_negative_multiscale_veto"
                    if self.multiscale_mode.startswith("negative_consensus_veto")
                    else "exact_topk_standardized_negative_multiscale_veto"
                    if self.multiscale_mode.startswith("standardized_negative_veto")
                    else "exact_topk_multiscale_product_of_experts"
                    if self.multiscale_mode.startswith("exact_candidate_consensus")
                    else "none"
                ),
                "Geometry_multiscale_candidate_weight_mean_abs_delta": (
                    self._candidate_consensus_weight_delta_sum
                    / max(self._candidate_consensus_weight_delta_count, 1)
                ),
                "Geometry_multiscale_residual_coefficients": (
                    torch.tanh(self.residual_laplacian_logits.detach())
                    .cpu()
                    .tolist()
                    if self.multiscale_mode.startswith(
                        "residual_laplacian_adapter"
                    )
                    else []
                ),
                "Geometry_multiscale_residual_router_mean": (
                    self._residual_coefficient_sum
                    / max(self._residual_coefficient_count, 1)
                ),
                "Geometry_multiscale_residual_router_std": (
                    max(
                        self._residual_coefficient_square_sum
                        / max(self._residual_coefficient_count, 1)
                        - (
                            self._residual_coefficient_sum
                            / max(self._residual_coefficient_count, 1)
                        )
                        ** 2,
                        0.0,
                    )
                    ** 0.5
                ),
                "Geometry_multiscale_residual_router_weight_l2": float(
                    torch.linalg.vector_norm(
                        self.residual_laplacian_router.projection.weight.detach()
                    ).cpu()
                ),
                "Geometry_multiscale_evidence_simplex": (
                    torch.softmax(
                        self.scale_evidence_logits.detach(), dim=0
                    ).cpu().tolist()
                    if self.multiscale_mode.startswith("learned_scale_consensus")
                    else []
                ),
                "Geometry_multiscale_predictive_fidelity_simplex": (
                    self.scale_predictive_fidelity_weights.detach().cpu().tolist()
                    if self.multiscale_mode.startswith(
                        "predictive_fidelity_consensus"
                    )
                    else []
                ),
            }
        )
        for name in (
            "rolegeo_validity_variant",
            "rolegeo_validity_semantics",
            "relation_density",
            "effective_peers_mean",
            "positive_edge_ratio",
            "negative_edge_ratio",
            "retrieval_similarity_mean",
            "same_channel_ratio",
            "relation_peer_ratio",
            "relation_drift_mean",
            "relation_sign_agreement_mean",
        ):
            if name in relational_summary:
                summary[name] = relational_summary[name]
        return summary


class MultiscaleRetrievalGeometryModel(GeometryModel):
    def __init__(self, configs, backbone_class: type[nn.Module]):
        relational_reference = None
        requested_mode = str(getattr(configs, "Geometry_multiscale_mode", "laplacian"))
        if requested_mode.startswith("cross_variable_dual_path_reproduced"):
            from ts_benchmark.baselines.coreg_public._engine.role_geometry import (
                RelationalPredictiveContrastGeometryModel,
            )

            reference_config = copy.deepcopy(configs)
            configured_control = str(
                getattr(configs, "memory_relational_contrast_control", "")
            )
            reference_control = configured_control or (
                "zero_peer" if requested_mode.endswith("_zero") else
                ("time_shuffled" if requested_mode.endswith("_shuffled") else "aligned")
            )
            setattr(
                reference_config,
                "memory_relational_contrast_control",
                reference_control,
            )
            
            
            
            setattr(reference_config, "memory_retrieval_truth_control", "full")
            setattr(reference_config, "memory_retrieval_truth_use_base_cache", True)
            setattr(
                reference_config,
                "memory_require_existing_cache",
                bool(getattr(configs, "memory_require_existing_cache", False)),
            )
            configured_relation_root = getattr(
                configs, "memory_relational_contrast_cache_root", None
            )
            if configured_relation_root:
                setattr(reference_config, "memory_cache_root", str(configured_relation_root))
            else:
                setattr(
                    reference_config,
                    "memory_cache_root",
                    str(
                        Path(getattr(configs, "memory_cache_root", ".memory_cache"))
                        / f"dual_path_relational_{reference_control}"
                    ),
                )
            cpu_reference_rng = torch.random.get_rng_state()
            cuda_reference_rng = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                torch.manual_seed(int(getattr(configs, "memory_seed", 2027)))
                relational_reference = RelationalPredictiveContrastGeometryModel(
                    reference_config, backbone_class=backbone_class
                )
            finally:
                torch.random.set_rng_state(cpu_reference_rng)
                if cuda_reference_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_reference_rng)
        super().__init__(configs, backbone_class=backbone_class)
        original_branch = self.memory_branch
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            branch = MultiscaleRetrievalGeometryBranch(configs)
            branch.load_state_dict(original_branch.state_dict(), strict=False)
            self.memory_branch = branch
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if self.memory_branch.multiscale_mode.startswith("tri_relational_multiscale"):
            original_fusion = self.fusion
            self.fusion = RelationalTriFusion(
                self.memory_branch.topk,
                int(getattr(configs, "memory_fusion_hidden", 128)),
            )
            if "_nested" in self.memory_branch.multiscale_mode:
                self.fusion.initialize_from_two_branch(original_fusion)
        if self.memory_branch.multiscale_mode.startswith("cross_variable_dual_path"):
            original_fusion = self.fusion
            self.fusion = DualPathFusion(
                self.memory_branch.topk,
                int(getattr(configs, "memory_fusion_hidden", 128)),
                self.pred_len,
            )
            self.fusion.multiscale = original_fusion
            if relational_reference is not None:
                self.memory_branch.relational_exact_branch = (
                    relational_reference.memory_branch
                )
                self.fusion.relational = relational_reference.fusion
        self._scale_fusion_weight_sum = 0.0
        self._scale_fusion_weight_square_sum = 0.0
        self._scale_fusion_weight_count = 0
        self._tri_original_weight_sum = 0.0
        self._tri_relational_weight_sum = 0.0
        self._tri_weight_count = 0
        self._weight_trace_dir = getattr(configs, "coreg_weight_trace_dir", "")
        self._weight_trace_batch = 0
        self._retrieval_trace_dir = getattr(configs, "coreg_retrieval_trace_dir", "")
        self._retrieval_trace_dir = getattr(configs, "coreg_retrieval_trace_dir", "")
        self._retrieval_trace_batch = 0
        self._retrieval_trace_max_rows = int(
            getattr(configs, "coreg_retrieval_trace_max_rows", 256)
        )

    def _save_retrieval_trace(self, x_enc, scale_indices, scale_scores) -> None:
        """Persist exact Top-K identities/scores for a bounded test batch."""
        if not self._retrieval_trace_dir or not scale_indices:
            return
        indices = scale_indices[-1]
        scores = scale_scores[-1] if scale_scores else None
        if indices is None:
            return
        rows = min(int(indices.shape[0]), self._retrieval_trace_max_rows)
        trace_dir = Path(self._retrieval_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        query_matrix = x_enc.permute(0, 2, 1).reshape(-1, x_enc.shape[1])
        payload = {
            "query_history": query_matrix[:rows]
            .detach().to(dtype=torch.float32).cpu().numpy(),
            "query_channels": (torch.arange(self.channels, device=x_enc.device)
                               .unsqueeze(0).expand(x_enc.shape[0], -1)
                               .reshape(-1)[:rows].cpu().numpy()),
            "neighbor_indices": indices[:rows].detach().cpu().numpy(),
            "neighbor_starts": self.memory_starts[indices[:rows]].detach().cpu().numpy(),
            "neighbor_channels": self.memory_channels[indices[:rows]].detach().cpu().numpy(),
            "neighbor_future": self.memory_future[indices[:rows]].detach().to(dtype=torch.float32).cpu().numpy(),
        }
        if scores is not None:
            payload["neighbor_scores"] = scores[:rows].detach().to(dtype=torch.float32).cpu().numpy()
        np.savez_compressed(
            trace_dir / f"retrieval_trace_{self._retrieval_trace_batch:06d}.npz",
            **payload,
        )
        self._retrieval_trace_batch += 1

    def _save_weight_trace(self, multiscale_weights, relational_weights) -> None:
        """Persist raw per-window/per-channel GeoRead weights when requested."""
        if not self._weight_trace_dir:
            return
        trace_dir = Path(self._weight_trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        batch_id = self._weight_trace_batch
        self._weight_trace_batch += 1
        np.savez_compressed(
            trace_dir / f"weights_batch_{batch_id:06d}.npz",
            multiscale_weights=multiscale_weights.detach().to(dtype=torch.float32).cpu().numpy(),
            relational_weights=relational_weights.detach().to(dtype=torch.float32).cpu().numpy(),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        index: torch.Tensor | None = None,
        mode: str = "train",
        return_branches: bool = False,
    ):
        if self.memory_branch.multiscale_mode.startswith("cross_variable_dual_path"):
            relational_branch = self.memory_branch.relational_exact_branch
            if relational_branch is not None:
                relational_branch.reliability_strength = self.memory_branch.reliability_strength
                relational_branch.reliability_enabled = self.memory_branch.reliability_enabled
            if relational_branch is None:
                raise RuntimeError("dual-path relational branch was not constructed")
            with stage("shared.backbone"):
                base_pred = self.backbone(x_enc, None, None, None)
            multiscale_pred, multiscale_scores, multiscale_diag = self.memory_branch(
                x_enc, index, mode
            )
            relational_pred, relational_scores, relational_diag = relational_branch(
                x_enc, index, mode
            )
            self._last_multiscale_retrieval_prediction = multiscale_pred
            self._last_relational_retrieval_prediction = relational_pred
            with stage("timegeo.backbone_fusion"):
                multiscale_fused, multiscale_weights = self.fusion.multiscale(
                    multiscale_pred, base_pred, multiscale_scores
                )
            with stage("rolegeo.backbone_fusion"):
                relational_fused, relational_weights = self.fusion.relational(
                    relational_pred, base_pred, relational_scores
                )
            if self.memory_branch.reliability_enabled:
                rel = relational_branch._last_candidate_reliability.unsqueeze(1)
                relational_fused = base_pred + rel * (relational_fused - base_pred)
                component_rels = self.memory_branch._last_component_reliabilities.unsqueeze(2)
                if self.memory_branch.moving_average_kernel:
                    base_components = moving_average_components(base_pred.permute(0, 2, 1), self.memory_branch.moving_average_kernel).permute(0, 1, 3, 2)
                else:
                    base_components = temporal_laplacian(base_pred.permute(0, 2, 1), self.memory_branch.scale_factors).permute(0, 1, 3, 2)
                
                retrieval_weight = multiscale_weights[..., 0].unsqueeze(1).unsqueeze(0)
                component_preds = self.memory_branch._last_component_predictions
                multiscale_fused = (base_components + retrieval_weight * component_rels * (component_preds - base_components)).sum(dim=0)
            self._last_multiscale_path_prediction = multiscale_fused
            self._last_relational_path_prediction = relational_fused
            with stage("dual_merge.path_average"):
                if "_horizon" in self.memory_branch.multiscale_mode:
                    relational_gate = torch.sigmoid(self.fusion.horizon_logits).view(
                        1, -1, 1
                    )
                    prediction = (
                        multiscale_fused
                        + relational_gate * (relational_fused - multiscale_fused)
                    )
                else:
                    prediction = 0.5 * (multiscale_fused + relational_fused)
                branch_weights = 0.5 * (multiscale_weights + relational_weights)
            self._last_branch_weights = branch_weights
            self._last_multiscale_weights = multiscale_weights
            self._last_relational_weights = relational_weights
            if mode == "test":
                self._save_weight_trace(multiscale_weights, relational_weights)
                trace_dir = getattr(self, "_retrieval_trace_dir", "")
                if trace_dir:
                    self.memory_branch._retrieval_trace_dir = str(Path(trace_dir) / "temporal")
                    self.memory_branch._save_retrieval_trace(
                        x_enc,
                        getattr(self.memory_branch, "_last_scale_indices", []),
                        getattr(self.memory_branch, "_last_scale_scores", []),
                        index,
                    )
                    if relational_branch is not None:
                        relational_branch._retrieval_trace_dir = str(Path(trace_dir) / "role")
                        relational_branch._save_retrieval_trace(
                            x_enc,
                            getattr(relational_branch, "_last_scale_indices", []),
                            getattr(relational_branch, "_last_scale_scores", []),
                            index,
                        )
            if mode == "test":
                with stage("dual_merge.test_diagnostics_sync"):
                    self._debug["count"] += 1.0
                    self._debug["retrieval_weight_sum"] += float(
                        branch_weights[..., 0].detach().mean().cpu()
                    )
                    self._debug["base_weight_sum"] += float(
                        branch_weights[..., 1].detach().mean().cpu()
                    )
                    self._debug["confidence_sum"] += 0.5 * float(
                        multiscale_diag["confidence_mean"].detach().cpu()
                        + relational_diag["confidence_mean"].detach().cpu()
                    )
                    self._debug["same_channel_ratio_sum"] += float(
                        multiscale_diag["same_channel_ratio"].detach().cpu()
                    )
            prediction = prediction[:, -self.pred_len :, :]
            if return_branches:
                retrieval_blend = 0.5 * (multiscale_pred + relational_pred)
                return prediction, base_pred, retrieval_blend
            return prediction

        if self.memory_branch.multiscale_mode.startswith("tri_relational_multiscale"):
            base_pred = self.backbone(x_enc, None, None, None)
            relational_pred, relational_scores, relational_diag = self.memory_branch(
                x_enc, index, mode
            )
            original_pred, original_scores, original_diag = (
                self.memory_branch.forward_original_multiscale(x_enc, index, mode)
            )
            prediction, branch_weights = self.fusion(
                original_pred,
                relational_pred,
                base_pred,
                original_scores,
                relational_scores,
            )
            self._last_branch_weights = branch_weights
            if mode == "test":
                self._debug["count"] += 1.0
                self._debug["retrieval_weight_sum"] += float(
                    branch_weights[..., :2].detach().sum(dim=-1).mean().cpu()
                )
                self._debug["base_weight_sum"] += float(
                    branch_weights[..., 2].detach().mean().cpu()
                )
                self._debug["confidence_sum"] += float(
                    relational_diag["confidence_mean"].detach().cpu()
                )
                self._debug["same_channel_ratio_sum"] += float(
                    original_diag["same_channel_ratio"].detach().cpu()
                )
                self._tri_original_weight_sum += float(
                    branch_weights[..., 0].detach().mean().cpu()
                )
                self._tri_relational_weight_sum += float(
                    branch_weights[..., 1].detach().mean().cpu()
                )
                self._tri_weight_count += 1
            prediction = prediction[:, -self.pred_len :, :]
            if return_branches:
                retrieval_total = branch_weights[..., :2].sum(dim=-1).clamp_min(1e-6)
                retrieval_blend = (
                    original_pred * branch_weights[..., 0].unsqueeze(1)
                    + relational_pred * branch_weights[..., 1].unsqueeze(1)
                ) / retrieval_total.unsqueeze(1)
                return prediction, base_pred, retrieval_blend
            return prediction

        if not self.memory_branch.multiscale_mode.startswith(
            "scale_resolved_fusion"
        ):
            return super().forward(x_enc, index, mode, return_branches)

        base_pred = self.backbone(x_enc, None, None, None)
        retrieval_pred, _, diagnostics = self.memory_branch(x_enc, index, mode)
        scale_scores = self.memory_branch._last_scale_scores
        if scale_scores is None:
            raise RuntimeError("scale-resolved fusion scores were not produced")
        if self.memory_branch.multiscale_mode.endswith("_shuffled"):
            scale_scores = torch.roll(scale_scores, shifts=1, dims=0)
        scale_count, batch_size, channels, _ = scale_scores.shape
        if self.fusion_mode == "fixed":
            scale_weights = base_pred.new_empty(
                scale_count, batch_size, channels, 2
            )
            scale_weights[..., 0] = self.fixed_gamma
            scale_weights[..., 1] = 1.0 - self.fixed_gamma
        else:
            scale_weights = self.fusion.network(
                scale_scores.reshape(-1, self.memory_branch.topk)
            ).reshape(scale_count, batch_size, channels, 2)
        prediction = scale_resolved_blend(
            retrieval_pred,
            base_pred,
            scale_weights,
            self.memory_branch.scale_factors,
        )
        branch_weights = scale_weights.mean(dim=0)
        self._last_branch_weights = branch_weights
        if mode == "test":
            self._debug["count"] += 1.0
            self._debug["retrieval_weight_sum"] += float(
                branch_weights[..., 0].detach().mean().cpu()
            )
            self._debug["base_weight_sum"] += float(
                branch_weights[..., 1].detach().mean().cpu()
            )
            self._debug["confidence_sum"] += float(
                diagnostics["confidence_mean"].detach().cpu()
            )
            self._debug["same_channel_ratio_sum"] += float(
                diagnostics["same_channel_ratio"].detach().cpu()
            )
            retrieval_weights = scale_weights[..., 0].detach()
            self._scale_fusion_weight_sum += float(retrieval_weights.sum().cpu())
            self._scale_fusion_weight_square_sum += float(
                retrieval_weights.square().sum().cpu()
            )
            self._scale_fusion_weight_count += retrieval_weights.numel()
        prediction = prediction[:, -self.pred_len :, :]
        if return_branches:
            return prediction, base_pred, retrieval_pred
        return prediction

    def get_debug_summary(self):
        summary = super().get_debug_summary()
        summary.update(self.memory_branch.get_memory_summary())
        summary["architecture"] = "MultiscaleRetrievalGeometry"
        count = max(self._scale_fusion_weight_count, 1)
        mean = self._scale_fusion_weight_sum / count
        summary["Geometry_multiscale_scale_fusion_retrieval_weight_mean"] = mean
        summary["Geometry_multiscale_scale_fusion_retrieval_weight_std"] = max(
            self._scale_fusion_weight_square_sum / count - mean**2, 0.0
        ) ** 0.5
        tri_count = max(self._tri_weight_count, 1)
        summary["Geometry_multiscale_tri_original_weight"] = (
            self._tri_original_weight_sum / tri_count
        )
        summary["Geometry_multiscale_tri_relational_weight"] = (
            self._tri_relational_weight_sum / tri_count
        )
        if isinstance(self.fusion, DualPathFusion):
            horizon_gate = torch.sigmoid(self.fusion.horizon_logits.detach()).cpu()
            summary["Geometry_multiscale_dual_horizon_gate_mean"] = float(
                horizon_gate.mean()
            )
            summary["Geometry_multiscale_dual_horizon_gate_first_quarter"] = float(
                horizon_gate[: max(1, self.pred_len // 4)].mean()
            )
            summary["Geometry_multiscale_dual_horizon_gate_last_quarter"] = float(
                horizon_gate[-max(1, self.pred_len // 4) :].mean()
            )
        return summary



