"""Deterministic counterfactual controls for StableRel."""
from __future__ import annotations

import torch

CONTROLS = {
    "aligned_stablerel", "zero_relation", "random_relation",
    "permuted_relation", "sign_shuffled_relation",
    "single_period_relation", "unstable_relation",
}


def _correlation(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    scaled = centered / centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    return scaled.T.matmul(scaled) / max(int(values.shape[0]), 1)


def _normalize(signed: torch.Tensor, strength: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    signed = signed.clone()
    strength = strength.clone()
    signed.fill_diagonal_(0.0)
    strength.fill_diagonal_(0.0)
    total = strength.sum(dim=1)
    weights = signed / total.clamp_min(1e-8).unsqueeze(1)
    coherence = (strength.square().sum(dim=1) / total.clamp_min(1e-8)).clamp(0.0, 1.0)
    empty = total <= 1e-8
    weights[empty] = 0.0
    coherence[empty] = 0.0
    return weights, coherence


def build_relation(values: torch.Tensor, control: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if control not in CONTROLS:
        raise ValueError(f"unknown StableRel control: {control}")
    differences = values[1:] - values[:-1]
    midpoint = int(differences.shape[0]) // 2
    if midpoint < 2 or differences.shape[0] - midpoint < 2:
        raise ValueError("StableRel requires two non-trivial training halves")
    first = _correlation(differences[:midpoint])
    second = _correlation(differences[midpoint:])
    stable = (first * second).clamp_min(0.0).sqrt()
    signed = torch.sign(first + second) * stable
    weights, coherence = _normalize(signed, stable)
    if control == "aligned_stablerel":
        return weights, coherence
    if control == "zero_relation":
        return torch.zeros_like(weights), torch.zeros_like(coherence)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    channels = weights.shape[0]
    if control == "permuted_relation":
        order = torch.randperm(channels, generator=generator, device="cpu").to(weights.device)
        return weights[order][:, order], coherence[order]
    if control == "sign_shuffled_relation":
        
        
        return -weights, coherence
    if control == "random_relation":
        result = torch.zeros_like(weights)
        for row in range(channels):
            active = torch.nonzero(weights[row] != 0, as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            values_row = weights[row, active]
            destinations = torch.tensor([i for i in range(channels) if i != row], device=weights.device)
            pick = torch.randperm(destinations.numel(), generator=generator)[:active.numel()].to(weights.device)
            result[row, destinations[pick]] = values_row
        return result, coherence
    if control == "single_period_relation":
        one = first.clone()
        one.fill_diagonal_(0.0)
        return _normalize(one, one.abs())
    conflict = (first * second).clamp_max(0.0).abs().sqrt()
    unstable_signed = torch.sign(first) * conflict
    return _normalize(unstable_signed, conflict)


def diagnostics(weights: torch.Tensor, first: torch.Tensor | None = None, second: torch.Tensor | None = None) -> dict:
    active = weights != 0
    result = {
        "channels": int(weights.shape[0]),
        "density": float(active.float().mean()),
        "positive_edges": int((weights > 0).sum()),
        "negative_edges": int((weights < 0).sum()),
        "effective_peers_mean": float(active.sum(dim=1).float().mean()),
    }
    if first is not None and second is not None:
        offdiag = ~torch.eye(first.shape[0], dtype=torch.bool, device=first.device)
        result["split_sign_agreement"] = float(((first * second > 0) & offdiag).float().sum() / offdiag.float().sum().clamp_min(1))
    return result


