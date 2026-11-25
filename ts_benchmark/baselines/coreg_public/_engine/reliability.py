"""Candidate-future disagreement and simple retrieval shrinkage."""
import torch
import torch.nn.functional as F


def candidate_disagreement(futures, weights):
    """Weighted mean variance over horizon; futures [N,K,H], weights [N,K]."""
    mean = (weights.unsqueeze(-1) * futures).sum(dim=1)
    return (weights.unsqueeze(-1) * (futures - mean.unsqueeze(1)).square()).sum(dim=1).mean(dim=-1)


def reliability(futures, weights, strength):
    return torch.exp(-float(strength) * candidate_disagreement(futures, weights))


def moving_average_components(x, kernel):
    """Additive trend and residual over the last axis, with replicated edges."""
    kernel = int(kernel)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("moving average kernel must be positive and odd")
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    pad = kernel // 2
    trend = F.avg_pool1d(F.pad(flat, (pad, pad), mode="replicate"), kernel, stride=1).reshape(shape)
    return torch.stack((trend, x - trend), dim=0)
