import numpy as np
import torch
import os
from pathlib import Path

from ts_benchmark.baselines.coreg_public import DLinearCoReG
from ts_benchmark.baselines.coreg_public._engine.role_geometry import RelationalPredictiveContrastGeometryBranch
from ts_benchmark.baselines.coreg_public._engine.memory_branch_base import _normalize_series


def test_relational_memory_fresh_and_cached_match(request):
    override = os.environ.get('RECORE_TEST_CACHE')
    tmp_path = Path(override) if override else request.getfixturevalue('tmp_path')
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    steps = rng.normal(size=(260, 3)).astype(np.float32)
    steps[:, 1] = 0.8 * steps[:, 0] + 0.2 * steps[:, 1]
    steps[:, 2] = -0.6 * steps[:, 0] + 0.4 * steps[:, 2]
    values = np.cumsum(steps, axis=0).astype(np.float32)
    config = DLinearCoReG(seq_len=24, pred_len=8, norm=True,
        memory_cache_root=str(tmp_path), memory_gmb_size=240, memory_topk=4,
        memory_pcl_epochs=1, memory_pcl_batch_size=64,
        memory_cluster_iterations=2, memory_encode_batch_size=31,
        memory_embedding_dim=8, memory_encoder_hidden=16).config
    config.enc_in = 3

    def make():
        torch.manual_seed(17)
        branch = RelationalPredictiveContrastGeometryBranch(config)
        branch.reliability_enabled = True
        branch.reliability_strength = 1.0
        branch.prepare_memory(values, torch.device('cpu'))
        branch.eval()
        return branch

    fresh = make()
    cached = make()
    assert not fresh.cache_used and cached.cache_used
    torch.testing.assert_close(fresh.memory_keys, cached.memory_keys, rtol=0, atol=1e-6)
    torch.testing.assert_close(fresh.memory_future, cached.memory_future, rtol=0, atol=0)
    torch.testing.assert_close(fresh.memory_starts, cached.memory_starts, rtol=0, atol=0)
    x = torch.tensor(values[-24:][None])
    _, _, q = _normalize_series(x.permute(0, 2, 1).reshape(3, 24))
    def rankings(branch):
        keys = branch._truth_keys(q)
        memory = branch.memory_keys[branch._same_channel_memory_indices]
        scores = torch.bmm(keys.reshape(1, 3, -1).permute(1, 0, 2), memory.transpose(1, 2)).squeeze(1)
        return scores.topk(4, dim=1).indices
    torch.testing.assert_close(rankings(fresh), rankings(cached), rtol=0, atol=0)
    with torch.no_grad():
        a = fresh(x, None, 'test')
        b = cached(x, None, 'test')
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=1e-6)
    torch.testing.assert_close(fresh._last_candidate_reliability, cached._last_candidate_reliability, rtol=0, atol=1e-6)
    
    pair_ids = fresh.memory_starts.cpu() * 3 + fresh.memory_channels.cpu()
    fresh.encode_batch_size = 32
    reencoded = fresh._encode_pairs(torch.from_numpy(values), pair_ids, torch.device('cpu'))
    torch.testing.assert_close(fresh.memory_keys, reencoded, rtol=0, atol=1e-6)
    
    
    path = Path(fresh.cache_path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    payload['memory_keys'] = torch.zeros_like(payload['memory_keys'])
    torch.save(payload, path)
    legacy = make()
    torch.testing.assert_close(fresh.memory_keys, legacy.memory_keys, rtol=0, atol=1e-6)
