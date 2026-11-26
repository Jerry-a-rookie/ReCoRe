import importlib.util
from pathlib import Path
import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'ts_benchmark/baselines/recore_public/_engine/reliability.py'
spec = importlib.util.spec_from_file_location('reliability', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_decomposition_reconstructs_and_handles_short_constant_input():
    x = torch.tensor([[3., 3., 3.], [1., 2., 3.]])
    c = m.moving_average_components(x, 25)
    torch.testing.assert_close(c.sum(0), x)
    torch.testing.assert_close(c[1, 0], torch.zeros(3))
    with pytest.raises(ValueError):
        m.moving_average_components(x, 4)

def test_weighted_disagreement_and_shrinkage():
    w = torch.tensor([[.5, .5]])
    same = torch.tensor([[[1., 2.], [1., 2.]]])
    different = torch.tensor([[[0., 1.], [2., 3.]]])
    torch.testing.assert_close(m.candidate_disagreement(same, w), torch.zeros(1))
    torch.testing.assert_close(m.candidate_disagreement(different, w), torch.ones(1))
    assert m.reliability(different, w, 1).item() < m.reliability(same, w, 1).item()
    assert torch.isfinite(m.reliability(different, w, 1)).all()
