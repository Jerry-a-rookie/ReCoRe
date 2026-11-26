from ts_benchmark.baselines.coreg_public import BACKBONE_ADAPTERS


def test_six_public_adapters_are_exposed():
    assert set(BACKBONE_ADAPTERS) == {
        "DLinearCoReG",
        "PatchMLPCoReG",
        "iTransformerCoReG",
        "xPatchCoReG",
        "MICNCoReG",
        "AmplifierCoReG",
    }


def test_public_adapters_keep_distinct_original_backbones():
    backbone_names = {cls.BACKBONE_CLASS.__name__ for cls in BACKBONE_ADAPTERS.values()}
    assert len(backbone_names) == 6


def test_temporal_decomposition_default_and_validation():
    import pytest
    from ts_benchmark.baselines.coreg_public import DLinearCoReG
    model = DLinearCoReG(seq_len=24, pred_len=8)
    assert model.config.memory_topk == 10
    assert model.config.coreg_moving_average_kernel == 25
    assert model.config.Geometry_multiscale_mode == "cross_variable_dual_path_reproduced"
    for kernel in (0, 2, -1):
        with pytest.raises(ValueError, match="positive and odd"):
            DLinearCoReG(seq_len=24, pred_len=8, coreg_moving_average_kernel=kernel)
