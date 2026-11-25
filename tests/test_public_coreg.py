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
