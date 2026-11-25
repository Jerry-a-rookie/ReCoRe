# ReCoRe

ReCoRe is a retrieval-augmented framework for multivariate time series forecasting, built on the TFB benchmark.

## Setup

Install PyTorch and the dependencies in `requirements.txt` and place the forecasting datasets under `dataset/forecasting/` following the TFB data format.

## Models

The public adapters are available from `ts_benchmark.baselines.coreg_public`: `DLinearCoReG`, `iTransformerCoReG`, `PatchMLPCoReG`, `xPatchCoReG`, `AmplifierCoReG`, and `MICNCoReG`.

Reliability-aware adjustment is enabled by default. Set `coreg_reliability_enabled=False` to disable it. To use trend--residual decomposition, set `coreg_moving_average_kernel` to a positive odd value. The default kernel is 0, which uses the existing multiscale path. Set the remaining hyperparameters according to the experiment configuration.

Run experiments using `scripts/run_benchmark.py`. The server configurations and checkpoints used for the paper are not included in this snapshot.

## Attribution

This project builds on [TFB](https://github.com/decisionintelligence/TFB); see `CITATION_TFB.bib`.

## License

ReCoRe contributions are available under the MIT License. Bundled third-party components are subject to their respective notices; see `THIRD_PARTY_NOTICES.md`.
