# ReCoRe

ReCoRe is a retrieval-augmented framework for multivariate time series forecasting, built on the TFB benchmark. It retrieves trend and residual references separately, uses training-period relations for channel retrieval, and adjusts each retrieval correction using candidate-future disagreement.

## Code and supplementary material

The implementation is in `ts_benchmark/baselines/recore_public/`. The available adapters are `DLinearReCoRe`, `iTransformerReCoRe`, `PatchMLPReCoRe`, `xPatchReCoRe`, `AmplifierReCoRe`, and `MICNReCoRe`.

The [supplementary material](supplement/supplement.pdf) describes the method, datasets, reported experimental settings, additional results, and selected implementation excerpts.

## Setup and experiments

Install PyTorch and the dependencies in `requirements.txt`. Obtain the forecasting datasets through [TFB](https://github.com/decisionintelligence/TFB) and place them under `dataset/forecasting/` using the TFB data format. `scripts/run_benchmark.py` provides the benchmark runner and accepts `--config-path`, `--model-name`, `--model-hyper-params`, `--strategy-args`, and `--seed`; run it with `--help` for the complete interface. The `config/` directory contains TFB evaluation configurations.

The paper reports an input length of 720, 10 retrieved candidates, a memory capacity of 1,000 entries per variable, moving-average window 25, reliability strength 1, and five seeds (2021--2025). See the supplement for the remaining reported training and hardware settings. The public adapters enable reliability adjustment and use a moving-average window of 25 by default.

The paper's server configurations, exact experiment commands, and trained checkpoints are not included in this snapshot. The supplied benchmark configurations describe the available runner; they should not be treated as records of every reported experiment.

## Attribution and license

This project builds on [TFB](https://github.com/decisionintelligence/TFB); see `CITATION_TFB.bib`. ReCoRe contributions are available under the MIT License. Bundled third-party components are subject to their respective notices; see `THIRD_PARTY_NOTICES.md`.
