# -*- coding: utf-8 -*-
import os


ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")


CONFIG_PATH = os.path.join(ROOT_PATH, "config")


THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
