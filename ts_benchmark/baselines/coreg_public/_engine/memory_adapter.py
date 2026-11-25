from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset

from ts_benchmark.baselines.amplifier.models.amplifier_model import AmplifierModel
from ts_benchmark.baselines.coreg_public._engine.memory_model import DLinearMemoryModel
from ts_benchmark.baselines.coreg_public._engine.window_sampling import nested_window_indices
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
    MODEL_HYPER_PARAMS as DLINEAR_ADAPTER_HYPER_PARAMS,
)
from ts_benchmark.baselines.time_series_library.models.DLinear import DLinear
from ts_benchmark.baselines.time_series_library.models.PatchTST import PatchTST
from ts_benchmark.baselines.time_series_library.models.iTransformer import iTransformer
from ts_benchmark.baselines.time_series_library.models.MICN import MICN
from ts_benchmark.baselines.patchmlp.models.patchmlp_model import PatchMLPModel
from ts_benchmark.baselines.xpatch.models.xPatch import xPatchModel
from ts_benchmark.baselines.xpatch.utils.tools import (
    adjust_learning_rate as adjust_xpatch_learning_rate,
)
from ts_benchmark.baselines.utils import (
    EarlyStopping,
    adjust_learning_rate,
    train_val_split,
)
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.get_device import get_device


logger = logging.getLogger(__name__)


class _PredictionSpool:
    """Disk-backed float32 batches for branch diagnostics."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self.sample_shape: tuple[int, ...] | None = None

    def __bool__(self) -> bool:
        return self.count > 0

    def append(self, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values, dtype=np.float32)
        sample_shape = tuple(values.shape[1:])
        if self.sample_shape is None:
            self.sample_shape = sample_shape
        elif self.sample_shape != sample_shape:
            raise ValueError(
                f"Prediction shape changed from {self.sample_shape} to {sample_shape}."
            )
        
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as output:
            values.tofile(output)
        self.count += int(values.shape[0])

    def open(self) -> np.memmap:
        if not self or self.sample_shape is None:
            raise RuntimeError(f"No predictions were written to {self.path}.")
        return np.memmap(
            self.path,
            dtype=np.float32,
            mode="r",
            shape=(self.count, *self.sample_shape),
        )


class xPatchMemoryBackbone(xPatchModel):
    """Expose xPatch through the common forecasting-backbone signature."""

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(x_enc)


class MICNMemoryBackbone(MICN):
    """Expose MICN through the common forecasting-backbone signature."""

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_dec is None:
            x_dec = x_enc.new_zeros(
                x_enc.shape[0], int(self.pred_len), x_enc.shape[2]
            )
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)


MODEL_HYPER_PARAMS = {
    **DLINEAR_ADAPTER_HYPER_PARAMS,
    "label_len": None,
    "memory_embedding_dim": 16,
    "memory_encoder_hidden": 64,
    "memory_gate_hidden": 128,
    "memory_fusion_hidden": 128,
    "memory_topk": 50,
    "memory_gmb_size": 1000,
    "memory_pcl_epochs": 80,
    "memory_pcl_batch_size": 256,
    "memory_pcl_lr": 1e-3,
    "memory_pcl_temperature": 0.05,
    "memory_retrieval_temperature": 1.0,
    "memory_base_batch_size": None,
    "memory_train_batch_size": None,
    "memory_valid_batch_size": None,
    "memory_eval_batch_size": None,
    "memory_eval_runtime_batch_size": None,
    "memory_collect_branch_debug": True,
    "memory_debug_horizon_segments": 0,
    "memory_encode_batch_size": 1024,
    "memory_cluster_batch_size": 4096,
    "memory_cluster_iterations": 100,
    "memory_temporal_coverage_anchors": 0,
    "memory_memory_selection": "medoid",
    "memory_uniform_memory_min_channels": 64,
    "memory_seed": 2021,
    "memory_train_window_ratio": 1.0,
    "memory_cache_root": ".memory_cache",
    "memory_backbone_checkpoint_root": None,
    "memory_training_checkpoint_root": None,
    "memory_fixed_gamma_candidates": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "validation_shuffle": False,
}


def _smoke_max_batches() -> int | None:
    value = os.environ.get("TFB_SMOKE_MAX_BATCHES")
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


class _Config:
    def __init__(self, **kwargs):
        for key, value in MODEL_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if hasattr(self, "horizon") and not hasattr(self, "pred_len"):
            self.pred_len = self.horizon
        if hasattr(self, "pred_len") and not hasattr(self, "horizon"):
            self.horizon = self.pred_len
        if int(self.pred_len) != int(self.horizon):
            raise ValueError(
                f"Forecast semantic mismatch: pred_len={self.pred_len}, horizon={self.horizon}."
            )
        if self.label_len is None:
            self.label_len = int(self.seq_len) // 2
        for name in (
            "memory_base_batch_size",
            "memory_train_batch_size",
            "memory_eval_batch_size",
        ):
            if getattr(self, name) is None:
                setattr(self, name, int(self.batch_size))
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.memory_valid_batch_size is None:
            self.memory_valid_batch_size = int(self.memory_train_batch_size)
        if self.memory_eval_runtime_batch_size is None:
            self.memory_eval_runtime_batch_size = int(self.memory_eval_batch_size)
        for name in ("memory_valid_batch_size", "memory_eval_runtime_batch_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        if not 0.0 < float(self.memory_train_window_ratio) <= 1.0:
            raise ValueError("memory_train_window_ratio must be in (0, 1]")


class _IndexedWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        seq_len: int,
        pred_len: int,
        indices: np.ndarray | None = None,
    ):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        total_windows = len(self.values) - self.seq_len - self.pred_len + 1
        self.indices = (
            np.arange(max(total_windows, 0), dtype=np.int64)
            if indices is None
            else np.ascontiguousarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        index = int(self.indices[index])
        x = self.values[index : index + self.seq_len]
        y = self.values[
            index + self.seq_len : index + self.seq_len + self.pred_len
        ]
        return (
            torch.tensor(index, dtype=torch.long),
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
        )


class DLinearMemory(ModelBase):
    """TFB adapter for DLinear with a parallel multivariate Memory branch."""

    MODEL_NAME = "DLinearMemory"
    memory_VARIANT = "global"
    TRAINING_MODE = "joint"
    FUSION_MODE = "learned"
    BACKBONE_CLASS = DLinear
    MODEL_CLASS = DLinearMemoryModel
    BASE_DEBUG_NAME = "dlinear_only"

    def __init__(self, **kwargs):
        configured_variant = kwargs.get("memory_variant", self.memory_VARIANT)
        if configured_variant != self.memory_VARIANT:
            raise ValueError(
                f"{self.MODEL_NAME} requires memory_variant={self.memory_VARIANT!r}, "
                f"got {configured_variant!r}."
            )
        if self.memory_VARIANT == "same_channel":
            kwargs.setdefault("memory_memory_selection", "stratified_medoid")
        kwargs["memory_variant"] = self.memory_VARIANT
        kwargs["memory_training_mode"] = self.TRAINING_MODE
        kwargs["memory_fusion_mode"] = self.FUSION_MODE
        self.config = _Config(**kwargs)
        self.model: DLinearMemoryModel | nn.DataParallel | None = None
        self.check_point = None
        self.scaler: StandardScaler | None = None
        self._debug_scaler: StandardScaler | None = None
        self._debug_predictions: dict[str, _PredictionSpool] = {}
        self._debug_spool_root: Path | None = None
        self._debug_branch_metrics: dict[str, object] = {}
        self._backbone_checkpoint_used = False
        self._training_checkpoint_signature: str | None = None

    def _training_checkpoint_path(self, stage_name: str) -> Path | None:
        root = self.config.memory_training_checkpoint_root
        if not root or self._training_checkpoint_signature is None:
            return None
        safe_stage = stage_name.replace(os.sep, "_")
        return Path(root) / f"{self.MODEL_NAME}_{safe_stage}_{self._training_checkpoint_signature}.pt"

    def _flush_retrieval_cache(self) -> None:
        branch = self._unwrap_model().memory_branch
        flush = getattr(branch, "flush_selection_cache", None)
        if flush is not None:
            flush()

    def _release_retrieval_host_cache(self) -> None:
        branch = self._unwrap_model().memory_branch
        release = getattr(branch, "release_host_cache", None)
        if release is not None:
            release()

    def _save_training_checkpoint(
        self,
        stage_name: str,
        *,
        epoch: int,
        model_state: dict[str, torch.Tensor],
        best_state: dict[str, torch.Tensor],
        optimizer: optim.Optimizer | None,
        early_stopping: EarlyStopping | None,
        complete: bool,
    ) -> None:
        path = self._training_checkpoint_path(stage_name)
        if path is None:
            return
        self._flush_retrieval_cache()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "memory_training_checkpoint_v1",
            "signature": self._training_checkpoint_signature,
            "stage_name": stage_name,
            "epoch": int(epoch),
            "complete": bool(complete),
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in model_state.items()
            },
            "best_state": {
                name: tensor.detach().cpu() for name, tensor in best_state.items()
            },
        }
        if not complete and optimizer is not None and early_stopping is not None:
            payload["optimizer_state"] = optimizer.state_dict()
            payload["early_stopping"] = {
                "counter": early_stopping.counter,
                "best_score": early_stopping.best_score,
                "early_stop": early_stopping.early_stop,
                "val_loss_min": early_stopping.val_loss_min,
            }
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self._release_retrieval_host_cache()
        logger.info(
            "Saved recoverable Memory stage checkpoint (%s, epoch=%d): %s",
            "complete" if complete else "in_progress",
            epoch + 1,
            path,
        )

    def _load_training_checkpoint(self, stage_name: str):
        path = self._training_checkpoint_path(stage_name)
        if path is None or not path.is_file():
            return None
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if (
            payload.get("schema") != "memory_training_checkpoint_v1"
            or payload.get("signature") != self._training_checkpoint_signature
            or payload.get("stage_name") != stage_name
        ):
            return None
        logger.info(
            "Loaded recoverable Memory stage checkpoint (%s, epoch=%d): %s",
            "complete" if payload.get("complete") else "in_progress",
            int(payload.get("epoch", -1)) + 1,
            path,
        )
        return payload

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @staticmethod
    def required_hyper_params() -> dict:
        return {
            "seq_len": "input_chunk_length",
            "pred_len": "output_chunk_length",
        }

    def _unwrap_model(self) -> DLinearMemoryModel:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        if self.model is None:
            raise RuntimeError("DLinearMemory model is not initialized.")
        return self.model

    def _backbone_checkpoint_metadata(self, train_data: pd.DataFrame) -> dict:
        config = {}
        for name, value in vars(self.config).items():
            
            
            
            if (
                name.startswith("memory_")
                or name.startswith("Geometry_multiscale_")
                or name.startswith("coreg_")
            ) and name not in {
                "memory_base_batch_size",
                "memory_seed",
            }:
                continue
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
                config[name] = value
        values = np.ascontiguousarray(train_data.values, dtype=np.float32)
        data_digest = hashlib.sha256(values.view(np.uint8))
        ratio = float(self.config.memory_train_window_ratio)
        if ratio < 1.0:
            total_windows = (
                len(values)
                - int(self.config.seq_len)
                - int(self.config.pred_len)
                + 1
            )
            indices = nested_window_indices(
                total_windows, ratio, int(self.config.memory_seed)
            )
            data_digest.update(indices.view(np.uint8))
        return {
            "version": 1,
            "backbone_class": (
                f"{self.BACKBONE_CLASS.__module__}.{self.BACKBONE_CLASS.__qualname__}"
            ),
            "data_shape": list(values.shape),
            "data_hash": data_digest.hexdigest(),
            "config": config,
        }

    @staticmethod
    def _backbone_config_compatible(left: dict, right: dict) -> bool:
        """Compare only fields that affect frozen Backbone weights.

        Older checkpoints included retrieval-branch controls in their metadata.
        Those controls must not prevent Exact Geometry and a Geometry variant from
        sharing the same frozen Backbone.
        """
        ignored_prefixes = ("Geometry_multiscale_", "coreg_")
        ignored_names = {
            "memory_seed",
            
            "batch_size",
            "memory_base_batch_size",
            
            
            "Geometry_stream_rows",
        }

        
        
        
        
        legacy_optional = {"period_len", "station_lr"}

        def filtered(config: dict) -> dict:
            return {
                key: value
                for key, value in config.items()
                if not key.startswith(ignored_prefixes)
                and key not in ignored_names
            }

        left_filtered = filtered(left)
        right_filtered = filtered(right)
        for key in legacy_optional:
            if key not in left_filtered or key not in right_filtered:
                left_filtered.pop(key, None)
                right_filtered.pop(key, None)
        return left_filtered == right_filtered

    def _backbone_checkpoint_path(self, metadata: dict) -> Path | None:
        root = self.config.memory_backbone_checkpoint_root
        if not root:
            return None
        signature = hashlib.sha256(
            json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        backbone_name = self.BACKBONE_CLASS.__name__.replace("MemoryBackbone", "")
        return Path(root) / f"{backbone_name}_pl{self.config.pred_len}_{signature}.pt"

    def _load_backbone_checkpoint(self, train_data: pd.DataFrame) -> bool:
        metadata = self._backbone_checkpoint_metadata(train_data)
        checkpoint_path = self._backbone_checkpoint_path(metadata)
        if checkpoint_path is None:
            return False
        candidate_paths = []
        if checkpoint_path.exists():
            candidate_paths.append(checkpoint_path)
        backbone_name = self.BACKBONE_CLASS.__name__.replace("MemoryBackbone", "")
        candidate_paths.extend(
            sorted(
                checkpoint_path.parent.glob(
                    f"{backbone_name}_pl{self.config.pred_len}_*.pt"
                )
            )
        )
        
        
        
        candidate_paths.extend(
            sorted(
                checkpoint_path.parent.glob(
                    f"00_{backbone_name}_pl{self.config.pred_len}_*.pt"
                )
            )
        )
        payload = None
        mismatch_reasons = []
        for candidate in candidate_paths:
            loaded = torch.load(candidate, map_location="cpu", weights_only=True)
            loaded_metadata = loaded.get("metadata", {})
            checks = {
                "backbone_class": loaded_metadata.get("backbone_class") == metadata["backbone_class"],
                "data_shape": loaded_metadata.get("data_shape") == metadata["data_shape"],
                "data_hash": loaded_metadata.get("data_hash") == metadata["data_hash"],
                "config": self._backbone_config_compatible(loaded_metadata.get("config", {}), metadata["config"]),
            }
            if all(checks.values()):
                payload = loaded
                checkpoint_path = candidate
                break
            diff = {
                key: (loaded_metadata.get("config", {}).get(key), metadata["config"].get(key))
                for key in set(loaded_metadata.get("config", {})) | set(metadata["config"])
                if loaded_metadata.get("config", {}).get(key) != metadata["config"].get(key)
            }
            mismatch_reasons.append({"path": str(candidate), "failed": [key for key, ok in checks.items() if not ok], "config_diff": diff})
        if payload is None:
            logger.warning("No compatible Memory Backbone checkpoint; candidates=%s", mismatch_reasons)
            return False
        self._unwrap_model().backbone.load_state_dict(payload["backbone_state"])
        logger.info("Loaded Memory Backbone checkpoint: %s", checkpoint_path)
        return True

    def _save_backbone_checkpoint(self, train_data: pd.DataFrame) -> Path | None:
        metadata = self._backbone_checkpoint_metadata(train_data)
        checkpoint_path = self._backbone_checkpoint_path(metadata)
        if checkpoint_path is None:
            return None
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + f".{os.getpid()}.tmp"
        )
        torch.save(
            {
                "metadata": metadata,
                "backbone_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self._unwrap_model().backbone.state_dict().items()
                },
            },
            temp_path,
        )
        os.replace(temp_path, checkpoint_path)
        logger.info("Saved Memory Backbone checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def _update_data_dims(self, data: pd.DataFrame) -> None:
        freq = pd.infer_freq(data.index)
        if freq is None:
            raise ValueError("DLinearMemory requires regular time intervals.")
        channels = data.shape[1]
        self.config.enc_in = channels
        self.config.dec_in = channels
        self.config.c_out = channels

    def _build_loader(
        self,
        data: pd.DataFrame,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        apply_train_window_sampling: bool = False,
    ) -> DataLoader:
        total_windows = (
            len(data)
            - int(self.config.seq_len)
            - int(self.config.pred_len)
            + 1
        )
        indices = None
        if apply_train_window_sampling:
            indices = nested_window_indices(
                total_windows,
                float(self.config.memory_train_window_ratio),
                int(self.config.memory_seed),
            )
        dataset = _IndexedWindowDataset(
            data.values,
            int(self.config.seq_len),
            int(self.config.pred_len),
            indices=indices,
        )
        if len(dataset) <= 0:
            raise ValueError(
                f"Data length {len(data)} is insufficient for seq_len={self.config.seq_len} "
                f"and pred_len={self.config.pred_len}."
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.memory_seed))
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.num_workers),
            drop_last=drop_last,
            generator=generator,
        )

    def _fit_scaler(self, train_data: pd.DataFrame) -> pd.DataFrame:
        self._debug_scaler = StandardScaler().fit(train_data.values)
        if bool(self.config.norm):
            self.scaler = self._debug_scaler
            values = self.scaler.transform(train_data.values)
            return pd.DataFrame(values, index=train_data.index, columns=train_data.columns)
        self.scaler = None
        return train_data

    def _transform_df(self, data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if data is None or self.scaler is None:
            return data
        values = self.scaler.transform(data.values)
        return pd.DataFrame(values, index=data.index, columns=data.columns)

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return values
        shape = values.shape
        return self.scaler.inverse_transform(values.reshape(-1, shape[-1])).reshape(shape)

    def _criterion(self) -> nn.Module:
        if self.config.loss == "MAE":
            return nn.L1Loss()
        if self.config.loss == "MSE":
            return nn.MSELoss()
        return nn.HuberLoss(delta=0.5)

    def _compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        return criterion(prediction, target)

    def _adjust_stage_learning_rate(
        self,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> None:
        adjust_learning_rate(optimizer, epoch, self.config)

    def _set_trainable_modules(
        self,
        *,
        backbone: bool,
        retrieval: bool,
        fusion: bool,
    ) -> None:
        model = self._unwrap_model()
        for parameter in model.backbone.parameters():
            parameter.requires_grad = backbone
        encoder_parameters = [
            parameter
            for name, parameter in model.memory_branch.named_parameters()
            if name.startswith("encoder.") or ".encoder." in name
        ]
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        for parameter in model.memory_branch.parameters():
            if id(parameter) not in encoder_parameter_ids:
                parameter.requires_grad = retrieval
        for parameter in encoder_parameters:
            parameter.requires_grad = False
        for parameter in model.fusion.parameters():
            parameter.requires_grad = fusion

    def _set_stage_module_modes(self) -> None:
        self.model.train()
        model = self._unwrap_model()
        if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
            model.backbone.eval()
        for name, module in model.memory_branch.named_modules():
            if name == "encoder" or name.endswith(".encoder"):
                module.eval()

    def _forward_base_for_training(
        self,
        batch_x: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self._unwrap_model().backbone(batch_x, None, None, None)
        return prediction, prediction.new_zeros(())

    def _stage_output_and_loss(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        index: torch.Tensor,
        criterion: nn.Module,
        loss_mode: str,
        mode: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        model = self._unwrap_model()
        if loss_mode == "base":
            base_pred, additional_loss = self._forward_base_for_training(
                batch_x, mode
            )
            base_loss = self._compute_loss(base_pred, batch_y, criterion)
            loss = base_loss + additional_loss
            return loss, {
                "base": float(base_loss.detach().cpu().item()),
                "additional": float(additional_loss.detach().cpu().item()),
            }

        fused, base_pred, retrieval_pred = self.model(
            batch_x,
            index,
            mode=mode,
            return_branches=True,
        )
        fused_loss = self._compute_loss(fused, batch_y, criterion)
        base_loss = self._compute_loss(base_pred, batch_y, criterion)
        retrieval_loss = self._compute_loss(retrieval_pred, batch_y, criterion)
        if loss_mode == "joint":
            loss = fused_loss
        elif loss_mode == "retrieval":
            loss = retrieval_loss
        else:
            raise ValueError(f"Unknown Memory loss mode: {loss_mode}.")
        return loss, {
            "fused": float(fused_loss.detach().cpu().item()),
            "base": float(base_loss.detach().cpu().item()),
            "retrieval": float(retrieval_loss.detach().cpu().item()),
        }

    def _train_stage(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        criterion: nn.Module,
        device: torch.device,
        smoke_limit: int | None,
        *,
        stage_name: str,
        loss_mode: str,
    ) -> dict[str, torch.Tensor]:
        parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(f"Memory stage {stage_name} has no trainable parameters.")
        optimizer = optim.Adam(parameters, lr=float(self.config.lr))
        early_stopping = EarlyStopping(patience=int(self.config.patience))
        best_state: dict[str, torch.Tensor] | None = None
        start_epoch = 0
        last_epoch = -1
        recovered = self._load_training_checkpoint(stage_name)
        if recovered is not None:
            best_state = recovered["best_state"]
            if recovered.get("complete"):
                self.model.load_state_dict(best_state)
                return best_state
            self.model.load_state_dict(recovered["model_state"])
            optimizer.load_state_dict(recovered["optimizer_state"])
            for state in optimizer.state.values():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[name] = value.to(device)
            early_state = recovered["early_stopping"]
            for name, value in early_state.items():
                setattr(early_stopping, name, value)
            start_epoch = int(recovered["epoch"]) + 1

        for epoch in range(start_epoch, int(self.config.num_epochs)):
            last_epoch = epoch
            self._set_stage_module_modes()
            epoch_start = time.perf_counter()
            train_losses = []
            component_losses: dict[str, list[float]] = {}
            for batch_idx, (index, batch_x, batch_y) in enumerate(train_loader):
                if smoke_limit is not None and batch_idx >= smoke_limit:
                    break
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                loss, components = self._stage_output_and_loss(
                    batch_x, batch_y, index, criterion, loss_mode, "train"
                )
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                for name, value in components.items():
                    component_losses.setdefault(name, []).append(value)

            if not train_losses:
                raise RuntimeError(f"Memory stage {stage_name} processed no batches.")
            details = " ".join(
                f"{name}_loss={np.mean(values):.6f}"
                for name, values in component_losses.items()
            )
            logger.info(
                "Memory %s epoch %d/%d: train_loss=%.6f %s duration=%.2fs",
                stage_name,
                epoch + 1,
                self.config.num_epochs,
                float(np.mean(train_losses)),
                details,
                time.perf_counter() - epoch_start,
            )

            if valid_loader is not None:
                valid_loss = self._validate_stage(
                    valid_loader, criterion, device, smoke_limit, loss_mode
                )
                if early_stopping(valid_loss, self.model):
                    best_state = copy.deepcopy(self.model.state_dict())
                if early_stopping.early_stop:
                    break
            else:
                best_state = copy.deepcopy(self.model.state_dict())
            self._adjust_stage_learning_rate(optimizer, epoch + 1)
            if best_state is not None:
                self._save_training_checkpoint(
                    stage_name,
                    epoch=epoch,
                    model_state=self.model.state_dict(),
                    best_state=best_state,
                    optimizer=optimizer,
                    early_stopping=early_stopping,
                    complete=False,
                )

        if best_state is None:
            raise RuntimeError(f"Memory stage {stage_name} produced no checkpoint.")
        self.model.load_state_dict(best_state)
        self._save_training_checkpoint(
            stage_name,
            epoch=max(0, last_epoch),
            model_state=best_state,
            best_state=best_state,
            optimizer=None,
            early_stopping=None,
            complete=True,
        )
        return best_state

    def _validate_stage(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        smoke_limit: int | None,
        loss_mode: str,
    ) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (index, batch_x, batch_y) in enumerate(loader):
                if smoke_limit is not None and batch_idx >= smoke_limit:
                    break
                loss, _ = self._stage_output_and_loss(
                    batch_x.to(device),
                    batch_y.to(device),
                    index,
                    criterion,
                    loss_mode,
                    "valid",
                )
                losses.append(float(loss.cpu().item()))
        self._set_stage_module_modes()
        if not losses:
            raise RuntimeError("Memory validation processed no batches.")
        return float(np.mean(losses))

    def _select_fixed_gamma(
        self,
        valid_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        smoke_limit: int | None,
    ) -> float:
        candidates = [float(value) for value in self.config.memory_fixed_gamma_candidates]
        if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
            raise ValueError("memory_fixed_gamma_candidates must contain values in [0, 1].")
        squared_errors = {gamma: [] for gamma in candidates}
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (index, batch_x, batch_y) in enumerate(valid_loader):
                if smoke_limit is not None and batch_idx >= smoke_limit:
                    break
                _, base_pred, retrieval_pred = self.model(
                    batch_x.to(device),
                    index,
                    mode="valid",
                    return_branches=True,
                )
                batch_y = batch_y.to(device)
                for gamma in candidates:
                    prediction = base_pred + gamma * (retrieval_pred - base_pred)
                    squared_errors[gamma].append(
                        float(
                            self._compute_loss(prediction, batch_y, criterion)
                            .cpu()
                            .item()
                        )
                    )
        if not squared_errors[candidates[0]]:
            raise RuntimeError("Fixed-gamma selection processed no validation batches.")
        losses = {
            gamma: float(np.mean(values)) for gamma, values in squared_errors.items()
        }
        selected = min(candidates, key=lambda gamma: losses[gamma])
        logger.info("Memory validation gamma losses: %s; selected gamma=%.4f", losses, selected)
        return selected

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "DLinearMemory":
        if covariates and covariates.get("exog") is not None:
            raise ValueError("DLinearMemory currently accepts endogenous series only.")
        self._install_debug_spools()
        self._debug_branch_metrics = {}
        self._update_data_dims(train_valid_data)

        train_data, valid_data = train_val_split(
            train_valid_data,
            train_ratio_in_tv,
            int(self.config.seq_len),
        )
        train_data = self._fit_scaler(train_data)
        valid_data = self._transform_df(valid_data)
        signature_config = {
            name: value
            for name, value in vars(self.config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
            and name not in {
                "memory_training_checkpoint_root",
                
                
                
                "memory_valid_batch_size",
                "memory_eval_runtime_batch_size",
            }
        }
        if str(getattr(self.config, "Geometry_multiscale_mode", "")).startswith(
            "cross_variable_dual_path_reproduced"
        ):
            
            
            signature_config["relational_memory_key_format"] = 1
        signature_payload = {
            "model": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "data_sha256": hashlib.sha256(
                np.ascontiguousarray(train_data.values, dtype=np.float32).view(np.uint8)
            ).hexdigest(),
            "config": signature_config,
        }
        self._training_checkpoint_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

        self.model = self.MODEL_CLASS(
            self.config, backbone_class=self.BACKBONE_CLASS
        ).float()
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        device = get_device()
        self._unwrap_model().prepare_memory(train_data.values.astype(np.float32), device)
        if os.environ.get("memory_CACHE_WARMUP_ONLY") == "1":
            logger.info("Memory cache warm-up complete; cache-only run exiting.")
            raise SystemExit(0)
        self.model.to(device)

        base_train_loader = self._build_loader(
            train_data,
            batch_size=int(self.config.memory_base_batch_size),
            shuffle=True,
            drop_last=train_data.shape[1] > 1,
            apply_train_window_sampling=True,
        )
        base_valid_loader = (
            self._build_loader(
                valid_data,
                batch_size=int(self.config.memory_valid_batch_size),
                shuffle=bool(self.config.validation_shuffle),
                drop_last=False,
            )
            if valid_data is not None
            else None
        )
        memory_train_loader = self._build_loader(
            train_data,
            batch_size=int(self.config.memory_train_batch_size),
            shuffle=True,
            drop_last=train_data.shape[1] > 1,
            apply_train_window_sampling=True,
        )
        memory_valid_loader = (
            self._build_loader(
                valid_data,
                batch_size=int(self.config.memory_valid_batch_size),
                shuffle=bool(self.config.validation_shuffle),
                drop_last=False,
            )
            if valid_data is not None
            else None
        )
        criterion = self._criterion()
        smoke_limit = _smoke_max_batches()

        if self.TRAINING_MODE in {"joint"}:
            self._set_trainable_modules(backbone=True, retrieval=True, fusion=True)
            self.check_point = self._train_stage(
                memory_train_loader,
                memory_valid_loader,
                criterion,
                device,
                smoke_limit,
                stage_name=self.TRAINING_MODE,
                loss_mode=self.TRAINING_MODE,
            )
        elif self.TRAINING_MODE in {"frozen_base_fusion", "frozen_base_fixed"}:
            if base_valid_loader is None or memory_valid_loader is None:
                raise ValueError(
                    f"{self.MODEL_NAME} requires validation data for staged training."
                )
            backbone_loaded = self._load_backbone_checkpoint(train_data)
            self._backbone_checkpoint_used = backbone_loaded
            if not backbone_loaded:
                if bool(
                    getattr(
                        self.config,
                        "memory_require_existing_backbone_checkpoint",
                        False,
                    )
                ):
                    raise FileNotFoundError(
                        "Required Memory Backbone checkpoint was not found; "
                        "Backbone pretraining is disabled."
                    )
                self._set_trainable_modules(backbone=True, retrieval=False, fusion=False)
                self._train_stage(
                    base_train_loader,
                    base_valid_loader,
                    criterion,
                    device,
                    smoke_limit,
                    stage_name="base_pretrain",
                    loss_mode="base",
                )
                self._save_backbone_checkpoint(train_data)
            if os.environ.get("memory_PRETRAIN_BACKBONE_ONLY") == "1":
                logger.info("Memory Backbone checkpoint ready; pretrain-only run complete.")
                raise SystemExit(0)
            train_fusion = self.TRAINING_MODE == "frozen_base_fusion"
            self._set_trainable_modules(
                backbone=False,
                retrieval=True,
                fusion=train_fusion,
            )
            self.check_point = self._train_stage(
                memory_train_loader,
                memory_valid_loader,
                criterion,
                device,
                smoke_limit,
                stage_name=("frozen_base_fusion" if train_fusion else "retrieval_only"),
                loss_mode=("joint" if train_fusion else "retrieval"),
            )
            if self.TRAINING_MODE == "frozen_base_fixed":
                selected_gamma = self._select_fixed_gamma(
                    memory_valid_loader, criterion, device, smoke_limit
                )
                self._unwrap_model().set_fixed_gamma(selected_gamma)
                self.check_point = copy.deepcopy(self.model.state_dict())
        else:
            raise ValueError(f"Unknown Memory training mode: {self.TRAINING_MODE}.")

        return self

    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        smoke_limit: int | None,
    ) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (index, batch_x, batch_y) in enumerate(loader):
                if smoke_limit is not None and batch_idx >= smoke_limit:
                    break
                output = self.model(batch_x.to(device), index, mode="valid")
                losses.append(
                    float(
                        self._compute_loss(output, batch_y.to(device), criterion)
                        .cpu()
                        .item()
                    )
                )
        self.model.train()
        if not losses:
            raise RuntimeError("DLinearMemory validation processed no batches.")
        return float(np.mean(losses))

    def _predict_batch(self, input_values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call forecast_fit before forecasting.")
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point)
        device = get_device()
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            batch_x = torch.tensor(input_values, dtype=torch.float32, device=device)
            output, base_pred, retrieval_pred = self.model(
                batch_x,
                index=(torch.as_tensor(getattr(self, "_last_query_starts", None), dtype=torch.long, device=device)
                       if getattr(self, "_last_query_starts", None) is not None else None),
                mode="test",
                return_branches=True,
            )
        self._release_retrieval_host_cache()
        if not self.config.memory_collect_branch_debug:
            return self._inverse(output.detach().cpu().numpy())
        branch_outputs = {
            self.BASE_DEBUG_NAME: base_pred,
            "retrieval_only": retrieval_pred,
            "fused": output,
        }
        inversed = {}
        for name, prediction in branch_outputs.items():
            values = self._inverse(prediction.detach().cpu().numpy())
            self._debug_predictions[name].append(values)
            inversed[name] = values
        return inversed["fused"]

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        if covariates and covariates.get("exog") is not None:
            raise ValueError("DLinearMemory currently accepts endogenous series only.")
        if int(horizon) != int(self.config.pred_len):
            raise ValueError(
                f"DLinearMemory requires horizon={self.config.pred_len}, got {horizon}."
            )
        transformed = self._transform_df(series)
        input_values = transformed.iloc[-self.config.seq_len :].values.astype(np.float32)
        return self._predict_batch(input_values[None, ...])[0]

    def batch_forecast(self, horizon: int, batch_maker: BatchMaker, **kwargs) -> np.ndarray:
        if int(horizon) != int(self.config.pred_len):
            raise ValueError(
                f"DLinearMemory requires horizon={self.config.pred_len}, got {horizon}."
            )
        input_data = batch_maker.make_batch(
            int(self.config.memory_eval_runtime_batch_size), self.config.seq_len
        )
        if (input_data.get("covariates") or {}).get("exog") is not None:
            raise ValueError("DLinearMemory currently accepts endogenous series only.")
        input_values = input_data["input"].astype(np.float32)
        self._last_query_starts = input_data.get("window_starts")
        if self.scaler is not None:
            shape = input_values.shape
            input_values = self.scaler.transform(
                input_values.reshape(-1, shape[-1])
            ).reshape(shape)
        return self._predict_batch(input_values)

    def reset_debug_summary(self) -> None:
        self._install_debug_spools()
        self._debug_branch_metrics = {}
        if self.model is not None:
            self._unwrap_model().reset_debug_summary()

    def _cleanup_debug_spools(self) -> None:
        if self._debug_spool_root is not None:
            shutil.rmtree(self._debug_spool_root, ignore_errors=True)
        self._debug_spool_root = None

    def _install_debug_spools(self) -> None:
        self._cleanup_debug_spools()
        if not self.config.memory_collect_branch_debug:
            self._debug_predictions = {}
            return
        configured_root = getattr(self.config, "coreg_spool_root", None)
        root = (Path(configured_root) / "memory_debug" if configured_root
                else Path(tempfile.mkdtemp(prefix="tfb_memory_debug_")))
        root.mkdir(parents=True, exist_ok=True)
        self._debug_spool_root = root
        self._debug_predictions = {
            name: _PredictionSpool(root / f"{name}.float32")
            for name in (self.BASE_DEBUG_NAME, "retrieval_only", "fused")
        }

    def update_debug_targets(
        self,
        targets: np.ndarray,
        *,
        window_starts: list[int] | np.ndarray | None = None,
    ) -> None:
        if not self.config.memory_collect_branch_debug:
            return
        targets = np.asarray(targets)
        metrics: dict[str, object] = {}
        try:
            for name, spool in self._debug_predictions.items():
                if not spool or spool.sample_shape is None:
                    raise RuntimeError(f"No cached {name} predictions for branch diagnostics.")
                prediction_shape = (spool.count, *spool.sample_shape)
                if prediction_shape != targets.shape:
                    raise RuntimeError(
                        f"{name} prediction shape {prediction_shape} != target shape "
                        f"{targets.shape}."
                    )
                self._accumulate_debug_metrics(name, spool, targets, metrics)
            self._debug_branch_metrics = metrics
            self._persist_window_debug_artifact(targets, window_starts)
        finally:
            self._cleanup_debug_spools()

    def _persist_window_debug_artifact(
        self,
        targets: np.ndarray,
        window_starts: list[int] | np.ndarray | None,
    ) -> None:
        """Persist the complete rolling-window branch predictions for interpretation.

        The branch spools are already float32 and disk-backed.  Copying them
        into the run-owned artifact directory preserves every evaluated window
        without materializing all branch predictions in Python memory.
        """
        configured_root = getattr(self.config, "coreg_window_debug_root", None)
        if not configured_root:
            return
        root = Path(configured_root)
        root.mkdir(parents=True, exist_ok=True)
        starts = np.asarray(
            window_starts if window_starts is not None else np.arange(len(targets)),
            dtype=np.int64,
        )
        if starts.shape != (len(targets),):
            raise ValueError(
                f"window_starts shape {starts.shape} does not match targets {targets.shape}."
            )
        np.save(root / "targets.npy", np.ascontiguousarray(targets, dtype=np.float32))
        np.save(root / "window_starts.npy", starts)
        branches = {}
        for name, spool in self._debug_predictions.items():
            destination = root / f"{name}.float32"
            shutil.copyfile(spool.path, destination)
            branches[name] = {
                "path": destination.name,
                "count": int(spool.count),
                "sample_shape": list(spool.sample_shape or ()),
                "dtype": "float32",
            }
        metadata = {
            "schema": "rolegeo_window_debug_v1",
            "window_count": int(len(targets)),
            "target_shape": list(targets.shape),
            "window_starts": "window_starts.npy",
            "targets": "targets.npy",
            "branches": branches,
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _accumulate_debug_metrics(
        self,
        name: str,
        spool: _PredictionSpool,
        targets: np.ndarray,
        metrics: dict[str, object],
    ) -> None:
            prediction_shape = (spool.count, *spool.sample_shape)
            if prediction_shape != targets.shape:
                raise RuntimeError(
                    f"{name} prediction shape {prediction_shape} != target shape "
                    f"{targets.shape}."
                )
            segment_count = int(self.config.memory_debug_horizon_segments)
            if segment_count < 0:
                raise ValueError("memory_debug_horizon_segments must be non-negative.")
            if segment_count > prediction_shape[1]:
                raise ValueError(
                    "memory_debug_horizon_segments cannot exceed the prediction length."
                )
            segments = [
                (
                    int(index * prediction_shape[1] // segment_count),
                    int((index + 1) * prediction_shape[1] // segment_count),
                )
                for index in range(segment_count)
            ]
            channels = int(prediction_shape[-1])
            squared_sum = absolute_sum = 0.0
            normalized_squared_sum = normalized_absolute_sum = 0.0
            squared_by_channel = np.zeros(channels, dtype=np.float64)
            absolute_by_channel = np.zeros(channels, dtype=np.float64)
            normalized_squared_by_channel = np.zeros(channels, dtype=np.float64)
            segment_sums = np.zeros((segment_count, 4), dtype=np.float64)
            predictions = spool.open()
            chunk_size = 8
            for start in range(0, prediction_shape[0], chunk_size):
                stop = min(start + chunk_size, prediction_shape[0])
                prediction = np.asarray(predictions[start:stop], dtype=np.float64)
                target = np.asarray(targets[start:stop], dtype=np.float64)
                error = prediction - target
                squared = np.square(error)
                absolute = np.abs(error)
                squared_sum += float(np.sum(squared, dtype=np.float64))
                absolute_sum += float(np.sum(absolute, dtype=np.float64))
                squared_by_channel += np.sum(squared, axis=(0, 1), dtype=np.float64)
                absolute_by_channel += np.sum(absolute, axis=(0, 1), dtype=np.float64)

                if self._debug_scaler is not None:
                    normalized_prediction = self._debug_scaler.transform(
                        prediction.reshape(-1, channels)
                    ).reshape(prediction.shape)
                    normalized_target = self._debug_scaler.transform(
                        target.reshape(-1, channels)
                    ).reshape(target.shape)
                    normalized_error = normalized_prediction - normalized_target
                else:
                    normalized_error = error
                normalized_squared = np.square(normalized_error)
                normalized_absolute = np.abs(normalized_error)
                normalized_squared_sum += float(
                    np.sum(normalized_squared, dtype=np.float64)
                )
                normalized_absolute_sum += float(
                    np.sum(normalized_absolute, dtype=np.float64)
                )
                normalized_squared_by_channel += np.sum(
                    normalized_squared, axis=(0, 1), dtype=np.float64
                )
                for index, (start, end) in enumerate(segments):
                    segment_sums[index] += (
                        np.sum(squared[:, start:end], dtype=np.float64),
                        np.sum(absolute[:, start:end], dtype=np.float64),
                        np.sum(normalized_squared[:, start:end], dtype=np.float64),
                        np.sum(normalized_absolute[:, start:end], dtype=np.float64),
                    )

            element_count = int(np.prod(prediction_shape))
            channel_count = int(prediction_shape[0] * prediction_shape[1])
            metrics[f"{name}_mse"] = squared_sum / element_count
            metrics[f"{name}_mae"] = absolute_sum / element_count
            metrics[f"{name}_mse_by_channel"] = (
                squared_by_channel / channel_count
            ).tolist()
            metrics[f"{name}_mae_by_channel"] = (
                absolute_by_channel / channel_count
            ).tolist()
            metrics[f"{name}_mse_norm"] = normalized_squared_sum / element_count
            metrics[f"{name}_mae_norm"] = normalized_absolute_sum / element_count
            metrics[f"{name}_mse_norm_by_channel"] = (
                normalized_squared_by_channel / channel_count
            ).tolist()
            if segment_count:
                metrics[f"{name}_horizon_segments"] = [
                    {
                        "start": start,
                        "end": end,
                        "mse": float(segment_sums[index, 0] / count),
                        "mae": float(segment_sums[index, 1] / count),
                        "mse_norm": float(segment_sums[index, 2] / count),
                        "mae_norm": float(segment_sums[index, 3] / count),
                    }
                    for index, (start, end) in enumerate(segments)
                    for count in [
                        int(prediction_shape[0] * (end - start) * channels)
                    ]
                ]
            del predictions

    def get_debug_summary(self) -> dict:
        if self.model is None:
            return {}
        summary = self._unwrap_model().get_debug_summary()
        summary["memory_training_mode"] = self.TRAINING_MODE
        summary["memory_backbone_checkpoint_used"] = bool(
            self._backbone_checkpoint_used
        )
        summary["memory_base_batch_size"] = int(self.config.memory_base_batch_size)
        summary["memory_train_batch_size"] = int(self.config.memory_train_batch_size)
        summary["memory_valid_batch_size"] = int(self.config.memory_valid_batch_size)
        summary["memory_eval_batch_size_configured"] = int(
            self.config.memory_eval_batch_size
        )
        summary["memory_eval_batch_size"] = int(
            self.config.memory_eval_runtime_batch_size
        )
        summary["memory_train_window_ratio"] = float(
            self.config.memory_train_window_ratio
        )
        summary["memory_collect_branch_debug"] = bool(
            self.config.memory_collect_branch_debug
        )
        summary.update(self._debug_branch_metrics)
        return summary


class DLinearSameChannelMemory(DLinearMemory):
    """DLinear with shared Memory representation and same-channel retrieval."""

    MODEL_NAME = "DLinearSameChannelMemory"
    memory_VARIANT = "same_channel"


class DLinearIndependentMemory(DLinearMemory):
    """DLinear with one independent Memory branch per target channel."""

    MODEL_NAME = "DLinearIndependentMemory"
    memory_VARIANT = "independent"


class DLinearSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen DLinear with learned Memory fusion."""

    MODEL_NAME = "DLinearSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"


class DLinearSameChannelMemoryFixedGamma(DLinearMemory):
    """Pretrained frozen branches with validation-selected residual fusion."""

    MODEL_NAME = "DLinearSameChannelMemoryFixedGamma"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"


class PatchTSTSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen PatchTST with same-channel learned Memory fusion."""

    MODEL_NAME = "PatchTSTSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = PatchTST
    BASE_DEBUG_NAME = "patchtst_only"


class PatchTSTSameChannelMemoryFixedGamma(PatchTSTSameChannelMemoryFrozenBase):
    """Pretrained frozen PatchTST with validation-selected Memory fusion."""

    MODEL_NAME = "PatchTSTSameChannelMemoryFixedGamma"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"


class PatchMLPSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen PatchMLP with same-channel learned Memory fusion."""

    MODEL_NAME = "PatchMLPSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = PatchMLPModel
    BASE_DEBUG_NAME = "patchmlp_only"


class PatchMLPSameChannelMemoryFixedGamma(PatchMLPSameChannelMemoryFrozenBase):
    """Pretrained frozen PatchMLP with validation-selected Memory fusion."""

    MODEL_NAME = "PatchMLPSameChannelMemoryFixedGamma"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"


class AmplifierSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen Amplifier with same-channel learned Memory fusion."""

    MODEL_NAME = "AmplifierSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = AmplifierModel
    BASE_DEBUG_NAME = "amplifier_only"


class AmplifierSameChannelMemoryFixedGamma(AmplifierSameChannelMemoryFrozenBase):
    """Pretrained frozen Amplifier with validation-selected Memory fusion."""

    MODEL_NAME = "AmplifierSameChannelMemoryFixedGamma"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"


class MICNSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen MICN with same-channel learned Memory fusion."""

    MODEL_NAME = "MICNSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = MICNMemoryBackbone
    BASE_DEBUG_NAME = "micn_only"


class xPatchSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen xPatch with same-channel learned Memory fusion."""

    MODEL_NAME = "xPatchSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = xPatchMemoryBackbone
    BASE_DEBUG_NAME = "xpatch_only"

    def __init__(self, **kwargs):
        xpatch_defaults = {
            "patch_len": 16,
            "stride": 8,
            "padding_patch": "end",
            "ma_type": "ema",
            "alpha": 0.3,
            "beta": 0.3,
            "loss": "MAE",
            "lradj": "type1",
            "revin": 1,
        }
        for key, value in xpatch_defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

    def _compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        steps = torch.arange(
            1,
            int(self.config.pred_len) + 1,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        ratio = -torch.atan(steps) + math.pi / 4 + 1
        ratio = ratio.view(1, -1, 1)
        return criterion(prediction * ratio, target * ratio)

    def _adjust_stage_learning_rate(
        self,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> None:
        adjust_xpatch_learning_rate(optimizer, epoch, self.config)


class xPatchSameChannelMemoryFixedGamma(xPatchSameChannelMemoryFrozenBase):
    """Pretrained frozen xPatch with validation-selected Memory fusion."""

    MODEL_NAME = "xPatchSameChannelMemoryFixedGamma"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"


class iTransformerSameChannelMemoryFrozenBase(DLinearMemory):
    """Pretrained frozen iTransformer with learned Memory fusion."""

    MODEL_NAME = "iTransformerSameChannelMemoryFrozenBase"
    memory_VARIANT = "same_channel"
    TRAINING_MODE = "frozen_base_fusion"
    BACKBONE_CLASS = iTransformer
    BASE_DEBUG_NAME = "itransformer_only"


class iTransformerSameChannelMemoryFixedGamma(
    iTransformerSameChannelMemoryFrozenBase
):
    """Pretrained frozen iTransformer with validation-selected Memory fusion."""

    MODEL_NAME = "iTransformerSameChannelMemoryFixedGamma"
    TRAINING_MODE = "frozen_base_fixed"
    FUSION_MODE = "fixed"



