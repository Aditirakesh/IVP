"""Training configuration, deterministic seeding, and model/data resolution."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
from torch import nn


DEFAULT_CONFIG_PATH = Path("config/train_config.json")
DATA_CANDIDATES = (
    Path("data/processed/modma_epochs.npz"),
    Path("src/data/processed/modma_epochs.npz"),
)


@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for CDL-MDDNet training."""

    experiment: str = "cdl_mddnet"
    model: str = "src.models.cdl_mddnet:CDLMDDNet"
    protocol: str = "kfold10"
    data_path: Optional[str] = None
    runs_dir: str = "runs"
    n_folds: int = 10
    val_fraction: float = 0.15
    seed: int = 42
    split_seed: int = 0
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    class_weighted_loss: bool = True
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = True
    deterministic: bool = True
    device: str = "auto"
    num_workers: int = 0
    n_classes: int = 2

    def validate(self) -> None:
        if not isinstance(self.experiment, str) or not self.experiment.strip():
            raise ValueError(f"experiment must be a non-empty string, got {self.experiment!r}.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"model must be a non-empty string, got {self.model!r}.")
        if self.protocol not in ("kfold10", "loso"):
            raise ValueError(f"Unsupported protocol '{self.protocol}'; expected 'kfold10' or 'loso'.")
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int) or self.n_folds < 2:
            raise ValueError(f"n_folds must be an integer >= 2, got {self.n_folds!r}.")
        if (
            isinstance(self.val_fraction, bool)
            or not isinstance(self.val_fraction, (int, float))
            or not math.isfinite(self.val_fraction)
            or not (0.0 < float(self.val_fraction) < 0.5)
        ):
            raise ValueError(f"val_fraction must be a finite float in (0, 0.5), got {self.val_fraction!r}.")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be an integer >= 1, got {self.epochs!r}.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {self.batch_size!r}.")
        if (
            isinstance(self.lr, bool)
            or not isinstance(self.lr, (int, float))
            or not math.isfinite(self.lr)
            or self.lr <= 0
        ):
            raise ValueError(f"lr must be a finite float > 0, got {self.lr!r}.")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError(f"weight_decay must be a finite float >= 0, got {self.weight_decay!r}.")
        if (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 1
        ):
            raise ValueError(
                f"early_stopping_patience must be an integer >= 1, got {self.early_stopping_patience!r}."
            )
        if (
            isinstance(self.early_stopping_min_delta, bool)
            or not isinstance(self.early_stopping_min_delta, (int, float))
            or not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError(
                f"early_stopping_min_delta must be a finite float >= 0, got {self.early_stopping_min_delta!r}."
            )
        if self.grad_clip_norm is not None and (
            isinstance(self.grad_clip_norm, bool)
            or not isinstance(self.grad_clip_norm, (int, float))
            or not math.isfinite(self.grad_clip_norm)
            or self.grad_clip_norm <= 0
        ):
            raise ValueError(
                f"grad_clip_norm must be None or a finite float > 0, got {self.grad_clip_norm!r}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {self.seed!r}.")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int) or self.split_seed < 0:
            raise ValueError(f"split_seed must be an integer >= 0, got {self.split_seed!r}.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(f"num_workers must be an integer >= 0, got {self.num_workers!r}.")
        if isinstance(self.n_classes, bool) or self.n_classes != 2:
            raise ValueError(
                f"Binary trainer requires n_classes == 2 (0=Healthy, 1=MDD), got {self.n_classes!r}."
            )
        if not isinstance(self.optimizer, str) or self.optimizer.lower() not in ("adamw", "adam"):
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'; expected 'adamw' or 'adam'.")
        if not isinstance(self.scheduler, str) or self.scheduler.lower() not in ("cosine", "plateau", "none"):
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'; expected 'cosine', 'plateau', or 'none'."
            )
        if not isinstance(self.device, str):
            raise ValueError(f"device must be a string, got {self.device!r}.")
        dev = self.device.lower()
        valid_cuda = dev == "cuda" or (dev.startswith("cuda:") and dev[5:].isdigit())
        if dev not in ("auto", "cpu") and not valid_cuda:
            raise ValueError(
                f"Unsupported device '{self.device}'; expected 'auto', 'cpu', 'cuda', or 'cuda:<idx>'."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainConfig":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        content = p.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Config file '{p}' is empty; expected a JSON object.")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file '{p}': {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file '{p}' must contain a JSON object, got {type(raw).__name__}."
            )
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw.keys()) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown config keys in {p}: {unknown}")
        try:
            cfg = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"Invalid configuration field types in '{p}': {exc}") from exc
        cfg.validate()
        return cfg

    @classmethod
    def load_default(cls, path: Optional[Union[str, Path]] = None) -> "TrainConfig":
        if path is not None:
            return cls.from_json(path)
        if DEFAULT_CONFIG_PATH.is_file():
            return cls.from_json(DEFAULT_CONFIG_PATH)
        cfg = cls()
        cfg.validate()
        return cfg


def set_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Set fixed seeds across Python, NumPy, and PyTorch (CPU + CUDA).

    Sets CUBLAS_WORKSPACE_CONFIG before CUDA context initialization whenever
    possible. Returns a status dictionary noting any limitations (for example,
    CUDA already initialized prior to calling `set_seed`, or PYTHONHASHSEED not
    set prior to interpreter launch) rather than assuming full determinism.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}.")

    cuda_initialized_before = bool(torch.cuda.is_initialized())
    limitations = []

    if deterministic:
        # Must be set before CUDA initialization to take effect for cuBLAS GEMM ops.
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if cuda_initialized_before:
            limitations.append(
                "CUDA was already initialized before set_seed; CUBLAS_WORKSPACE_CONFIG "
                "or cuDNN settings may not apply retroactively."
            )

    if "PYTHONHASHSEED" not in os.environ:
        limitations.append(
            "PYTHONHASHSEED was not set before Python startup; set it in the shell "
            "before launching Python if hash order matters."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            limitations.append(
                "torch.use_deterministic_algorithms uses warn_only=True; some GPU/reduction "
                "ops may still run non-deterministically depending on driver/kernel support."
            )
        except Exception as exc:
            limitations.append(f"Could not enable torch.use_deterministic_algorithms: {exc}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return {
        "seed": int(seed),
        "deterministic_requested": bool(deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_initialized_before": cuda_initialized_before,
        "strict_determinism_guaranteed": False if not deterministic or limitations else True,
        "limitations": limitations,
    }


def resolve_device(device_pref: str = "auto") -> torch.device:
    """Resolve torch.device from preference ('auto', 'cuda', or 'cpu')."""
    pref = device_pref.lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    return torch.device(pref)


def resolve_data_path(data_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the processed MODMA epochs NPZ path across documented locations."""
    if data_path is not None:
        p = Path(data_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Processed EEG dataset not found at '{p}'. "
                f"Run Member 2's preprocessing pipeline or pass the correct --data path."
            )
        return p

    for candidate in DATA_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(c) for c in DATA_CANDIDATES)
    raise FileNotFoundError(
        f"Processed EEG dataset 'modma_epochs.npz' not found (checked: {searched}). "
        f"Pass --data /path/to/modma_epochs.npz or run Member 2's preprocessing first."
    )


def load_model_class(model_spec: str) -> Type[nn.Module]:
    """Load a model class from 'dummy' or 'module.path:ClassName'."""
    if model_spec == "dummy":
        from src.eval.benchmark import Dummy

        return Dummy

    if ":" not in model_spec:
        raise ValueError(
            f"Invalid model specification '{model_spec}'. "
            f"Expected 'dummy' or 'module.path:ClassName' (e.g. 'src.models.cdl_mddnet:CDLMDDNet')."
        )

    mod_name, cls_name = model_spec.split(":", 1)
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import model module '{mod_name}' for '{model_spec}'. "
            f"Ensure Member 1/6's model file exists, or use '--model dummy' for pipeline testing."
        ) from exc

    if not hasattr(module, cls_name):
        raise AttributeError(f"Module '{mod_name}' does not define '{cls_name}'.")
    return getattr(module, cls_name)


def build_model(
    model_spec: str,
    n_channels: int,
    n_samples: int = 1000,
    n_classes: int = 2,
) -> nn.Module:
    """Instantiate a model conforming to the CDL-MDDNet interface contract.

    Expected interface:
    - Constructor: `ModelClass(n_channels, n_samples=1000, n_classes=2)`
    - Input tensor: `(batch_size, n_channels, n_samples)` (e.g. `(B, 128, 1000)`)
    - Output logits: `(batch_size, 2)` for labels `(batch_size,)` (`0=Healthy, 1=MDD`)
    """
    cls = load_model_class(model_spec)
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs: Dict[str, Any] = {
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
    }
    if "n_classes" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        kwargs["n_classes"] = int(n_classes)
    return cls(**kwargs)

