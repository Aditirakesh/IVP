"""Member 3 training package: participant-level splits, training loop, and runner."""

__all__ = [
    "EarlyStopping",
    "TrainConfig",
    "build_model",
    "kfold_splits",
    "load_model_class",
    "load_processed_dataset",
    "loso_splits",
    "make_val_split",
    "resolve_data_path",
    "resolve_device",
    "run_experiment",
    "select_channels",
    "set_seed",
    "train_fold",
]


def __getattr__(name: str):
    """Lazy module attribute loader to avoid eager import warnings during `python -m`."""
    if name in (
        "TrainConfig",
        "build_model",
        "load_model_class",
        "resolve_data_path",
        "resolve_device",
        "set_seed",
    ):
        from src.training import config

        return getattr(config, name)

    if name in ("kfold_splits", "loso_splits", "make_val_split"):
        from src.training import splits

        return getattr(splits, name)

    if name in ("EarlyStopping", "train_fold"):
        from src.training import trainer

        return getattr(trainer, name)

    if name in ("load_processed_dataset", "run_experiment", "select_channels"):
        from src.training import run_experiment

        return getattr(run_experiment, name)

    raise AttributeError(f"module 'src.training' has no attribute '{name}'")
