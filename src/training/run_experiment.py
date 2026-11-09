"""Experiment runner for local CPU and Google Colab T4 GPU.

Examples
--------
10-fold cross-validation with CDL-MDDNet:
    python -m src.training.run_experiment --model src.models.cdl_mddnet:CDLMDDNet --protocol kfold10 --experiment cdl_mddnet

53-fold LOSO on Google Colab T4 with Google Drive persistence and resume support:
    python -m src.training.run_experiment --model src.models.cdl_mddnet:CDLMDDNet --protocol loso --experiment cdl_mddnet_loso --data /content/drive/MyDrive/IVP/modma_epochs.npz --runs-dir /content/drive/MyDrive/IVP/runs --resume

Reduced montage from Member 5's channel_ranking.csv:
    python -m src.training.run_experiment --model src.models.cdl_mddnet:CDLMDDNet --channel-ranking channel_ranking.csv --top-k-channels 32 --experiment cdl_mddnet_ch32

Pipeline smoke test before Member 1's model is ready:
    python -m src.training.run_experiment --model dummy --protocol kfold10 --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.training.config import (
    TrainConfig,
    build_model,
    resolve_data_path,
    resolve_device,
    set_seed,
)
from src.training.splits import kfold_splits, loso_splits
from src.training.trainer import train_fold


PARAM_LIMIT = 100_000


def load_processed_dataset(
    data_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Path]:
    """Load `modma_epochs.npz` without pickle and validate the README contract."""
    resolved = resolve_data_path(data_path)
    with np.load(str(resolved), allow_pickle=False) as npz:
        required = ("X", "y", "subject_id", "ch_names")
        missing = [k for k in required if k not in npz]
        if missing:
            raise KeyError(
                f"Dataset '{resolved}' is missing required keys {missing}. Found: {list(npz.files)}"
            )
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64)
        subject_id = np.asarray(npz["subject_id"], dtype=np.int64)
        ch_names = [str(c) for c in npz["ch_names"].tolist()]

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, C, T), got {X.shape}.")
    if y.shape != (X.shape[0],) or subject_id.shape != (X.shape[0],):
        raise ValueError(
            f"Shape mismatch: X={X.shape}, y={y.shape}, subject_id={subject_id.shape}."
        )
    if len(ch_names) != X.shape[1]:
        raise ValueError(
            f"Channel count mismatch: len(ch_names)={len(ch_names)} vs X.shape[1]={X.shape[1]}."
        )
    return X, y, subject_id, ch_names, resolved


def select_channels(
    X: np.ndarray,
    ch_names: Sequence[str],
    channels: Optional[Sequence[str]] = None,
    channel_ranking_csv: Optional[Union[str, Path]] = None,
    top_k_channels: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Select an ordered subset of EEG channels by name, index, or ranking CSV."""
    if channels is not None and channel_ranking_csv is not None:
        raise ValueError("Specify either --channels or --channel-ranking, not both.")

    ch_list = list(ch_names)
    name_to_idx = {name: i for i, name in enumerate(ch_list)}

    if channel_ranking_csv is not None:
        rank_path = Path(channel_ranking_csv)
        if not rank_path.is_file():
            raise FileNotFoundError(f"Channel ranking file not found: {rank_path}")
        with open(rank_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "channel" not in rows[0] or "score" not in rows[0]:
            raise ValueError(
                f"Channel ranking '{rank_path}' must have columns 'channel,score'."
            )
        seen_csv_channels: set[str] = set()
        parsed_rows: List[Tuple[str, float]] = []
        for row in rows:
            ch_name = str(row["channel"]).strip()
            if not ch_name:
                raise ValueError(f"Empty channel name found in '{rank_path}'.")
            if ch_name in seen_csv_channels:
                raise ValueError(
                    f"Duplicate channel '{ch_name}' found in channel ranking '{rank_path}'."
                )
            seen_csv_channels.add(ch_name)
            try:
                score_val = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid score for channel '{ch_name}' in '{rank_path}': {row['score']!r}"
                ) from exc
            if not math.isfinite(score_val):
                raise ValueError(
                    f"Non-finite score for channel '{ch_name}' in '{rank_path}': {score_val}"
                )
            parsed_rows.append((ch_name, score_val))

        parsed_rows.sort(key=lambda item: item[1], reverse=True)
        k = top_k_channels if top_k_channels is not None else len(parsed_rows)
        if k < 1 or k > len(parsed_rows):
            raise ValueError(f"top_k_channels must be in [1, {len(parsed_rows)}], got {k}.")
        chosen_names = [name for name, _ in parsed_rows[:k]]
        indices = []
        for name in chosen_names:
            if name not in name_to_idx:
                raise KeyError(f"Ranked channel '{name}' not found in dataset ch_names.")
            idx = name_to_idx[name]
            if not (0 <= idx < X.shape[1]):
                raise IndexError(f"Channel index {idx} out of range [0, {X.shape[1]}).")
            indices.append(idx)
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate channel indices resolved from channel ranking.")
        return X[:, indices, :], [ch_list[i] for i in indices], indices

    if channels is not None:
        tokens: List[str] = []
        for item in channels:
            tokens.extend(t.strip() for t in str(item).split(",") if t.strip())
        if not tokens:
            raise ValueError("--channels was provided but empty.")
        indices = []
        for tok in tokens:
            if tok in name_to_idx:
                indices.append(name_to_idx[tok])
            elif tok.lstrip("+-").isdigit() and tok not in ("+", "-"):
                idx = int(tok)
                if not (0 <= idx < len(ch_list)):
                    raise IndexError(f"Channel index {idx} out of range [0, {len(ch_list)}).")
                indices.append(idx)
            else:
                raise KeyError(f"Unknown channel '{tok}' not in dataset ch_names.")
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate channels specified in --channels.")
        return X[:, indices, :], [ch_list[i] for i in indices], indices

    return X, ch_list, list(range(len(ch_list)))


def parse_fold_selection(spec: Optional[Sequence[str]], total_folds: int) -> Optional[set[int]]:
    """Parse fold indices or inclusive ranges like '0-9'."""
    if not spec:
        return None
    selected: set[int] = set()
    for raw in spec:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    raise ValueError(f"Invalid fold range '{part}'.")
                selected.update(range(lo, hi + 1))
            else:
                selected.add(int(part))
    bad = [f for f in sorted(selected) if f < 0 or f >= total_folds]
    if bad:
        raise ValueError(f"Requested folds {bad} out of range [0, {total_folds - 1}].")
    return selected


def _hardware_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
    return info


COMPATIBLE_CONFIG_KEYS = (
    "model",
    "protocol",
    "n_folds",
    "val_fraction",
    "seed",
    "split_seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "optimizer",
    "scheduler",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "class_weighted_loss",
    "grad_clip_norm",
    "use_amp",
    "deterministic",
    "n_classes",
)


def _validate_resume_compatibility(
    existing_meta: Dict[str, Any],
    current_meta: Dict[str, Any],
    exp_dir: Path,
) -> None:
    mismatches: List[str] = []
    old_cfg = existing_meta.get("config", {})
    new_cfg = current_meta.get("config", {})

    for key in COMPATIBLE_CONFIG_KEYS:
        if old_cfg.get(key) != new_cfg.get(key):
            mismatches.append(f"config.{key}: saved={old_cfg.get(key)!r} != current={new_cfg.get(key)!r}")

    for key in (
        "n_epochs",
        "n_channels",
        "n_samples",
        "n_subjects",
        "n_folds_total",
        "selected_channels",
        "selected_channel_indices",
    ):
        if existing_meta.get(key) != current_meta.get(key):
            mismatches.append(f"{key}: saved={existing_meta.get(key)!r} != current={current_meta.get(key)!r}")

    old_data = existing_meta.get("data_path")
    new_data = current_meta.get("data_path")
    if old_data is not None and new_data is not None:
        if Path(old_data).resolve() != Path(new_data).resolve():
            mismatches.append(f"data_path: saved={old_data!r} != current={new_data!r}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"Cannot safely resume experiment in '{exp_dir}' due to configuration mismatch: {joined}"
        )


def run_experiment(
    cfg: TrainConfig,
    channels: Optional[Sequence[str]] = None,
    channel_ranking_csv: Optional[Union[str, Path]] = None,
    top_k_channels: Optional[int] = None,
    fold_subset: Optional[Sequence[str]] = None,
    resume: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run 10-fold CV or 53-fold LOSO experiment and save fold artifacts under `runs/<experiment>/`."""
    cfg.validate()
    # Configure reproducibility before resolving/initializing CUDA context.
    seed_status = set_seed(cfg.seed, deterministic=cfg.deterministic)

    X, y, subject_id, ch_names, resolved_data = load_processed_dataset(cfg.data_path)
    X, selected_ch_names, selected_ch_idx = select_channels(
        X,
        ch_names,
        channels=channels,
        channel_ranking_csv=channel_ranking_csv,
        top_k_channels=top_k_channels,
    )

    device = resolve_device(cfg.device)
    hw = _hardware_info(device)

    # Verify model instantiation and count trainable parameters once before training
    probe_model = build_model(
        cfg.model,
        n_channels=int(X.shape[1]),
        n_samples=int(X.shape[2]),
        n_classes=int(cfg.n_classes),
    )
    n_params = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
    del probe_model

    exp_dir = Path(cfg.runs_dir) / cfg.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_path = exp_dir / "run_config.json"

    if cfg.protocol == "kfold10":
        folds = list(kfold_splits(subject_id, n_folds=cfg.n_folds, seed=cfg.split_seed, y=y))
    else:
        folds = list(loso_splits(subject_id))

    selected_folds = parse_fold_selection(fold_subset, len(folds))

    run_meta = {
        "config": cfg.to_dict(),
        "data_path": str(resolved_data),
        "n_epochs": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_samples": int(X.shape[2]),
        "n_subjects": int(len(np.unique(subject_id))),
        "n_folds_total": int(len(folds)),
        "selected_channels": selected_ch_names,
        "selected_channel_indices": selected_ch_idx,
        "trainable_params": int(n_params),
        "under_100k": bool(n_params < PARAM_LIMIT),
        "hardware": hw,
        "reproducibility": seed_status,
    }

    if run_cfg_path.is_file():
        existing_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        if resume and not overwrite:
            _validate_resume_compatibility(existing_cfg, run_meta, exp_dir)
        elif not overwrite:
            prev_proto = existing_cfg.get("config", {}).get("protocol")
            if prev_proto and prev_proto != cfg.protocol:
                raise ValueError(
                    f"Experiment directory '{exp_dir}' already contains protocol '{prev_proto}', "
                    f"refusing to overwrite with '{cfg.protocol}'. Use a distinct --experiment name "
                    f"(e.g. '{cfg.experiment}_{cfg.protocol}') or pass --overwrite."
                )
            run_cfg_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        else:
            run_cfg_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    else:
        if resume and not overwrite and any(exp_dir.glob("fold_*")):
            raise ValueError(
                f"Cannot safely resume in '{exp_dir}': found existing fold directories without 'run_config.json'."
            )
        run_cfg_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    if verbose:
        gpu_desc = f"{hw['gpu_name']} ({hw['gpu_vram_gb']} GB)" if "gpu_name" in hw else "CPU"
        print(
            f"[Experiment: {cfg.experiment}] protocol={cfg.protocol} ({len(folds)} folds) | "
            f"model={cfg.model} ({n_params:,} params, under_100k={n_params < PARAM_LIMIT}) | "
            f"data={X.shape} | device={gpu_desc} | amp={cfg.use_amp and device.type == 'cuda'}"
        )
        if n_params >= PARAM_LIMIT:
            print(f"[!] Warning: trainable parameter count {n_params:,} >= {PARAM_LIMIT:,} target.")

    summaries: List[Dict[str, Any]] = []
    for k, (train_idx, test_idx) in enumerate(folds):
        if selected_folds is not None and k not in selected_folds:
            continue

        fold_dir = exp_dir / f"fold_{k}"
        preds_file = fold_dir / "preds.csv"
        model_file = fold_dir / "model.pt"
        meta_file = fold_dir / "fold_meta.json"

        if resume and preds_file.is_file() and model_file.is_file() and meta_file.is_file():
            summary = json.loads(meta_file.read_text(encoding="utf-8"))
            summaries.append(summary)
            if verbose:
                print(f"  [fold {k:>2}/{len(folds)-1}] skipped (--resume); test_acc={summary['test_acc']:.4f}")
            continue

        summary = train_fold(
            fold_idx=k,
            X=X,
            y=y,
            subject_ids=subject_id,
            train_idx=train_idx,
            test_idx=test_idx,
            cfg=cfg,
            output_dir=fold_dir,
            ch_names=selected_ch_names,
        )
        summaries.append(summary)
        if verbose:
            print(
                f"  [fold {k:>2}/{len(folds)-1}] best_ep={summary['best_epoch']:>2}/{summary['epochs_ran']:<2} "
                f"val_loss={summary['best_val_loss']} test_acc={summary['test_acc']:.4f} "
                f"({summary['train_time_s']:.1f}s)"
            )

    # Aggregate results from all completed folds across batches in numeric fold order
    all_completed: List[Dict[str, Any]] = []
    for k in range(len(folds)):
        meta_file = exp_dir / f"fold_{k}" / "fold_meta.json"
        if meta_file.is_file():
            all_completed.append(json.loads(meta_file.read_text(encoding="utf-8")))

    summary_path = exp_dir / "training_summary.csv"
    rows_to_write = all_completed if all_completed else summaries
    if rows_to_write:
        fieldnames = [
            "fold",
            "best_epoch",
            "epochs_ran",
            "best_val_loss",
            "best_val_acc",
            "best_val_auc",
            "test_loss",
            "test_acc",
            "test_auc",
            "n_train_epochs",
            "n_val_epochs",
            "n_test_epochs",
            "train_time_s",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_to_write:
                writer.writerow({k: row.get(k) for k in fieldnames})

    return {
        "experiment_dir": str(exp_dir),
        "run_config_path": str(run_cfg_path),
        "summary_path": str(summary_path),
        "folds": summaries,
        "all_folds": all_completed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run CDL-MDDNet / baseline EEG cross-validation.")
    ap.add_argument("--config", type=str, default=None, help="Path to JSON training config.")
    ap.add_argument("--model", type=str, default=None, help="'dummy' or module:ClassName")
    ap.add_argument("--protocol", type=str, choices=["kfold10", "loso"], default=None)
    ap.add_argument("--experiment", type=str, default=None, help="Output folder name under --runs-dir.")
    ap.add_argument("--data", type=str, default=None, help="Path to modma_epochs.npz.")
    ap.add_argument("--runs-dir", type=str, default=None, help="Root output directory (default: runs).")
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="'auto', 'cuda', or 'cpu'.")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    ap.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights.")
    ap.add_argument("--channels", nargs="+", default=None, help="Channel names or indices to keep.")
    ap.add_argument("--channel-ranking", type=str, default=None, help="Path to channel_ranking.csv.")
    ap.add_argument("--top-k-channels", type=int, default=None, help="Top K channels from ranking CSV.")
    ap.add_argument("--folds", nargs="+", default=None, help="Specific fold indices/ranges (e.g. 0-9).")
    ap.add_argument("--resume", action="store_true", help="Skip folds that already have preds.csv and model.pt.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting an experiment directory with a different protocol.")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainConfig.load_default(args.config)

    if args.model is not None:
        cfg.model = args.model
    if args.protocol is not None:
        cfg.protocol = args.protocol
    if args.experiment is not None:
        cfg.experiment = args.experiment
    elif args.model == "dummy":
        cfg.experiment = "dummy"
    if args.data is not None:
        cfg.data_path = args.data
    if args.runs_dir is not None:
        cfg.runs_dir = args.runs_dir
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.seed is not None:
        cfg.seed = args.seed
    if args.split_seed is not None:
        cfg.split_seed = args.split_seed
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.use_amp = False
    if args.no_class_weights:
        cfg.class_weighted_loss = False

    run_experiment(
        cfg=cfg,
        channels=args.channels,
        channel_ranking_csv=args.channel_ranking,
        top_k_channels=args.top_k_channels,
        fold_subset=args.folds,
        resume=args.resume,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

