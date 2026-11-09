"""Per-fold training loop, early stopping, checkpointing, and prediction export."""

from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.normalize import apply_normalizer, fit_normalizer
from src.training.config import TrainConfig, build_model, resolve_device, set_seed
from src.training.splits import make_val_split


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 12, min_delta: float = 1e-4) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be an integer >= 1, got {patience!r}.")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError(f"min_delta must be a finite float >= 0, got {min_delta!r}.")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, val_loss: float, epoch: int) -> Tuple[bool, bool]:
        """Update state with current epoch's validation loss.

        Returns
        -------
        is_best : bool
            True if `val_loss` improved over `best_loss - min_delta`.
        should_stop : bool
            True if patience is exhausted.
        """
        if not math.isfinite(val_loss):
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Cannot create DataLoader from an empty dataset.")
    x_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    ds = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    # Prevent BatchNorm ValueError on a trailing batch of size 1 without discarding small datasets.
    drop_last = bool(shuffle and n_samples > batch_size and (n_samples % batch_size == 1))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=drop_last,
    )


def _compute_class_weights(y_train: np.ndarray, n_classes: int) -> Optional[torch.Tensor]:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    if np.any(counts <= 0):
        return None
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _safe_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, p_pos))
    except ValueError:
        return float("nan")


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    probs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        bs = yb.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        preds = logits.argmax(dim=-1)
        correct += int((preds == yb).sum().item())
        p_mdd = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        probs_list.append(p_mdd)
        targets_list.append(yb.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    acc = correct / max(1, total_samples)
    all_probs = np.concatenate(probs_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    auc = _safe_auc(all_targets, all_probs)
    return avg_loss, acc, auc, all_probs, all_targets


def _save_state_dict_cpu(model: nn.Module, path: Path) -> None:
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, path)


def _load_checkpoint_into_model(
    model: nn.Module,
    model_path: Path,
    device: torch.device,
    fold_idx: int,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Fold {fold_idx}: expected checkpoint not found at '{model_path}'."
        )
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold_idx}: failed to load checkpoint '{model_path}': {exc}"
        ) from exc
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Fold {fold_idx}: invalid checkpoint format in '{model_path}'; "
            f"expected state_dict mapping, got {type(state_dict).__name__}."
        )
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold_idx}: checkpoint '{model_path}' is incompatible with model: {exc}"
        ) from exc


def train_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
    output_dir: Union[str, Path],
    ch_names: Optional[Sequence[str]] = None,
    epoch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Train, early-stop, checkpoint, and evaluate a single cross-validation fold.

    Expected model interface:
    - Input: `(batch_size, n_channels, n_samples)` (e.g. `(B, 128, 1000)`)
    - Output: `(batch_size, 2)` logits
    - Labels: `(batch_size,)` binary integers (`0 = Healthy`, `1 = MDD`)

    Outputs under `output_dir`:
    - `model.pt`: CPU `state_dict` of best validation-loss checkpoint
    - `preds.csv`: `epoch_idx,subject_id,y_true,p_mdd` on `test_idx`
    - `train_log.csv`: per-epoch metrics and learning rate
    - `normalizer_stats.npz`: fold-only normalization stats (mean, std, eps, n_channels)
    - `fold_meta.json`: fold metadata, split subject IDs, and summary metrics
    """
    cfg.validate()
    y_arr = np.asarray(y)
    unique_labels = set(np.unique(y_arr).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Binary trainer expects labels in {{0, 1}}, got {sorted(unique_labels)}."
        )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    overlap_sids = set(np.asarray(subject_ids[train_idx]).tolist()) & set(
        np.asarray(subject_ids[test_idx]).tolist()
    )
    if overlap_sids:
        raise ValueError(
            f"Fold {fold_idx}: subject leakage detected between train and test: {sorted(overlap_sids)}"
        )

    fold_dir = Path(output_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pt"
    preds_path = fold_dir / "preds.csv"
    log_path = fold_dir / "train_log.csv"
    norm_path = fold_dir / "normalizer_stats.npz"
    meta_path = fold_dir / "fold_meta.json"

    # Clear any stale artifacts from a previously interrupted fold run.
    for stale_path in (model_path, preds_path, meta_path):
        if stale_path.exists():
            stale_path.unlink()

    fold_seed = int(cfg.seed) + int(fold_idx)
    set_seed(fold_seed, deterministic=cfg.deterministic)

    device = resolve_device(cfg.device)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    pin_memory = device.type == "cuda"

    # Carve a subject-disjoint validation split strictly from train_idx
    sub_train_idx, val_idx = make_val_split(
        train_idx=train_idx,
        subject_ids=subject_ids,
        y=y,
        val_fraction=cfg.val_fraction,
        seed=fold_seed,
    )

    # Fit normalizer ONLY on sub_train_idx (never val or test)
    norm_stats = fit_normalizer(X[sub_train_idx])
    np.savez(
        str(norm_path),
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        eps=np.array(norm_stats["eps"], dtype=np.float32),
        n_channels=np.array(norm_stats["n_channels"], dtype=np.int64),
    )

    X_sub_train = apply_normalizer(X[sub_train_idx], norm_stats)
    X_val = apply_normalizer(X[val_idx], norm_stats)
    X_test = apply_normalizer(X[test_idx], norm_stats)

    y_sub_train = y[sub_train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_loader = _make_loader(
        X_sub_train,
        y_sub_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        seed=fold_seed,
    )
    val_loader = _make_loader(
        X_val,
        y_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        seed=fold_seed,
    )
    test_loader = _make_loader(
        X_test,
        y_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        seed=fold_seed,
    )

    model = build_model(
        cfg.model,
        n_channels=int(X.shape[1]),
        n_samples=int(X.shape[2]),
        n_classes=int(cfg.n_classes),
    ).to(device)

    class_weights = (
        _compute_class_weights(y_sub_train, cfg.n_classes)
        if cfg.class_weighted_loss
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    val_criterion = nn.CrossEntropyLoss()

    opt_name = cfg.optimizer.lower()
    if opt_name == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    sched_name = cfg.scheduler.lower()
    scheduler: Any = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, cfg.early_stopping_patience // 3)
        )

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )

    fold_t0 = time.perf_counter()
    best_val_acc = 0.0
    best_val_auc = float("nan")
    epochs_ran = 0
    has_valid_checkpoint = False

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_auc",
            "lr",
            "epoch_time_s",
        ])
        log_file.flush()

        for epoch in range(1, cfg.epochs + 1):
            ep_t0 = time.perf_counter()
            model.train()
            tr_loss = 0.0
            tr_samples = 0
            tr_correct = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                bs = yb.size(0)
                tr_loss += float(loss.item()) * bs
                tr_samples += bs
                tr_correct += int((logits.argmax(dim=-1) == yb).sum().item())

            train_loss = tr_loss / max(1, tr_samples)
            train_acc = tr_correct / max(1, tr_samples)

            val_loss, val_acc, val_auc, _, _ = _evaluate_loader(
                model, val_loader, val_criterion, device, amp_enabled
            )

            current_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                if sched_name == "plateau":
                    if math.isfinite(val_loss):
                        scheduler.step(val_loss)
                else:
                    scheduler.step()

            ep_time = time.perf_counter() - ep_t0
            epochs_ran = epoch

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                "" if math.isnan(val_auc) else f"{val_auc:.6f}",
                f"{current_lr:.8f}",
                f"{ep_time:.3f}",
            ])
            log_file.flush()

            is_best, should_stop = early_stopper.step(val_loss, epoch)
            if is_best:
                _save_state_dict_cpu(model, model_path)
                has_valid_checkpoint = True
                best_val_acc = val_acc
                best_val_auc = val_auc

            if epoch_callback is not None:
                epoch_callback({
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "is_best": is_best,
                })

            if should_stop:
                break

    if not has_valid_checkpoint or early_stopper.best_epoch < 1 or not model_path.is_file():
        raise RuntimeError(
            f"Fold {fold_idx}: no valid checkpoint was saved during {epochs_ran} epoch(s) "
            f"(validation loss was never finite)."
        )

    # Reload best checkpoint for held-out test inference
    _load_checkpoint_into_model(model, model_path, device, fold_idx)

    test_loss, test_acc, test_auc, test_probs, _ = _evaluate_loader(
        model, test_loader, val_criterion, device, amp_enabled
    )

    # Export exact README contract: epoch_idx, subject_id, y_true, p_mdd
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_idx", "subject_id", "y_true", "p_mdd"])
        for ep_idx, sid, yt, pm in zip(
            test_idx.tolist(),
            subject_ids[test_idx].tolist(),
            y_test.tolist(),
            test_probs.tolist(),
        ):
            writer.writerow([int(ep_idx), int(sid), int(yt), f"{float(pm):.6f}"])

    total_time = time.perf_counter() - fold_t0
    summary: Dict[str, Any] = {
        "fold": int(fold_idx),
        "fold_seed": int(fold_seed),
        "best_epoch": int(early_stopper.best_epoch),
        "epochs_ran": int(epochs_ran),
        "best_val_loss": None if not math.isfinite(early_stopper.best_loss) else round(early_stopper.best_loss, 6),
        "best_val_acc": round(float(best_val_acc), 6),
        "best_val_auc": None if math.isnan(best_val_auc) else round(float(best_val_auc), 6),
        "test_loss": round(float(test_loss), 6),
        "test_acc": round(float(test_acc), 6),
        "test_auc": None if math.isnan(test_auc) else round(float(test_auc), 6),
        "n_train_epochs": int(len(sub_train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "train_subjects": [int(s) for s in np.unique(subject_ids[sub_train_idx]).tolist()],
        "val_subjects": [int(s) for s in np.unique(subject_ids[val_idx]).tolist()],
        "test_subjects": [int(s) for s in np.unique(subject_ids[test_idx]).tolist()],
        "ch_names": list(ch_names) if ch_names is not None else [f"E{i+1}" for i in range(X.shape[1])],
        "train_time_s": round(float(total_time), 3),
        "model_path": str(model_path),
        "preds_path": str(preds_path),
        "log_path": str(log_path),
    }
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Release per-fold arrays and CUDA cache (critical for 53-fold LOSO on Colab T4)
    del X_sub_train, X_val, X_test, train_loader, val_loader, test_loader, model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary

