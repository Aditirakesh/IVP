"""Unit and integration tests for Member 3's splits, training loop, and runner."""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models.cdl_mddnet import CDLMDDNet
from src.training.config import TrainConfig, build_model, set_seed
from src.training.run_experiment import (
    main as runner_main,
    parse_fold_selection,
    run_experiment,
    select_channels,
)
from src.training.splits import kfold_splits, loso_splits, make_val_split
from src.training.trainer import (
    EarlyStopping,
    TRAIN_CKPT_FILENAME,
    _load_checkpoint_into_model,
    _make_loader,
    train_fold,
)


def _make_synthetic_npz(
    path: Path,
    n_mdd: int = 6,
    n_hc: int = 6,
    epochs_per_subj: int = 4,
    n_channels: int = 8,
    n_samples: int = 100,
    seed: int = 7,
    subject_offset: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    n_subj = n_mdd + n_hc
    n_total = n_subj * epochs_per_subj

    X = rng.normal(0.0, 1.0, size=(n_total, n_channels, n_samples)).astype(np.float32)
    y = np.repeat(np.array([1] * n_mdd + [0] * n_hc, dtype=np.int64), epochs_per_subj)
    # Give MDD epochs a mild signal offset so training has a real learnable signal
    X[y == 1] += 0.75
    subject_id = np.repeat(
        np.arange(subject_offset, subject_offset + n_subj, dtype=np.int64), epochs_per_subj
    )
    ch_names = np.array([f"E{i+1}" for i in range(n_channels)], dtype="<U8")

    np.savez(str(path), X=X, y=y, subject_id=subject_id, ch_names=ch_names)
    return path


class TestSplits:
    def test_stratified_kfold_53_modma_subjects(self):
        rng = np.random.default_rng(0)
        # MODMA cohort: 24 MDD (1) + 29 HC (0)
        subj_labels = np.array([1] * 24 + [0] * 29, dtype=np.int64)
        epochs_per_subj = rng.integers(60, 85, size=53)
        subject_id = np.repeat(np.arange(53, dtype=np.int64), epochs_per_subj)
        y = np.repeat(subj_labels, epochs_per_subj)

        folds = list(kfold_splits(subject_id, n_folds=10, seed=0, y=y))
        assert len(folds) == 10

        seen = np.zeros(len(subject_id), dtype=int)
        for tr, te in folds:
            assert set(subject_id[tr]).isdisjoint(set(subject_id[te]))
            te_subs = np.unique(subject_id[te])
            te_sub_labels = [int(y[subject_id == s][0]) for s in te_subs]
            # Every fold should contain both MDD (2-3) and HC (2-3) participants
            assert sum(te_sub_labels) in (2, 3)
            assert (len(te_sub_labels) - sum(te_sub_labels)) in (2, 3)
            seen[te] += 1
        assert (seen == 1).all()

    def test_loso_53_subjects_yields_53_folds(self):
        subject_id = np.repeat(np.arange(53, dtype=np.int64), 5)
        folds = list(loso_splits(subject_id))
        assert len(folds) == 53

        seen = np.zeros(len(subject_id), dtype=int)
        for k, (tr, te) in enumerate(folds):
            assert np.unique(subject_id[te]).tolist() == [k]
            assert k not in set(subject_id[tr])
            seen[te] += 1
        assert (seen == 1).all()

    def test_make_val_split_is_subject_disjoint(self):
        subject_id = np.repeat(np.arange(20, dtype=np.int64), 4)
        y = np.repeat(np.array([1] * 10 + [0] * 10, dtype=np.int64), 4)
        train_idx = np.where(subject_id < 18)[0]

        sub_tr, val = make_val_split(train_idx, subject_id, y=y, val_fraction=0.2, seed=42)
        tr_subs = set(subject_id[sub_tr].tolist())
        val_subs = set(subject_id[val].tolist())
        assert tr_subs.isdisjoint(val_subs)
        assert tr_subs | val_subs == set(range(18))
        assert np.array_equal(np.sort(np.concatenate([sub_tr, val])), train_idx)
        # Both classes should be represented in validation
        assert set(np.unique(y[val]).tolist()) == {0, 1}


class TestConfigAndEarlyStopping:
    def test_default_train_config_loads(self):
        cfg = TrainConfig.load_default()
        assert cfg.protocol == "kfold10"
        assert cfg.n_folds == 10
        assert cfg.seed == 42
        assert cfg.use_dann is True
        assert cfg.dann_gamma == 10.0
        assert cfg.dann_loss_weight == 1.0

    def test_early_stopping_triggers_on_plateau(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        assert es.step(1.0, 1) == (True, False)
        assert es.step(0.8, 2) == (True, False)
        assert es.step(0.8, 3) == (False, False)
        assert es.step(0.81, 4) == (False, True)
        assert es.best_epoch == 2

    def test_reproducibility_with_fixed_seed(self, tmp_path):
        npz = _make_synthetic_npz(tmp_path / "data.npz", n_mdd=4, n_hc=4, epochs_per_subj=3)
        with np.load(str(npz)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr, te = next(kfold_splits(sids, n_folds=4, seed=0, y=y))
        cfg = TrainConfig(
            model="dummy",
            use_dann=False,
            epochs=3,
            batch_size=8,
            seed=123,
            device="cpu",
            use_amp=False,
        )

        out1 = train_fold(0, X, y, sids, tr, te, cfg, tmp_path / "r1")
        out2 = train_fold(0, X, y, sids, tr, te, cfg, tmp_path / "r2")

        p1 = Path(out1["preds_path"]).read_text(encoding="utf-8")
        p2 = Path(out2["preds_path"]).read_text(encoding="utf-8")
        assert p1 == p2

    def test_set_seed_produces_identical_weights(self):
        set_seed(99)
        m1 = build_model("dummy", n_channels=8, n_samples=100, n_classes=2)
        set_seed(99)
        m2 = build_model("dummy", n_channels=8, n_samples=100, n_classes=2)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)


class TestRunnerAndOutputs:
    def test_kfold_end_to_end_contracts_and_resume(self, tmp_path):
        data_path = _make_synthetic_npz(tmp_path / "modma_epochs.npz", n_mdd=5, n_hc=5)
        runs_dir = tmp_path / "runs"

        cfg = TrainConfig(
            experiment="dummy_kfold",
            model="dummy",
            use_dann=False,
            protocol="kfold10",
            data_path=str(data_path),
            runs_dir=str(runs_dir),
            n_folds=5,
            val_fraction=0.2,
            epochs=4,
            batch_size=8,
            early_stopping_patience=2,
            seed=42,
            split_seed=0,
            device="cpu",
            use_amp=False,
        )

        res = run_experiment(cfg, verbose=False)
        assert len(res["folds"]) == 5

        exp_dir = runs_dir / "dummy_kfold"
        assert (exp_dir / "run_config.json").is_file()
        assert (exp_dir / "training_summary.csv").is_file()

        seen_epochs = set()
        for k in range(5):
            fold_dir = exp_dir / f"fold_{k}"
            model_pt = fold_dir / "model.pt"
            preds_csv = fold_dir / "preds.csv"
            log_csv = fold_dir / "train_log.csv"
            norm_npz = fold_dir / "normalizer_stats.npz"
            assert model_pt.is_file()
            assert preds_csv.is_file()
            assert log_csv.is_file()
            assert norm_npz.is_file()

            # Verify model.pt loads cleanly with weights_only=True
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            model = build_model("dummy", n_channels=8, n_samples=100, n_classes=2)
            model.load_state_dict(state)

            with open(preds_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ["epoch_idx", "subject_id", "y_true", "p_mdd"]
                for row in reader:
                    seen_epochs.add(int(row["epoch_idx"]))
                    assert int(row["y_true"]) in (0, 1)
                    p = float(row["p_mdd"])
                    assert 0.0 <= p <= 1.0

        assert seen_epochs == set(range(40))

        # Resume should keep existing files untouched
        mtime_before = (exp_dir / "fold_0" / "model.pt").stat().st_mtime_ns
        run_experiment(cfg, resume=True, verbose=False)
        mtime_after = (exp_dir / "fold_0" / "model.pt").stat().st_mtime_ns
        assert mtime_before == mtime_after

        # Switching protocol into same experiment directory without overwrite must fail
        cfg_loso = TrainConfig(**{**cfg.to_dict(), "protocol": "loso"})
        with pytest.raises(ValueError, match="already contains protocol"):
            run_experiment(cfg_loso, verbose=False)

    def test_loso_and_channel_ranking_selection(self, tmp_path):
        data_path = _make_synthetic_npz(
            tmp_path / "modma_epochs.npz", n_mdd=3, n_hc=3, epochs_per_subj=3, n_channels=8
        )
        ranking_csv = tmp_path / "channel_ranking.csv"
        ranking_csv.write_text(
            "channel,score\nE3,0.95\nE1,0.88\nE7,0.72\nE2,0.40\n",
            encoding="utf-8",
        )

        cfg = TrainConfig(
            experiment="dummy_ch3_loso",
            model="dummy",
            use_dann=False,
            protocol="loso",
            data_path=str(data_path),
            runs_dir=str(tmp_path / "runs"),
            val_fraction=0.25,
            epochs=2,
            batch_size=4,
            device="cpu",
            use_amp=False,
        )

        res = run_experiment(
            cfg,
            channel_ranking_csv=ranking_csv,
            top_k_channels=3,
            verbose=False,
        )
        assert len(res["folds"]) == 6
        run_meta = json.loads(Path(res["run_config_path"]).read_text(encoding="utf-8"))
        assert run_meta["n_channels"] == 3
        assert run_meta["selected_channels"] == ["E3", "E1", "E7"]

    def test_select_channels_and_fold_range_parsing(self):
        X = np.zeros((4, 5, 10), dtype=np.float32)
        ch = ["E1", "E2", "E3", "E4", "E5"]
        X_sub, names, idx = select_channels(X, ch, channels=["E4,E2"])
        assert X_sub.shape == (4, 2, 10)
        assert names == ["E4", "E2"]
        assert idx == [3, 1]

        assert parse_fold_selection(["0-2", "5"], total_folds=10) == {0, 1, 2, 5}


class _BatchNormTestNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(n_channels, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(8)
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.conv(x))).mean(dim=-1)
        return self.fc(h)


class _ControlledValNaNNet(nn.Module):
    def __init__(self, n_channels: int, fail_from_eval_call: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_channels, 2)
        self.fail_from_eval_call = fail_from_eval_call
        self.eval_calls = 0
        self.was_in_val = False

    def train(self, mode: bool = True):
        if mode and self.was_in_val:
            self.was_in_val = False
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.mean(dim=-1))
        if not self.training:
            if not self.was_in_val:
                self.eval_calls += 1
                self.was_in_val = True
            if self.eval_calls >= self.fail_from_eval_call:
                return out * float("nan")
        return out


class TestRegressionFixes:
    def test_make_loader_avoids_singleton_last_batch_and_preserves_small_data(self):
        X9 = np.zeros((9, 4, 10), dtype=np.float32)
        y9 = np.zeros(9, dtype=np.int64)
        loader_drop = _make_loader(X9, y9, batch_size=4, shuffle=True, num_workers=0, pin_memory=False, seed=0)
        assert loader_drop.drop_last is True
        assert [yb.size(0) for _, yb in loader_drop] == [4, 4]

        X5 = np.zeros((5, 4, 10), dtype=np.float32)
        y5 = np.zeros(5, dtype=np.int64)
        loader_small = _make_loader(X5, y5, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, seed=0)
        assert loader_small.drop_last is False
        assert [yb.size(0) for _, yb in loader_small] == [5]

        X1 = np.zeros((1, 4, 10), dtype=np.float32)
        y1 = np.zeros(1, dtype=np.int64)
        loader_one = _make_loader(X1, y1, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, seed=0)
        assert loader_one.drop_last is False
        assert [yb.size(0) for _, yb in loader_one] == [1]

        with pytest.raises(ValueError, match="empty"):
            _make_loader(np.zeros((0, 4, 10), dtype=np.float32), np.zeros(0, dtype=np.int64), 4, True, 0, False, 0)

    def test_batchnorm_singleton_final_batch_and_small_dataset_train_fold(self, tmp_path, monkeypatch):
        # 5 subjects total (3 epochs each): outer test = 1 subj (3 epochs), outer train = 4 subjs (12 epochs).
        # val_fraction=0.25 -> 1 val subj (3 epochs) + 3 train subjs (9 epochs). With batch_size=4, 9 % 4 == 1.
        npz = _make_synthetic_npz(tmp_path / "bn.npz", n_mdd=3, n_hc=2, epochs_per_subj=3, n_channels=8, n_samples=20)
        with np.load(str(npz)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr = np.where(sids < 4)[0]
        te = np.where(sids == 4)[0]

        monkeypatch.setattr(
            "src.training.trainer.build_model",
            lambda model_spec, n_channels, n_samples=1000, n_classes=2, n_subjects=0: _BatchNormTestNet(
                n_channels, n_samples, n_classes
            ),
        )

        # Case 1: N % batch_size == 1 (9 % 4 == 1) must not crash BatchNorm1d
        cfg_bn = TrainConfig(
            model="dummy",
            use_dann=False,
            val_fraction=0.25,
            epochs=2,
            batch_size=4,
            device="cpu",
            use_amp=False,
        )
        out = train_fold(0, X, y, sids, tr, te, cfg_bn, tmp_path / "fold_bn")
        assert out["n_train_epochs"] == 9
        assert Path(out["preds_path"]).is_file()

        # Case 2: small dataset (N=9 < batch_size=16) must not discard all training batches
        cfg_small = TrainConfig(
            model="dummy",
            use_dann=False,
            val_fraction=0.25,
            epochs=2,
            batch_size=16,
            device="cpu",
            use_amp=False,
        )
        out_small = train_fold(0, X, y, sids, tr, te, cfg_small, tmp_path / "fold_small")
        assert out_small["n_train_epochs"] == 9
        assert Path(out_small["model_path"]).is_file()

    def test_nan_validation_loss_clears_stale_checkpoint_and_raises(self, tmp_path, monkeypatch):
        npz = _make_synthetic_npz(tmp_path / "nan.npz", n_mdd=3, n_hc=3, epochs_per_subj=3)
        with np.load(str(npz)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr, te = next(kfold_splits(sids, n_folds=3, seed=0, y=y))

        fold_dir = tmp_path / "fold_stale"
        fold_dir.mkdir(parents=True)
        stale_ckpt = fold_dir / "model.pt"
        torch.save({"stale": torch.tensor([1.0])}, stale_ckpt)
        (fold_dir / "preds.csv").write_text("stale\n", encoding="utf-8")
        (fold_dir / "fold_meta.json").write_text("{}\n", encoding="utf-8")

        monkeypatch.setattr(
            "src.training.trainer.build_model",
            lambda model_spec, n_channels, n_samples=1000, n_classes=2, n_subjects=0: _ControlledValNaNNet(
                n_channels, fail_from_eval_call=1
            ),
        )
        cfg = TrainConfig(
            model="dummy",
            use_dann=False,
            val_fraction=0.25,
            epochs=3,
            batch_size=4,
            scheduler="plateau",
            early_stopping_patience=2,
            device="cpu",
            use_amp=False,
        )

        with pytest.raises(RuntimeError, match="no valid checkpoint was saved"):
            train_fold(0, X, y, sids, tr, te, cfg, fold_dir)

        assert not stale_ckpt.exists()
        assert not (fold_dir / "preds.csv").exists()
        assert not (fold_dir / "fold_meta.json").exists()

    def test_nan_validation_loss_after_valid_epoch_preserves_best_checkpoint(self, tmp_path, monkeypatch):
        npz = _make_synthetic_npz(tmp_path / "nan_later.npz", n_mdd=3, n_hc=3, epochs_per_subj=3)
        with np.load(str(npz)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr, te = next(kfold_splits(sids, n_folds=3, seed=0, y=y))

        class _EvalEpoch2NaNNet(nn.Module):
            def __init__(self, n_channels: int) -> None:
                super().__init__()
                self.linear = nn.Linear(n_channels, 2)
                self.nan_next_eval = False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.linear(x.mean(dim=-1))
                if not self.training and self.nan_next_eval:
                    return out * float("nan")
                return out

        holders = []

        def _make_net(model_spec, n_channels, n_samples=1000, n_classes=2, n_subjects=0):
            m = _EvalEpoch2NaNNet(n_channels)
            holders.append(m)
            return m

        monkeypatch.setattr("src.training.trainer.build_model", _make_net)
        cfg = TrainConfig(
            model="dummy",
            use_dann=False,
            val_fraction=0.25,
            epochs=5,
            batch_size=8,
            early_stopping_patience=2,
            device="cpu",
            use_amp=False,
        )

        def _flip(info):
            # Force NaN during validation on epochs 2 and 3, then clear before final test eval.
            holders[0].nan_next_eval = info["epoch"] in (1, 2)

        summary = train_fold(0, X, y, sids, tr, te, cfg, tmp_path / "fold_nan_ep2", epoch_callback=_flip)
        assert summary["best_epoch"] == 1
        assert summary["epochs_ran"] == 3
        assert summary["best_val_loss"] is not None
        assert np.isfinite(summary["test_loss"])

    def test_missing_and_invalid_checkpoint_errors(self, tmp_path):
        model = build_model("dummy", n_channels=8, n_samples=100, n_classes=2)
        device = torch.device("cpu")

        with pytest.raises(FileNotFoundError, match="expected checkpoint not found"):
            _load_checkpoint_into_model(model, tmp_path / "missing.pt", device, fold_idx=0)

        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"not a valid torch checkpoint")
        with pytest.raises(RuntimeError, match="failed to load checkpoint"):
            _load_checkpoint_into_model(model, corrupt, device, fold_idx=0)

        non_dict = tmp_path / "list.pt"
        torch.save([1, 2, 3], non_dict)
        with pytest.raises(RuntimeError, match="invalid checkpoint format"):
            _load_checkpoint_into_model(model, non_dict, device, fold_idx=0)

        wrong_model = build_model("dummy", n_channels=4, n_samples=100, n_classes=2)
        wrong_ckpt = tmp_path / "wrong_shape.pt"
        torch.save(wrong_model.state_dict(), wrong_ckpt)
        with pytest.raises(RuntimeError, match="incompatible with model"):
            _load_checkpoint_into_model(model, wrong_ckpt, device, fold_idx=0)

    def test_resume_rejects_changed_config_and_preserves_original_run_config(self, tmp_path):
        data_a = _make_synthetic_npz(tmp_path / "modma_a.npz", n_mdd=4, n_hc=4, epochs_per_subj=3)
        data_b = _make_synthetic_npz(tmp_path / "modma_b.npz", n_mdd=4, n_hc=4, epochs_per_subj=3, seed=99)
        runs_dir = tmp_path / "runs"
        base_cfg = TrainConfig(
            experiment="safe_resume_exp",
            model="dummy",
            use_dann=False,
            protocol="kfold10",
            data_path=str(data_a),
            runs_dir=str(runs_dir),
            n_folds=4,
            val_fraction=0.2,
            epochs=2,
            batch_size=8,
            seed=42,
            split_seed=0,
            device="cpu",
            use_amp=False,
        )

        run_experiment(base_cfg, fold_subset=["0"], verbose=False)
        run_cfg_path = runs_dir / "safe_resume_exp" / "run_config.json"
        orig_text = run_cfg_path.read_text(encoding="utf-8")
        orig_mtime = run_cfg_path.stat().st_mtime_ns

        # Mutate lr
        cfg_bad_lr = TrainConfig(**{**base_cfg.to_dict(), "lr": 5e-4})
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_experiment(cfg_bad_lr, fold_subset=["1"], resume=True, verbose=False)

        # Mutate seed
        cfg_bad_seed = TrainConfig(**{**base_cfg.to_dict(), "seed": 7})
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_experiment(cfg_bad_seed, fold_subset=["1"], resume=True, verbose=False)

        # Mutate data_path
        cfg_bad_data = TrainConfig(**{**base_cfg.to_dict(), "data_path": str(data_b)})
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_experiment(cfg_bad_data, fold_subset=["1"], resume=True, verbose=False)

        # Mutate selected channels
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_experiment(base_cfg, channels=["E1", "E2"], fold_subset=["1"], resume=True, verbose=False)

        assert run_cfg_path.read_text(encoding="utf-8") == orig_text
        assert run_cfg_path.stat().st_mtime_ns == orig_mtime

        # Compatible resume preserves run_config.json and fold_0 without overwriting
        fold0_mtime = (runs_dir / "safe_resume_exp" / "fold_0" / "model.pt").stat().st_mtime_ns
        res = run_experiment(base_cfg, fold_subset=["0", "1"], resume=True, verbose=False)
        assert len(res["folds"]) == 2
        assert run_cfg_path.read_text(encoding="utf-8") == orig_text
        assert run_cfg_path.stat().st_mtime_ns == orig_mtime
        assert (runs_dir / "safe_resume_exp" / "fold_0" / "model.pt").stat().st_mtime_ns == fold0_mtime

        # Resume without run_config.json when fold_* directories already exist must fail
        exp_no_cfg = runs_dir / "missing_cfg_exp"
        (exp_no_cfg / "fold_0").mkdir(parents=True)
        with pytest.raises(ValueError, match="without 'run_config.json'"):
            run_experiment(TrainConfig(**{**base_cfg.to_dict(), "experiment": "missing_cfg_exp"}), resume=True, verbose=False)

    def test_summary_aggregation_across_fold_batches(self, tmp_path):
        data_path = _make_synthetic_npz(tmp_path / "batched.npz", n_mdd=5, n_hc=5, epochs_per_subj=3)
        runs_dir = tmp_path / "runs"
        cfg = TrainConfig(
            experiment="batched_kfold",
            model="dummy",
            use_dann=False,
            protocol="kfold10",
            data_path=str(data_path),
            runs_dir=str(runs_dir),
            n_folds=5,
            val_fraction=0.2,
            epochs=2,
            batch_size=8,
            device="cpu",
            use_amp=False,
        )

        res1 = run_experiment(cfg, fold_subset=["0-1"], verbose=False)
        assert [r["fold"] for r in res1["all_folds"]] == [0, 1]

        res2 = run_experiment(cfg, fold_subset=["2-3"], resume=True, verbose=False)
        assert [r["fold"] for r in res2["all_folds"]] == [0, 1, 2, 3]

        res3 = run_experiment(cfg, fold_subset=["4"], resume=True, verbose=False)
        assert [r["fold"] for r in res3["folds"]] == [4]
        assert [r["fold"] for r in res3["all_folds"]] == [0, 1, 2, 3, 4]

        summary_csv = runs_dir / "batched_kfold" / "training_summary.csv"
        with open(summary_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [int(r["fold"]) for r in rows] == [0, 1, 2, 3, 4]

    def test_channel_ranking_and_selection_validation(self, tmp_path):
        X = np.zeros((4, 5, 10), dtype=np.float32)
        ch = ["E1", "E2", "E3", "E4", "E5"]

        with pytest.raises(ValueError, match="Duplicate channels"):
            select_channels(X, ch, channels=["E1", "E1"])
        with pytest.raises(ValueError, match="Duplicate channels"):
            select_channels(X, ch, channels=["0", "E1"])
        with pytest.raises(IndexError, match="out of range"):
            select_channels(X, ch, channels=["-1"])
        with pytest.raises(IndexError, match="out of range"):
            select_channels(X, ch, channels=["5"])

        dup_csv = tmp_path / "dup_rank.csv"
        dup_csv.write_text("channel,score\nE1,0.9\nE1,0.8\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate channel"):
            select_channels(X, ch, channel_ranking_csv=dup_csv)

        bad_score_csv = tmp_path / "nan_rank.csv"
        bad_score_csv.write_text("channel,score\nE1,nan\nE2,0.5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Non-finite score"):
            select_channels(X, ch, channel_ranking_csv=bad_score_csv)

        valid_csv = tmp_path / "valid_rank.csv"
        valid_csv.write_text("channel,score\nE1,0.9\nE2,0.5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top_k_channels"):
            select_channels(X, ch, channel_ranking_csv=valid_csv, top_k_channels=3)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"weight_decay": -1e-4},
            {"weight_decay": float("nan")},
            {"early_stopping_patience": 0},
            {"early_stopping_min_delta": -0.01},
            {"early_stopping_min_delta": float("inf")},
            {"grad_clip_norm": 0.0},
            {"grad_clip_norm": -1.0},
            {"grad_clip_norm": float("nan")},
            {"seed": -1},
            {"seed": True},
            {"split_seed": -5},
            {"num_workers": -1},
            {"n_classes": 3},
            {"n_classes": 1},
            {"lr": 0.0},
            {"val_fraction": 0.5},
            {"optimizer": "sgd"},
            {"scheduler": "invalid"},
            {"device": "tpu"},
            {"use_dann": "true"},
            {"dann_gamma": -1.0},
            {"dann_gamma": float("nan")},
            {"dann_loss_weight": -0.5},
            {"dann_loss_weight": float("inf")},
        ],
    )
    def test_invalid_config_values_raise_value_error(self, overrides):
        cfg = TrainConfig(**overrides)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_invalid_json_config_files_raise_clear_errors(self, tmp_path):
        empty_cfg = tmp_path / "empty.json"
        empty_cfg.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            TrainConfig.from_json(empty_cfg)

        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            TrainConfig.from_json(bad_json)

        list_json = tmp_path / "list.json"
        list_json.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            TrainConfig.from_json(list_json)

        unknown_json = tmp_path / "unknown.json"
        unknown_json.write_text('{"unknown_param": 123}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown config keys"):
            TrainConfig.from_json(unknown_json)

        bad_val_json = tmp_path / "bad_val.json"
        bad_val_json.write_text('{"n_classes": 3}', encoding="utf-8")
        with pytest.raises(ValueError, match="n_classes == 2"):
            TrainConfig.from_json(bad_val_json)

    def test_non_binary_labels_rejected_and_reproducibility_metadata(self, tmp_path, monkeypatch):
        X = np.zeros((12, 4, 10), dtype=np.float32)
        y = np.array([0, 1, 2] * 4, dtype=np.int64)
        sids = np.repeat(np.arange(6, dtype=np.int64), 2)
        cfg = TrainConfig(model="dummy", use_dann=False, epochs=1, batch_size=4, device="cpu", use_amp=False)
        with pytest.raises(ValueError, match="labels in \\{0, 1\\}"):
            train_fold(0, X, y, sids, np.arange(8), np.arange(8, 12), cfg, tmp_path / "bad_y")

        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        status = set_seed(42, deterministic=True)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert status["cublas_workspace_config"] == ":4096:8"
        assert isinstance(status["limitations"], list) and len(status["limitations"]) >= 1
        assert status["strict_determinism_guaranteed"] is False

        with pytest.raises(ValueError, match="seed"):
            set_seed(-1)

        es = EarlyStopping(patience=2, min_delta=1e-4)
        assert es.step(float("inf"), 1) == (False, False)
        assert es.step(float("nan"), 2) == (False, True)
        assert es.best_epoch == 0
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)
        with pytest.raises(ValueError, match="min_delta"):
            EarlyStopping(patience=2, min_delta=-1e-4)


class TestDANNIntegration:
    def test_dann_unsupported_model_fails_explicitly(self, tmp_path):
        data_path = _make_synthetic_npz(
            tmp_path / "unsupported.npz", n_mdd=3, n_hc=3, epochs_per_subj=3
        )
        cfg = TrainConfig(
            experiment="dummy_with_dann",
            model="dummy",
            use_dann=True,
            data_path=str(data_path),
            runs_dir=str(tmp_path / "runs"),
            n_folds=3,
            epochs=2,
            batch_size=8,
            device="cpu",
            use_amp=False,
        )
        with pytest.raises(ValueError, match="does not accept 'n_subjects'"):
            run_experiment(cfg, verbose=False)

        with np.load(str(data_path)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr, te = next(kfold_splits(sids, n_folds=3, seed=0, y=y))
        with pytest.raises(ValueError, match="does not accept 'n_subjects'"):
            train_fold(0, X, y, sids, tr, te, cfg, tmp_path / "fold_err")

        with pytest.raises(ValueError, match="does not accept 'n_subjects'"):
            runner_main([
                "--model", "dummy",
                "--use-dann",
                "--data", str(data_path),
                "--runs-dir", str(tmp_path / "runs"),
            ])

        class _AcceptsNSubjectsNoForwardTrain(nn.Module):
            def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2, n_subjects: int = 0) -> None:
                super().__init__()
                self.fc = nn.Linear(n_channels, n_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x.mean(dim=-1))

        # Test second check: accepts n_subjects in __init__, but lacks forward_train(x, lambd)
        import src.training.config as cfg_mod

        orig_loader = cfg_mod.load_model_class
        try:
            cfg_mod.load_model_class = lambda spec: _AcceptsNSubjectsNoForwardTrain
            with pytest.raises(ValueError, match="forward_train"):
                build_model("custom:NoForwardTrain", n_channels=8, n_samples=32, n_subjects=4)
        finally:
            cfg_mod.load_model_class = orig_loader

    def test_dann_active_losses_gradient_reversal_and_val_test_exclusion(self, tmp_path, monkeypatch):
        data_path = _make_synthetic_npz(
            tmp_path / "dann_active.npz",
            n_mdd=5,
            n_hc=5,
            epochs_per_subj=4,
            n_channels=16,
            n_samples=64,
            subject_offset=2010002,
        )
        with np.load(str(data_path)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]

        tr, te = next(kfold_splits(sids, n_folds=5, seed=0, y=y))
        cfg = TrainConfig(
            model="src.models.cdl_mddnet:CDLMDDNet",
            use_dann=True,
            dann_gamma=10.0,
            dann_loss_weight=1.0,
            val_fraction=0.25,
            epochs=3,
            batch_size=8,
            lr=1e-2,
            seed=77,
            device="cpu",
            use_amp=False,
        )

        # Capture initial weights with identical seed/fold_seed and n_subjects=len(sub_train)
        sub_tr, _ = make_val_split(tr, sids, y=y, val_fraction=0.25, seed=77)
        k_train = int(len(np.unique(sids[sub_tr])))
        set_seed(77, deterministic=cfg.deterministic)
        init_model = CDLMDDNet(n_channels=16, n_samples=64, n_classes=2, n_subjects=k_train)
        init_subj_w = init_model.subject_head[0].weight.detach().clone()
        init_spatial_w = init_model.spatial[0].weight.detach().clone()

        orig_forward = CDLMDDNet.forward
        orig_forward_train = CDLMDDNet.forward_train
        train_calls = []
        eval_calls = []

        def _spy_forward_train(self, xb, lambd=1.0):
            cls_l, subj_l = orig_forward_train(self, xb, lambd=lambd)
            train_calls.append((bool(self.training), float(lambd), tuple(subj_l.shape)))
            return cls_l, subj_l

        def _spy_forward(self, xb):
            out = orig_forward(self, xb)
            eval_calls.append((bool(self.training), tuple(out.shape)))
            return out

        monkeypatch.setattr(CDLMDDNet, "forward_train", _spy_forward_train)
        monkeypatch.setattr(CDLMDDNet, "forward", _spy_forward)

        fold_dir = tmp_path / "fold_dann_active"
        ep_records = []
        summary = train_fold(
            0, X, y, sids, tr, te, cfg, fold_dir, epoch_callback=ep_records.append
        )

        # 1. forward_train called ONLY during training, forward ONLY during eval/test
        assert len(train_calls) >= 3
        assert all(is_train is True for is_train, _, _ in train_calls)
        assert all(lambd > 0.0 for _, lambd, _ in train_calls)
        assert all(shape[1] == k_train for _, _, shape in train_calls)
        assert len(eval_calls) >= 4  # >= 3 val epochs + 1 test eval
        assert all(is_train is False for is_train, _ in eval_calls)

        # 2. Both classification and subject losses logged and combined accurately
        assert len(ep_records) == 3
        for rec in ep_records:
            assert rec["train_cls_loss"] > 0.0
            assert rec["train_subj_loss"] is not None and rec["train_subj_loss"] > 0.0
            assert rec["dann_lambda"] is not None and rec["dann_lambda"] > 0.0
            expected_total = rec["train_cls_loss"] + cfg.dann_loss_weight * rec["train_subj_loss"]
            assert abs(rec["train_loss"] - expected_total) < 1e-4

        with open(fold_dir / "train_log.csv", newline="", encoding="utf-8") as f:
            log_rows = list(csv.DictReader(f))
        assert len(log_rows) == 3
        assert all(float(r["train_subj_loss"]) > 0.0 for r in log_rows)
        assert all(float(r["dann_lambda"]) > 0.0 for r in log_rows)

        # 3. Subject head and encoder actually updated by optimization
        train_ckpt = torch.load(
            fold_dir / TRAIN_CKPT_FILENAME, map_location="cpu", weights_only=True
        )
        trained_subj_w = train_ckpt["model_state_dict"]["subject_head.0.weight"]
        trained_spatial_w = train_ckpt["model_state_dict"]["spatial.0.weight"]
        assert not torch.allclose(init_subj_w, trained_subj_w)
        assert not torch.allclose(init_spatial_w, trained_spatial_w)

        # 4. Verify inference checkpoint strictly matches standard CDLMDDNet(n_channels, n_samples, n_classes)
        inf_state = torch.load(fold_dir / "model.pt", map_location="cpu", weights_only=True)
        assert not any(k.startswith("subject_head.") for k in inf_state)
        inf_model = CDLMDDNet(n_channels=16, n_samples=64, n_classes=2)
        inf_model.load_state_dict(inf_state, strict=True)
        assert summary["use_dann"] is True
        assert summary["n_train_subjects"] == k_train

    def test_dann_per_fold_remapping_no_leakage_and_checkpoints_10fold_loso(self, tmp_path):
        # 10 subjects (5 MDD, 5 HC) with MODMA-like IDs for 10-fold CV with real CDLMDDNet
        data_10 = _make_synthetic_npz(
            tmp_path / "modma_10subj.npz",
            n_mdd=5,
            n_hc=5,
            epochs_per_subj=4,
            n_channels=16,
            n_samples=64,
            subject_offset=2010002,
        )
        runs_dir = tmp_path / "runs"
        cfg_10 = TrainConfig(
            experiment="cdl_dann_10fold",
            model="src.models.cdl_mddnet:CDLMDDNet",
            protocol="kfold10",
            use_dann=True,
            dann_gamma=10.0,
            dann_loss_weight=1.0,
            data_path=str(data_10),
            runs_dir=str(runs_dir),
            n_folds=10,
            val_fraction=0.2,
            epochs=4,
            early_stopping_patience=2,
            batch_size=8,
            device="cpu",
            use_amp=False,
        )
        res_10 = run_experiment(cfg_10, verbose=False)
        assert len(res_10["folds"]) == 10

        seen_epochs = set()
        fold_subject_maps = []
        for k in range(10):
            fdir = runs_dir / "cdl_dann_10fold" / f"fold_{k}"
            meta = json.loads((fdir / "fold_meta.json").read_text(encoding="utf-8"))
            assert meta["use_dann"] is True
            assert meta["dann_gamma"] == 10.0
            assert meta["dann_loss_weight"] == 1.0
            tr_s = set(meta["train_subjects"])
            va_s = set(meta["val_subjects"])
            te_s = set(meta["test_subjects"])
            assert tr_s.isdisjoint(va_s)
            assert tr_s.isdisjoint(te_s)
            assert va_s.isdisjoint(te_s)

            smap = meta["subject_id_map"]
            assert smap is not None
            assert {int(sid) for sid in smap.keys()} == tr_s
            assert sorted(smap.values()) == list(range(len(tr_s)))
            assert meta["n_train_subjects"] == len(tr_s)
            fold_subject_maps.append(smap)

            # Inference checkpoint strict load without subject_head
            inf_state = torch.load(fdir / "model.pt", map_location="cpu", weights_only=True)
            assert not any(key.startswith("subject_head.") for key in inf_state)
            CDLMDDNet(n_channels=16, n_samples=64, n_classes=2).load_state_dict(
                inf_state, strict=True
            )

            # Training checkpoint includes subject_head + resume metadata
            ckpt = torch.load(fdir / TRAIN_CKPT_FILENAME, map_location="cpu", weights_only=True)
            assert any(key.startswith("subject_head.") for key in ckpt["model_state_dict"])
            assert ckpt["subject_id_map"] == smap
            assert ckpt["n_subjects"] == len(tr_s)

            with open(fdir / "preds.csv", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ["epoch_idx", "subject_id", "y_true", "p_mdd"]
                for row in reader:
                    seen_epochs.add(int(row["epoch_idx"]))
                    assert int(row["subject_id"]) in te_s
                    assert 0.0 <= float(row["p_mdd"]) <= 1.0

        assert seen_epochs == set(range(40))
        # Different folds hold out different participants so remappings differ cleanly per fold
        assert fold_subject_maps[0] != fold_subject_maps[1]

        # Synthetic LOSO with actual CDLMDDNet + DANN
        data_loso = _make_synthetic_npz(
            tmp_path / "modma_loso.npz",
            n_mdd=3,
            n_hc=3,
            epochs_per_subj=4,
            n_channels=16,
            n_samples=64,
            subject_offset=2020010,
        )
        cfg_loso = TrainConfig(
            experiment="cdl_dann_loso",
            model="src.models.cdl_mddnet:CDLMDDNet",
            protocol="loso",
            use_dann=True,
            data_path=str(data_loso),
            runs_dir=str(runs_dir),
            val_fraction=0.25,
            epochs=2,
            batch_size=8,
            device="cpu",
            use_amp=False,
        )
        res_loso = run_experiment(cfg_loso, verbose=False)
        assert len(res_loso["folds"]) == 6
        for k, fold_meta in enumerate(res_loso["folds"]):
            assert fold_meta["use_dann"] is True
            assert len(fold_meta["test_subjects"]) == 1
            assert set(int(s) for s in fold_meta["subject_id_map"].keys()) == set(
                fold_meta["train_subjects"]
            )
            inf_state = torch.load(
                runs_dir / "cdl_dann_loso" / f"fold_{k}" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            CDLMDDNet(n_channels=16, n_samples=64, n_classes=2).load_state_dict(
                inf_state, strict=True
            )

    def test_dann_mid_fold_resume_and_never_uses_inference_checkpoint(self, tmp_path):
        data_path = _make_synthetic_npz(
            tmp_path / "dann_resume.npz",
            n_mdd=4,
            n_hc=4,
            epochs_per_subj=4,
            n_channels=16,
            n_samples=64,
            subject_offset=100,
        )
        with np.load(str(data_path)) as d:
            X, y, sids = d["X"], d["y"], d["subject_id"]
        tr, te = next(kfold_splits(sids, n_folds=4, seed=0, y=y))

        cfg = TrainConfig(
            experiment="dann_resume_exp",
            model="src.models.cdl_mddnet:CDLMDDNet",
            use_dann=True,
            dann_gamma=10.0,
            dann_loss_weight=1.0,
            n_folds=4,
            val_fraction=0.2,
            epochs=4,
            early_stopping_patience=4,
            batch_size=8,
            data_path=str(data_path),
            runs_dir=str(tmp_path / "runs"),
            device="cpu",
            use_amp=False,
        )

        # Simulate interruption at end of epoch 2
        fold0_dir = Path(cfg.runs_dir) / cfg.experiment / "fold_0"
        (Path(cfg.runs_dir) / cfg.experiment).mkdir(parents=True, exist_ok=True)
        # First create valid run_config.json via folds 1 only or test direct train_fold then runner
        def _interrupt_ep2(info):
            if info["epoch"] == 2:
                raise RuntimeError("Simulated Colab GPU disconnect")

        with pytest.raises(RuntimeError, match="Simulated Colab GPU disconnect"):
            train_fold(0, X, y, sids, tr, te, cfg, fold0_dir, epoch_callback=_interrupt_ep2)

        assert (fold0_dir / "model.pt").is_file()
        assert (fold0_dir / TRAIN_CKPT_FILENAME).is_file()
        assert not (fold0_dir / "preds.csv").exists()
        assert not (fold0_dir / "fold_meta.json").exists()

        ckpt_ep2 = torch.load(
            fold0_dir / TRAIN_CKPT_FILENAME, map_location="cpu", weights_only=True
        )
        assert ckpt_ep2["epoch"] == 2
        step_at_ep2 = ckpt_ep2["global_step"]
        assert step_at_ep2 > 0
        assert any(k.startswith("subject_head.") for k in ckpt_ep2["model_state_dict"])

        # Refuse resuming from inference checkpoint if train_checkpoint.pt is missing
        ckpt_backup = fold0_dir / "backup_train_ckpt.pt"
        (fold0_dir / TRAIN_CKPT_FILENAME).replace(ckpt_backup)
        with pytest.raises(FileNotFoundError, match="must not be used to resume training"):
            train_fold(0, X, y, sids, tr, te, cfg, fold0_dir, resume=True)
        ckpt_backup.replace(fold0_dir / TRAIN_CKPT_FILENAME)

        # Refuse resuming DANN training checkpoint with ERM config
        cfg_erm = TrainConfig(**{**cfg.to_dict(), "use_dann": False})
        with pytest.raises(ValueError, match="cannot safely resume"):
            train_fold(0, X, y, sids, tr, te, cfg_erm, fold0_dir, resume=True)

        # Resume fold 0 from epoch 2 -> finishes epochs 3..4 cleanly without resetting global_step
        resumed = train_fold(0, X, y, sids, tr, te, cfg, fold0_dir, resume=True)
        assert resumed["epochs_ran"] == 4
        assert resumed["global_step"] == step_at_ep2 * 2
        with open(fold0_dir / "train_log.csv", newline="", encoding="utf-8") as f:
            epochs_logged = [int(r["epoch"]) for r in csv.DictReader(f)]
        assert epochs_logged == [1, 2, 3, 4]
        assert (fold0_dir / "preds.csv").is_file()
        assert (fold0_dir / "fold_meta.json").is_file()

    def test_erm_no_dann_cli_and_no_mixing_between_dann_and_erm(self, tmp_path):
        data_path = _make_synthetic_npz(
            tmp_path / "erm_vs_dann.npz",
            n_mdd=4,
            n_hc=4,
            epochs_per_subj=4,
            n_channels=16,
            n_samples=64,
        )
        runs_dir = tmp_path / "runs"

        # CLI --no-dann without --experiment automatically isolates to runs/cdl_mddnet_erm/
        runner_main([
            "--model", "src.models.cdl_mddnet:CDLMDDNet",
            "--no-dann",
            "--data", str(data_path),
            "--runs-dir", str(runs_dir),
            "--n-folds", "4",
            "--epochs", "2",
            "--batch-size", "8",
            "--device", "cpu",
            "--no-amp",
            "--quiet",
        ])
        erm_dir = runs_dir / "cdl_mddnet_erm"
        assert (erm_dir / "run_config.json").is_file()
        erm_fold0 = json.loads((erm_dir / "fold_0" / "fold_meta.json").read_text(encoding="utf-8"))
        assert erm_fold0["use_dann"] is False
        assert erm_fold0["subject_id_map"] is None

        # Mixing DANN and ERM in the same experiment directory must fail on resume and overwrite
        cfg_dann = TrainConfig(
            experiment="mixed_guard",
            model="src.models.cdl_mddnet:CDLMDDNet",
            use_dann=True,
            data_path=str(data_path),
            runs_dir=str(runs_dir),
            n_folds=4,
            epochs=2,
            batch_size=8,
            device="cpu",
            use_amp=False,
        )
        run_experiment(cfg_dann, fold_subset=["0"], verbose=False)

        cfg_erm_same_dir = TrainConfig(**{**cfg_dann.to_dict(), "use_dann": False})
        with pytest.raises(ValueError, match="configuration mismatch"):
            run_experiment(cfg_erm_same_dir, fold_subset=["1"], resume=True, verbose=False)
        with pytest.raises(ValueError, match="already contains DANN"):
            run_experiment(cfg_erm_same_dir, verbose=False)
