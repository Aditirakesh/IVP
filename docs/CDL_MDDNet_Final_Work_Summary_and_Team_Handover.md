# CDL-MDDNet: Final Work Summary and Team Handover

- **Project:** CDL-MDDNet — Lightweight, Subject-Independent, Interpretable EEG Model for Major Depressive Disorder (MDD) Detection
- **Course Context:** Image and Video Processing (IVP) Course Project (7-Member Team)
- **Author / Role:** Mooli Hasini — Member 3 (Splits and Training Engineer, DANN Integration, Colab Experiment Execution & Release Packaging)
- **Repository:** https://github.com/Susan5504R/IVP
- **Branch:** `m3/splits-and-training` (latest implementation/results commit: `18792b714238bf6404551fb0484f84c329b2a305`)
- **Published Release:** `v1.0.0` — https://github.com/Susan5504R/IVP/releases/tag/v1.0.0
- **Document Date:** 2026-11-10

---

## Executive Summary

This document summarizes the completed Member 3 engineering work, experimental evaluation, and team handover procedures for **CDL-MDDNet** on the real **MODMA 128-channel resting-state EEG** cohort (`53` participants: `24` MDD, `29` Healthy Controls; `4,002` four-second epochs).

Key verified outcomes:
1. **End-to-End Subject-Independent Training & DANN Pipeline**: Implemented participant-disjoint 10-fold stratified cross-validation and 53-participant Leave-One-Subject-Out (LOSO) evaluation, fold-only normalization, per-fold training-subject remapping, Domain-Adversarial Neural Network (DANN) gradient-reversal training, Empirical Risk Minimization (ERM) baseline support (`--no-dann`), automatic mixed precision (AMP), early stopping, and dual inference/resume checkpointing (`model.pt` and `train_checkpoint.pt`).
2. **Six Completed Full-Dataset Experiments**: Executed four 10-fold experiments (`10` folds each) and two 53-fold LOSO experiments (`53` folds each) on Google Colab Tesla T4 GPUs, covering all `4,002` epochs and `53` participants per experiment with zero participant leakage.
3. **Honest Empirical Findings**: Across all `53` participants, 10-fold ERM baselines achieved `62.26%` accuracy (`33/53` participants; `ROC-AUC = 0.5632` at `lr=0.001` and `0.6207` at `lr=0.0003`), whereas 10-fold DANN achieved `47.17%` accuracy (`ROC-AUC = 0.4770`). Under strict 53-subject LOSO, ERM baseline achieved `37.74%` accuracy (`ROC-AUC = 0.3405`) and LOSO DANN achieved `39.62%` accuracy (`ROC-AUC = 0.3592`). **None of the six configurations met the project's target thresholds (`≥ 85%` 10-fold accuracy, `≥ 80%` LOSO accuracy, `≥ 0.88` AUC), and no clinical utility is claimed.**
4. **Reproducible Handover & Release `v1.0.0`**: Consolidated result tables are committed in [`results/`](../results) (`18792b7`) and all six experiment folders (checkpoints, fold normalizers, logs, and predictions) are archived in `final_CDL_MDDNet_all_experiments.zip` (`60,190,171` bytes) on GitHub Release [`v1.0.0`](https://github.com/Susan5504R/IVP/releases/tag/v1.0.0).

---

## A. Project Overview

### A.1 Project Objective and Problem Statement

Major Depressive Disorder (MDD) diagnosis relies predominantly on clinical interviews and symptom questionnaires (such as PHQ-9), motivating objective physiological biomarkers from non-invasive resting-state electroencephalography (EEG). However, EEG signals exhibit pronounced inter-subject variability, and many published EEG deep learning studies inflate accuracy by splitting overlapping or non-overlapping epochs from the same participant across both training and test sets.

As defined in [`README.md`](../README.md) and [`src/eval/results_table.py`](../src/eval/results_table.py), **CDL-MDDNet** targets:
- **Lightweight parameter footprint**: `< 100,000` trainable parameters.
- **Strict subject-independent evaluation**: Zero participant overlap between training, validation, and test splits under both 10-fold cross-validation (`kfold10`) and Leave-One-Subject-Out (`loso`).
- **Domain generalization**: Training-only subject-adversarial learning via a Gradient Reversal Layer (DANN) to suppress participant-specific EEG features.
- **Interpretability & channel reduction readiness**: Modular support for electrode ranking and reduced-montage training (`≤ 32` channels).

Project acceptance benchmarks tracked in [`src/eval/results_table.py`](../src/eval/results_table.py):

| Evaluation Criterion | Published Literature Reference (in repo) | Project Target |
| --- | --- | --- |
| MODMA 10-fold cross-subject accuracy | `99.33% [1]` | `≥ 85%` (`0.85`) |
| MODMA LOSO accuracy | `99.32% [5]` | `≥ 80%` (`0.80`) |
| Subject-independent ROC-AUC (10-fold) | `99.96% [1]` | `≥ 0.88` |
| Trainable parameters (128 channels) | `34,812 [1]` | `< 100,000` |
| Reduced montage accuracy drop (`≤ 32` channels) | `16 of 128 [6]` | Within `3.0` percentage points of 128-ch |

### A.2 Dataset Details

- **Dataset Name**: MODMA 128-channel Resting-State EEG (`EEG_128channels_resting_lanzhou_2015`, Lanzhou University; official portal: https://modma.lzu.edu.cn).
- **Cohort Composition** ([`src/data/load_modma.py`](../src/data/load_modma.py), [`reports/preprocessing_report.md`](../reports/preprocessing_report.md)):
  - **Total participants**: `53` (`24` MDD with IDs `0201*`, label `1`; `29` Healthy Controls with IDs `0202*` and `0203*`, label `0`).
  - **Recording format**: `128` scalp channels (`E1`–`E128`; 129th Cz reference channel omitted during loading), sampled at `250.0 Hz`.
  - **Processed epoch tensor** (`data/processed/modma_epochs.npz`, `2,049,349,506` bytes / `1.91 GiB`):
    - `X`: shape `(4002, 128, 1000)`, dtype `float32` (`4.0 s` non-overlapping windows at `250 Hz`).
    - `y`: shape `(4002,)`, dtype `int64` (`2,192` Healthy Control epochs; `1,810` MDD epochs).
    - `subject_id`: shape `(4002,)`, dtype `int64` (contiguous participant indices `0..52`).
    - `ch_names`: shape `(128,)` (`['E1', ..., 'E128']`), plus `subject_str_id` shape `(4002,)`.

### A.3 Model Architecture and Methodology

Implemented in [`src/models/cdl_mddnet.py`](../src/models/cdl_mddnet.py) (`CDLMDDNet`), the network processes an EEG epoch `x` of shape `(B, C, 1000)` through five inference stages plus a training-only DANN branch:

1. **Spatial Convolution (`self.spatial`)**: `Conv2d(1, 16, kernel_size=(C, 1), bias=False)` -> `BatchNorm2d(16)` -> `ELU`, mixing signals across all `C` electrodes into `16` spatial filters (`(B, 16, 1, 1000)`).
2. **Multi-Scale Temporal Convolution (`self.branches`)**: Three parallel `Conv2d(16, 12, kernel_size=(1, k), padding=(0, k // 2), bias=False)` branches with temporal kernels `k ∈ {15, 31, 63}` (corresponding to `~60 ms`, `~124 ms`, and `~252 ms` windows at `250 Hz`), concatenated to `36` channels -> `BatchNorm2d(36)` -> `ELU` -> `AvgPool2d((1, 4))` -> `Dropout(0.5)` (`(B, 36, 1, 250)`).
3. **Depthwise-Separable Convolution (`self.dw`, `self.pw`)**: Depthwise `Conv2d(36, 36, kernel_size=(1, 16), padding=(0, 8), groups=36, bias=False)` followed by pointwise `Conv2d(36, 48, kernel_size=1, bias=False)` -> `BatchNorm2d(48)` -> `ELU` -> `AvgPool2d((1, 4))` -> `Dropout(0.5)` (`(B, 48, 1, 62)`).
4. **Attention Pooling over Time (`self.attn`)**: `Conv1d(48, 1, kernel_size=1)` followed by `softmax` over the `62` temporal positions to compute a weighted feature vector `f` of shape `(B, 48)`.
5. **Diagnostic Classifier (`self.classifier`)**: `Linear(48, 2)` outputting raw class logits `(B, 2)` (`0 = Healthy`, `1 = MDD`).
6. **Training-Only Subject-Adversarial Head (`self.subject_head`)**: Activated when `n_subjects > 0`. Passes `f` through `_GradReverse` (scaling backward gradients by `-lambd`) into `Linear(48, 32) -> ReLU -> Linear(32, n_subjects)`.

**Verified Parameter Counts** (via `count_parameters` in [`src/models/cdl_mddnet.py`](../src/models/cdl_mddnet.py)):

| Montage Channels | Inference Parameters (`n_subjects=0`) | Training Parameters (`n_subjects=38`) | Training Parameters (`n_subjects=52`) | Training Parameters (`n_subjects=53`) | Under `100,000` Limit? |
| --- | --- | --- | --- | --- | --- |
| `128` channels | **25,627** | 28,449 | 28,911 | 28,944 | Yes (`25.6%` of limit) |
| `64` channels | **24,603** | 27,425 | 27,887 | 27,920 | Yes (`24.6%` of limit) |
| `32` channels | **24,091** | 26,913 | 27,375 | 27,408 | Yes (`24.1%` of limit) |
| `16` channels | **23,835** | 26,657 | 27,119 | 27,152 | Yes (`23.8%` of limit) |

### A.4 Technologies, Frameworks, and Libraries

- **Language & Core Deep Learning**: Python `3.10+` (CI runs Python `3.10`; local verification ran Python `3.13.2`), PyTorch (`torch>=2.0`; Colab GPU runs executed on `torch 2.14.1+cu130` on an NVIDIA Tesla T4 with `14.56 GB` VRAM).
- **Scientific & Data Stack** ([`requirements.txt`](../requirements.txt)): `numpy`, `scipy` (IIR notch & Butterworth SOS filtering), `pandas`, `scikit-learn` (`StratifiedKFold`, `FastICA`, `roc_auc_score`), `matplotlib`, `mne`, `captum`, `pytest`, `python-pptx`, and optional `h5py` (for MATLAB v7.3 HDF5 `.mat` files in [`src/data/load_modma.py`](../src/data/load_modma.py)).
- **CI/CD & Version Control**: GitHub Actions ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) running `pytest -q` on pull requests and pushes to `main`.

---

## B. My Completed Contributions (Member 3)

All Member 3 implementation work was committed on branch `m3/splits-and-training` across commits `01defe4` (`Complete Member 3 training infrastructure and regression tests`) and `18792b7` (`Add DANN training and verified experiment results`):

1. **DANN Training Implementation** ([`src/training/trainer.py`](../src/training/trainer.py), [`src/training/config.py`](../src/training/config.py), [`src/training/run_experiment.py`](../src/training/run_experiment.py)):
   - Implemented `_remap_training_subjects` to map **only** inner training (`sub_train`) participants to contiguous subject indices `0..K_train-1` per fold, explicitly asserting zero overlap with validation or outer test participants.
   - Integrated step-wise DANN scheduling (`progress = global_step / total_steps`, `lambd = dann_lambda(progress, gamma=cfg.dann_gamma)`) and `model.forward_train(xb, lambd=epoch_lambd)`.
   - Combined inverse-frequency class-weighted cross-entropy (`cls_loss`) with subject cross-entropy (`subj_loss`) via `loss = cls_loss + cfg.dann_loss_weight * subj_loss`, while logging `train_loss`, `train_cls_loss`, `train_subj_loss`, `train_acc`, `train_subj_acc`, and `dann_lambda` every epoch in `train_log.csv`.
   - Enforced classification-only validation and test evaluation via `model(xb)` (`n_subjects=0`), and validated DANN model compatibility (`n_subjects` constructor parameter and callable `forward_train`) in `build_model`.
2. **Baseline ERM Preservation** ([`src/training/run_experiment.py`](../src/training/run_experiment.py), [`src/training/trainer.py`](../src/training/trainer.py)):
   - Added mutually exclusive CLI flags `--use-dann` and `--no-dann` (defaulting to `use_dann=True` from `config/train_config.json`).
   - When `--no-dann` is passed, `CDLMDDNet` is instantiated with `n_subjects=0` (`subject_head=None`), trained with standard classification cross-entropy (`model(xb)`), and automatically written to `runs/cdl_mddnet_erm` if `--experiment` is omitted.
   - Added directory guards preventing accidental mixing or overwriting between DANN and ERM runs or between `kfold10` and `loso` runs inside the same experiment folder.
3. **K-Fold and LOSO Experiment Support** ([`src/training/splits.py`](../src/training/splits.py), [`src/training/run_experiment.py`](../src/training/run_experiment.py)):
   - Implemented participant-level stratified `kfold_splits(subject_ids, n_folds=10, seed=0, y=y)` and deterministic `loso_splits(subject_ids)` (`53` folds sorted by unique `subject_id`).
   - Implemented subject-disjoint inner `make_val_split` (`val_fraction=0.15`, seeded per fold as `cfg.seed + fold_idx`) carved strictly from outer `train_idx`.
   - Added `--folds` subset selection (e.g., `--folds 0-9`, `--folds 0-17`) with cross-batch `training_summary.csv` aggregation, plus `--channels` and `--channel-ranking` / `--top-k-channels` support for Member 5/6 channel-reduction studies.
4. **Checkpoint Saving and Recovery** ([`src/training/trainer.py`](../src/training/trainer.py)):
   - Saved two distinct checkpoints per fold:
     - `model.pt`: Best-validation-loss CPU `state_dict` with `subject_head.*` excluded (`110,170` bytes), strictly compatible with `CDLMDDNet(n_channels, n_samples, n_classes)` (`n_subjects=0`).
     - `train_checkpoint.pt`: Full training-resume state (`333–374` KB) storing `model_state_dict` (including `subject_head`), `optimizer_state_dict`, `scheduler_state_dict`, AMP `scaler_state_dict`, `early_stopper` counters, `global_step`, `epoch`, `config`, and `subject_id_map`.
   - Implemented strict resume validation (`_validate_train_resume_checkpoint` and `_validate_resume_compatibility`), refusal to resume training from inference-only `model.pt`, and automatic cleanup of stale fold artifacts when a non-resumed run fails before producing a valid checkpoint.
5. **Training Configuration** ([`config/train_config.json`](../config/train_config.json), [`src/training/config.py`](../src/training/config.py)):
   - Created `TrainConfig` dataclass with strict validation and JSON serialization (`epochs=60`, `batch_size=64`, `lr=0.001`, `weight_decay=0.0001`, `optimizer="adamw"`, `scheduler="cosine"`, `early_stopping_patience=12`, `early_stopping_min_delta=0.0001`, `class_weighted_loss=true`, `grad_clip_norm=1.0`, `use_dann=true`, `dann_gamma=10.0`, `dann_loss_weight=1.0`, `use_amp=true`, `deterministic=true`).
6. **Unit and Integration Tests** ([`tests/test_splits.py`](../tests/test_splits.py), [`tests/test_training.py`](../tests/test_training.py)):
   - Authored `7` test functions in `tests/test_splits.py` and `26` test functions in `tests/test_training.py` (`33` Member 3 test functions; `63` total test functions across all `7` test files, expanding via `pytest.mark.parametrize` to `93` passing test items verified prior to documentation commit).
   - Covered subject-disjoint splits, inner validation isolation, deterministic seeding, BatchNorm singleton-batch prevention (`N % batch_size == 1`), NaN validation loss handling, mid-fold DANN checkpoint recovery, config mismatch guards, and CLI flags.
7. **Experiment Execution, Prediction Verification, and Result Comparison**:
   - Executed all 6 real-data MODMA experiments on Google Colab Tesla T4 GPUs, verified `4,002` epoch predictions and `53` unique participants per run, and organized the verified result tables in [`results/`](../results) and the `v1.0.0` release archive.

---

## C. Six Completed Experiments

All six experiments ran on the full preprocessed MODMA dataset (`4,002` epochs, `53` participants: `29` HC, `24` MDD; `128` channels, `1000` samples/epoch) on NVIDIA Tesla T4 (`14.56 GB` VRAM, `torch 2.14.1+cu130`, `use_amp=True`, `batch_size=64`, `epochs=60`, `early_stopping_patience=12`, `optimizer=adamw`, `scheduler=cosine`).

### C.1 Verified Participant-Level Results & Confusion Matrices

Source: [`results/final_verified_results.csv`](../results/final_verified_results.csv) and [`results/final_experiment_configurations.csv`](../results/final_experiment_configurations.csv) (cross-checked directly against all `fold_*/preds.csv` files inside `final_CDL_MDDNet_all_experiments.zip`):

| # | Experiment Name | Run Folder (`runs/<folder>` / ZIP root) | Protocol | `lr` | `weight_decay` | `use_dann` | Folds | Accuracy | Balanced Accuracy | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **K-fold baseline** | `cdl_mddnet_kfold10_baseline` | `kfold10` | `0.001` | `0.0001` | `False` | 10 | **0.622642** (`33/53`) | **0.626437** | **0.563218** | 17 | 12 | 8 | 16 |
| 2 | **K-fold DANN** | `cdl_mddnet_kfold10_dann` | `kfold10` | `0.001` | `0.0001` | `True` | 10 | **0.471698** (`25/53`) | **0.470546** | **0.477011** | 14 | 15 | 13 | 11 |
| 3 | **K-fold WD 0.001** | `cdl_mddnet_kfold10_baseline_wd001` | `kfold10` | `0.001` | `0.001` | `False` | 10 | **0.622642** (`33/53`) | **0.626437** | **0.563218** | 17 | 12 | 8 | 16 |
| 4 | **K-fold LR 0.0003** | `cdl_mddnet_kfold10_baseline_lr0003` | `kfold10` | `0.0003` | `0.0001` | `False` | 10 | **0.622642** (`33/53`) | **0.619253** | **0.620690** | 19 | 10 | 10 | 14 |
| 5 | **LOSO baseline** | `cdl_mddnet_loso_baseline` | `loso` | `0.001` | `0.0001` | `False` | 53 | **0.377358** (`20/53`) | **0.380747** | **0.340517** | 10 | 19 | 14 | 10 |
| 6 | **LOSO DANN** | `cdl_mddnet_loso_dann` | `loso` | `0.001` | `0.0001` | `True` | 53 | **0.396226** (`21/53`) | **0.390805** | **0.359195** | 13 | 16 | 16 | 8 |

### C.2 Fold-Level Training Dynamics (from `training_summary.csv` inside ZIP)

| Run Folder | Fold Epoch Acc (`Mean ± SD`) | Fold Epoch AUC (`Mean ± SD`) | Best Val Loss (`Mean ± SD`) | Best Epoch (`Mean`, Range) | Epochs Ran (`Mean`, Range) | Total T4 GPU Time |
| --- | --- | --- | --- | --- | --- | --- |
| `cdl_mddnet_kfold10_baseline` | `0.5300 ± 0.1066` | `0.5512 ± 0.1683` | `0.8512 ± 0.1923` | `1.20` (`1..3`) | `13.20` (`13..15`) | `165.4 s` (`2.76 min`) |
| `cdl_mddnet_kfold10_dann` | `0.4696 ± 0.1441` | `0.4259 ± 0.1989` | `0.7571 ± 0.1269` | `2.60` (`1..13`) | `14.60` (`13..25`) | `186.5 s` (`3.11 min`) |
| `cdl_mddnet_kfold10_baseline_wd001` | `0.5292 ± 0.1060` | `0.5510 ± 0.1685` | `0.8512 ± 0.1923` | `1.20` (`1..3`) | `13.20` (`13..15`) | `168.0 s` (`2.80 min`) |
| `cdl_mddnet_kfold10_baseline_lr0003` | `0.5726 ± 0.1236` | `0.5644 ± 0.1537` | `0.7609 ± 0.0894` | `2.50` (`1..13`) | `14.50` (`13..25`) | `191.7 s` (`3.19 min`) |
| `cdl_mddnet_loso_baseline` | `0.4205 ± 0.3166` | N/A (1 test class/fold) | `0.8397 ± 0.1820` | `1.42` (`1..7`) | `13.42` (`13..19`) | `958.9 s` (`15.98 min`) |
| `cdl_mddnet_loso_dann` | `0.4225 ± 0.3420` | N/A (1 test class/fold) | `0.7890 ± 0.1973` | `2.62` (`1..21`) | `14.62` (`13..33`) | `1,081.0 s` (`18.02 min`) |

### C.3 Output Folder Paths and File Completeness

When extracted into `runs/` (the repository's standard output directory), the six folders reside at:
- `runs/cdl_mddnet_kfold10_baseline/` (`64` files: `run_config.json`, `training_summary.csv`, `combined_preds.csv`, `participant_level_preds.csv`, plus `fold_0`..`fold_9` x `6` files)
- `runs/cdl_mddnet_kfold10_dann/` (`64` files: `run_config.json`, `training_summary.csv`, `combined_preds.csv`, `participant_level_preds.csv`, plus `fold_0`..`fold_9` x `6` files)
- `runs/cdl_mddnet_kfold10_baseline_wd001/` (`62` files: `run_config.json`, `training_summary.csv`, plus `fold_0`..`fold_9` x `6` files; omits `combined_preds.csv` and `participant_level_preds.csv`)
- `runs/cdl_mddnet_kfold10_baseline_lr0003/` (`61` files: `run_config.json`, `training_summary.csv`, plus `fold_0`..`fold_9`; omits `combined_preds.csv` and `participant_level_preds.csv`, and `fold_9` lacks `train_log.csv`)
- `runs/cdl_mddnet_loso_baseline/` (`322` files: `run_config.json`, `training_summary.csv`, `combined_preds.csv`, `participant_level_preds.csv`, plus `fold_0`..`fold_52` x `6` files)
- `runs/cdl_mddnet_loso_dann/` (`321` files: `run_config.json`, `training_summary.csv`, `combined_preds.csv`, `participant_level_preds.csv`, plus `fold_0`..`fold_52`; `fold_47` lacks `model.pt` while retaining `preds.csv`, `fold_meta.json`, `normalizer_stats.npz`, `train_log.csv`, and `train_checkpoint.pt`)

---

## D. Results and Interpretation

### D.1 Paired Comparison and 95% Confidence Intervals

From [`results/final_paired_comparison_53_participants.csv`](../results/final_paired_comparison_53_participants.csv), each candidate experiment is compared against the `53`-participant `K-fold baseline` (`Acc = 0.6226`, `ROC-AUC = 0.5632`):

| Comparison vs. K-Fold Baseline | Accuracy | Accuracy Diff (`95% CI`) | ROC-AUC | ROC-AUC Diff (`95% CI`) | Discordant Pairs (`Base-only / Cand-only`) | McNemar `p`-value |
| --- | --- | --- | --- | --- | --- | --- |
| **K-fold baseline (ref)** | `0.6226` | `0.0000` (`[0.0000, 0.0000]`) | `0.5632` | `0.0000` (`[0.0000, 0.0000]`) | `0 / 0` | `1.0000` |
| **K-fold WD 0.001** | `0.6226` | `0.0000` (`[0.0000, 0.0000]`) | `0.5632` | `0.0000` (`[0.0000, 0.0000]`) | `0 / 0` | `1.0000` |
| **K-fold LR 0.0003** | `0.6226` | `0.0000` (`[-0.1321, +0.1321]`) | `0.6207` | `+0.0575` (`[-0.0647, +0.1925]`) | `6 / 6` | `1.0000` |
| **K-fold DANN** | `0.4717` | `-0.1509` (`[-0.3208, +0.0189]`) | `0.4770` | `-0.0862` (`[-0.2788, +0.0949]`) | `15 / 7` | `0.1338` |
| **LOSO baseline** | `0.3774` | `-0.2453` (`[-0.4151, -0.0755]`) | `0.3405` | `-0.2227` (`[-0.4182, -0.0273]`) | `18 / 5` | `0.0106` |
| **LOSO DANN** | `0.3962` | `-0.2264` (`[-0.3962, -0.0377]`) | `0.3592` | `-0.2040` (`[-0.4340, +0.0288]`) | `19 / 7` | `0.0290` |

### D.2 Statistical Significance Limitations

- **Small Participant Sample Size (`N = 53`)**: Because participant-level evaluation aggregates into `53` binary decisions (`24` MDD, `29` HC), a single participant flip changes overall accuracy by `1 / 53 ≈ 1.89%`. Consequently, `95%` bootstrap confidence intervals are roughly `26` to `46` percentage points wide.
- **No Statistically Significant Improvement Over Baseline**: Neither `LR 0.0003` (`p = 1.0`) nor `WD 0.001` (`p = 1.0`) improved participant-level accuracy over the `K-fold baseline`. The only statistically significant differences (`p < 0.05`) are the **degradations** from `10-fold` to `LOSO` (`p = 0.0106` for LOSO baseline; `p = 0.0290` for LOSO DANN).
- **Sensitivity/Specificity Trade-Off**: In `K-fold baseline`, sensitivity is `16/24 = 66.7%` and specificity is `17/29 = 58.6%` (`BalAcc = 62.64%`). Lowering learning rate to `0.0003` shifted specificity up to `19/29 = 65.5%` while lowering sensitivity to `14/24 = 58.3%` (`BalAcc = 61.93%`), improving rank ordering (`ROC-AUC = 0.6207` vs `0.5632`).

### D.3 Honest Interpretation of DANN Performance

- **10-Fold DANN Degraded Classification**: Enabling DANN under 10-fold CV reduced participant accuracy from `62.26%` to `47.17%` (`-15.09%`) and ROC-AUC from `0.5632` to `0.4770` (`-0.0862`).
- **LOSO DANN Marginal Gain Remains Below Chance**: Under LOSO, DANN achieved `39.62%` accuracy (`ROC-AUC = 0.3592`) vs `37.74%` (`ROC-AUC = 0.3405`) for LOSO ERM (`+1.89%` accuracy, `+0.0187` AUC), but both remain substantially below random-chance (`50%` / `0.50`).
- **Mechanistic Explanation**: With only `40–41` inner training participants in `10-fold` (and `44` in `LOSO`) and a compact `48`-dimensional bottleneck, adversarial gradient reversal (`dann_loss_weight = 1.0`, `gamma = 10.0`) appears to suppress subject-specific features that are partially confounded with diagnostic traits in resting-state EEG, destabilizing the shared encoder rather than isolating pure MDD biomarkers.

### D.4 Overfitting, Generalization Limitations, and Non-Clinical Disclaimer

- **Rapid Overfitting / Early-Stopping Collapse**: Across all six experiments, early stopping selected weights at `best_epoch = 1` to `3` in most folds (mean `best_epoch` `1.20` for baseline, `2.60` for DANN), terminating after `13–15` epochs on average. Training accuracy increased steadily while validation loss on unseen validation participants rose after epoch `1–2`.
- **Sub-0.50 LOSO AUC**: Both LOSO models yielded `ROC-AUC < 0.36`, indicating systematic calibration/decision-boundary inversion when generalizing to single held-out subjects under small inner validation sets (`~8` validation subjects per LOSO fold).
- **No Clinical Validation**: All results reflect a single-site research dataset (`53` subjects) and **must not** be interpreted as clinically validated or suitable for medical screening or diagnostic use.

---

## E. Dataset and Preprocessing

### E.1 Dataset Structure and Participant Count

- **Cohort**: `53` participants (`24` MDD `0201*`, `29` HC `0202*`/`0203*`), `300.4 s` to `335.5 s` per recording, yielding `75` to `83` non-overlapping 4-second epochs per participant (`4,002` epochs total; `2,192` HC, `1,810` MDD). Full per-subject breakdown is documented in [`reports/preprocessing_report.md`](../reports/preprocessing_report.md).
- **NPZ Schema** (`data/processed/modma_epochs.npz`, loaded with `allow_pickle=False` in [`load_processed_dataset`](../src/training/run_experiment.py)):
  - `X`: `(4002, 128, 1000)` `float32`
  - `y`: `(4002,)` `int64` (`0` = HC, `1` = MDD)
  - `subject_id`: `(4002,)` `int64` (`0..52`, alphabetical mapping from 8-digit MODMA IDs via `get_subject_mapping`)
  - `ch_names`: `(128,)` `['E1', ..., 'E128']`
  - `subject_str_id`: `(4002,)` 8-digit string IDs

### E.2 Preprocessing Pipeline

Implemented in [`src/data/load_modma.py`](../src/data/load_modma.py) and [`src/data/preprocess.py`](../src/data/preprocess.py):
1. **Channel & Rate Standardization**: Load MATLAB v5/v7 or v7.3 `.mat` recording, select the `128` scalp channels (`E1..E128`, dropping the 129th Cz reference channel), and verify/resample to `250.0 Hz`.
2. **50 Hz Notch Filtering** (`apply_notch_filter`): Zero-phase forward-backward IIR notch filter (`w0=50.0 Hz`, `Q=30.0`).
3. **0.5–45.0 Hz Bandpass Filtering** (`apply_bandpass_filter`): 4th-order zero-phase Butterworth Second-Order Sections (`sosfiltfilt`) isolating delta through beta/low-gamma bands.
4. **FastICA Artifact Rejection** (`apply_ica_cleaning`): `32`-component `FastICA` (`random_state=42`); automatically zeros out up to `3` components exceeding Fisher kurtosis `|k| > 5.0`.
5. **Non-Overlapping Epoching** (`create_epochs`): Segments continuous cleaned EEG into `4.0 s` (`1000`-sample) windows with `overlap=0.0`.

### E.3 Train/Validation/Test Separation and Leakage Prevention

- **Outer Split**: [`kfold_splits`](../src/training/splits.py) (`10` folds, `~5–6` test participants/fold) or [`loso_splits`](../src/training/splits.py) (`53` folds, `1` test participant/fold) partitions at the **participant level** (`subject_id`).
- **Inner Validation Split**: [`make_val_split`](../src/training/splits.py) carves `15%` (`val_fraction=0.15`) of *outer training participants* (stratified across classes) for early stopping and checkpoint selection, leaving `~85%` of outer training participants (`sub_train`) for gradient updates.
- **Fold-Only Normalization**: [`fit_normalizer`](../src/data/normalize.py) computes per-channel mean and standard deviation `(1, 128, 1)` **exclusively on `X[sub_train_idx]`**. [`apply_normalizer`](../src/data/normalize.py) applies those training statistics to `X_sub_train`, `X_val`, and `X_test`, saving `normalizer_stats.npz` inside each fold folder.
- **Automated Leakage Verification**: [`tests/test_no_subject_leakage.py`](../tests/test_no_subject_leakage.py) and [`tests/test_normalizer.py`](../tests/test_normalizer.py) run in CI on every PR.

### E.4 Participant-Level Evaluation

Each fold exports epoch-level test predictions (`epoch_idx,subject_id,y_true,p_mdd`) in `fold_<k>/preds.csv`. Because every participant is held out in the test set of exactly one fold, pooling across all `10` (or `53`) folds yields one out-of-fold prediction for all `4,002` epochs. Participant-level probability `p_mdd(subject)` is computed as the mean predicted MDD probability across that participant's epochs (`75–83` epochs), with binary diagnosis `y_pred = 1` if `p_mdd(subject) >= 0.5` else `0` (verified in `participant_level_preds.csv` and against `final_verified_results.csv`).

### E.5 Dataset Access Requirements

- Raw MODMA EEG recordings must be requested/downloaded directly from the official MODMA repository (https://modma.lzu.edu.cn) and placed in `src/data/raw/` (or `data/raw/`).
- Alternatively, teammates can obtain the preprocessed `modma_epochs.npz` (`2,049,349,506` bytes) directly from the team's shared Google Drive / local transfer and save it to `data/processed/modma_epochs.npz` (auto-detected by [`resolve_data_path`](../src/training/config.py)).

---

## F. GitHub and Release

* **Repository URL**: https://github.com/Susan5504R/IVP
* **Active Working Branch**: `m3/splits-and-training`
* **Key Commits on `m3/splits-and-training`**:
  * `01defe465cea47b4f3a279dd48a653b001b33d41` — *Complete Member 3 training infrastructure and regression tests*
  * `18792b714238bf6404551fb0484f84c329b2a305` — *Add DANN training and verified experiment results* (current `v1.0.0` release target)
  * *(Note: `origin/main` is currently at `5d5d53407866ba16b6f88f4e652c56fa2b03f9dd`; `m3/splits-and-training` is ahead and ready for PR merge.)*
* **Published GitHub Release (`v1.0.0`)**:
  * **Release URL**: https://github.com/Susan5504R/IVP/releases/tag/v1.0.0
  * **Release Title**: `CDL-MDDNet – Final Experiment Results and Checkpoints`
  * **Direct Asset Download URL**: https://github.com/Susan5504R/IVP/releases/download/v1.0.0/final_CDL_MDDNet_all_experiments.zip
  * **Archive Filename**: `final_CDL_MDDNet_all_experiments.zip`
  * **Exact Archive Size**: `60,190,171` bytes (`57.40 MiB`)
  * **Verified SHA-256 Checksum**:
    ```text
    cfe747a9b864554df9bd34066bb3c2bdea3f4a012cc2d1e2e196117bdede2146
    ```
* **Archive Contents** (`897` files total):
  * Root consolidated CSVs (`3` files, identical to [`results/`](../results)):
    - `final_verified_results.csv` (`703` B)
    - `final_experiment_configurations.csv` (`491` B)
    - `final_paired_comparison_53_participants.csv` (`1,113` B)
  * Six experiment directories (`894` files total):
    - `cdl_mddnet_kfold10_baseline/` (`64` files: `10` folds)
    - `cdl_mddnet_kfold10_dann/` (`64` files: `10` folds)
    - `cdl_mddnet_kfold10_baseline_wd001/` (`62` files: `10` folds)
    - `cdl_mddnet_kfold10_baseline_lr0003/` (`61` files: `10` folds)
    - `cdl_mddnet_loso_baseline/` (`322` files: `53` folds)
    - `cdl_mddnet_loso_dann/` (`321` files: `53` folds)
* **Why the Dataset Is Excluded from Git and Release**:
  1. **Size**: Preprocessed `modma_epochs.npz` is `2.05 GB` (`1.91 GiB`) and raw `.mat` files are multi-gigabyte, exceeding GitHub's `100 MB` per-file limit and degrading repository cloning.
  2. **Repository Hygiene**: [`.gitignore`](../.gitignore) explicitly ignores `data/raw/*`, `data/processed/*`, `runs/*`, `*.pt`, `*.npz`, `*.edf`, `*.mat`, and `*.zip`.
  3. **Data Governance**: MODMA requires dataset access registration through Lanzhou University (https://modma.lzu.edu.cn).

---

## G. Instructions for Other Team Members

### Step 1: Clone the Repository and Check Out `m3/splits-and-training`

```bash
git clone https://github.com/Susan5504R/IVP.git CDL-MDDNet
cd CDL-MDDNet
git checkout m3/splits-and-training
```
*(Once the pull request into `main` is merged, `git checkout main && git pull` will have the same files.)*

### Step 2: Download the Release ZIP (`v1.0.0`)

Download `final_CDL_MDDNet_all_experiments.zip` (`57.40 MiB`) from https://github.com/Susan5504R/IVP/releases/tag/v1.0.0 and save it inside `experiment_artifacts/` (ignored by `.gitignore`):

```bash
mkdir -p experiment_artifacts
curl -L -o experiment_artifacts/final_CDL_MDDNet_all_experiments.zip \
  https://github.com/Susan5504R/IVP/releases/download/v1.0.0/final_CDL_MDDNet_all_experiments.zip
```

Optional checksum verification:
```bash
# Linux / macOS / Git Bash
sha256sum experiment_artifacts/final_CDL_MDDNet_all_experiments.zip
# Expected: cfe747a9b864554df9bd34066bb3c2bdea3f4a012cc2d1e2e196117bdede2146
```
```powershell
# Windows PowerShell
Get-FileHash experiment_artifacts\final_CDL_MDDNet_all_experiments.zip -Algorithm SHA256
```

### Step 3 & Step 4: Extract Experiment Folders and Place Them in `runs/`

Extract the six experiment directories into `runs/` so they follow the repository's `runs/<experiment>/fold_<k>/...` layout (ignored by `.gitignore`) without duplicating the three CSVs already tracked in `results/`:

```bash
# Linux / macOS / Git Bash
unzip experiment_artifacts/final_CDL_MDDNet_all_experiments.zip "cdl_mddnet_*" -d runs/
```
```powershell
# Windows PowerShell
Expand-Archive experiment_artifacts\final_CDL_MDDNet_all_experiments.zip -DestinationPath runs\
Remove-Item runs\final_*.csv -ErrorAction SilentlyContinue
```

Expected directory layout after extraction:
```text
CDL-MDDNet/
├── results/
│   ├── final_verified_results.csv
│   ├── final_experiment_configurations.csv
│   └── final_paired_comparison_53_participants.csv
├── experiment_artifacts/
│   └── final_CDL_MDDNet_all_experiments.zip
└── runs/
    ├── cdl_mddnet_kfold10_baseline/
    ├── cdl_mddnet_kfold10_dann/
    ├── cdl_mddnet_kfold10_baseline_wd001/
    ├── cdl_mddnet_kfold10_baseline_lr0003/
    ├── cdl_mddnet_loso_baseline/
    └── cdl_mddnet_loso_dann/
```

### Step 5: Understand the Artifacts Inside Each Experiment Folder

| Artifact Path | Format | Purpose |
| --- | --- | --- |
| `run_config.json` | JSON | Exact hyperparameters, dataset metadata, channel list, hardware info, and reproducibility status. |
| `training_summary.csv` | CSV | Per-fold `best_epoch`, `epochs_ran`, `best_val_loss`, `best_val_acc`, `test_loss`, `test_acc`, `test_auc`, `train_time_s`. |
| `combined_preds.csv` *(4 runs)* | CSV | Pooled out-of-fold epoch predictions (`4,002` rows: `epoch_idx,subject_id,y_true,p_mdd`). |
| `participant_level_preds.csv` *(4 runs)* | CSV | Aggregated participant-level predictions (`53` rows: `subject_id,y_true,p_mdd,n_epochs,y_pred`). |
| `fold_<k>/model.pt` | PyTorch `state_dict` | Best-validation-loss inference weights (`subject_head` stripped; `110,170` B). Use for inference & attribution. |
| `fold_<k>/train_checkpoint.pt` | PyTorch dict | Full training-resume state (model + `subject_head` + optimizer + scheduler + AMP scaler + early stopper). |
| `fold_<k>/normalizer_stats.npz` | NumPy NPZ | Fold-only z-score `mean` `(1,128,1)`, `std` `(1,128,1)`, `eps`, and `n_channels`. |
| `fold_<k>/preds.csv` | CSV | Held-out test predictions for fold `k` (`epoch_idx,subject_id,y_true,p_mdd`). |
| `fold_<k>/train_log.csv` | CSV | Per-epoch training/validation loss, accuracy, AUC, DANN lambda, learning rate, and epoch duration. |
| `fold_<k>/fold_meta.json` | JSON | Fold split participant IDs (`train_subjects`, `val_subjects`, `test_subjects`), DANN mapping, and summary metrics. |

### Step 6: Access the Final Verified CSV Results

The three consolidated CSVs are already committed in [`results/`](../results):
- [`results/final_verified_results.csv`](../results/final_verified_results.csv) — Accuracy, balanced accuracy, ROC-AUC, `TN/FP/FN/TP`, and integrity status for all 6 experiments.
- [`results/final_experiment_configurations.csv`](../results/final_experiment_configurations.csv) — Exact protocol, learning rate, weight decay, `use_dann`, optimizer, and scheduler per experiment.
- [`results/final_paired_comparison_53_participants.csv`](../results/final_paired_comparison_53_participants.csv) — Paired differences, 95% confidence intervals, discordant participant counts, and McNemar p-values vs. `K-fold baseline`.

### Step 7: Set Up Environment and Run the Test Suite

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
*(If running CI-minimal tests without MNE/Captum/PPTX: `pip install torch numpy scikit-learn pytest && pytest -q`.)*

### Step 8: Load Saved Fold Checkpoints for Inference

There is currently **no standalone CLI prediction script** (`predict.py` or `evaluate.py`) in the repository; however, programmatic Python inference is directly supported using [`CDLMDDNet`](../src/models/cdl_mddnet.py) and [`apply_normalizer`](../src/data/normalize.py):

```python
import numpy as np
import torch
from src.data.normalize import apply_normalizer
from src.models.cdl_mddnet import CDLMDDNet

fold_dir = "runs/cdl_mddnet_kfold10_baseline/fold_0"

# 1. Load fold-specific training normalization statistics
with np.load(f"{fold_dir}/normalizer_stats.npz", allow_pickle=False) as ns:
    stats = {
        "mean": ns["mean"],
        "std": ns["std"],
        "eps": float(ns["eps"]),
        "n_channels": int(ns["n_channels"]),
    }

# 2. Load inference-only model weights (n_subjects=0)
model = CDLMDDNet(n_channels=stats["n_channels"], n_samples=1000, n_classes=2)
state_dict = torch.load(f"{fold_dir}/model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.eval()

# 3. Normalize raw epochs of shape (B, 128, 1000) and run forward inference
# x_raw = ...  # np.ndarray of shape (B, 128, 1000), float32
x_norm = torch.from_numpy(apply_normalizer(x_raw, stats))
with torch.no_grad():
    logits = model(x_norm)
    probs_mdd = torch.softmax(logits, dim=-1)[:, 1].numpy()
```

> **Caution**: In `runs/cdl_mddnet_loso_dann/fold_47/`, `model.pt` is absent from the `v1.0.0` ZIP (`52/53` LOSO DANN `model.pt` files are present). Any batch inference loop over `cdl_mddnet_loso_dann` should check `Path(f"{fold_dir}/model.pt").is_file()` before loading.

### Step 9: Integrate `m3/splits-and-training` into `main` via Pull Request

1. Open https://github.com/Susan5504R/IVP/pull/new/m3/splits-and-training.
2. Verify **base: `main`** `<-` **compare: `m3/splits-and-training`**.
3. Wait for the GitHub Actions `tests` check ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) to pass (`pytest -q`).
4. Complete team review and merge into `main` (per [`README.md`](../README.md), Member 1 handles final merges).
5. **Never commit** `data/`, `runs/`, `experiment_artifacts/*.zip`, `*.pt`, or `*.npz` files.

### Step 10: Which Experiments Do NOT Need to Be Rerun

- **Do NOT rerun** the six full 128-channel experiments (`cdl_mddnet_kfold10_baseline`, `cdl_mddnet_kfold10_dann`, `cdl_mddnet_kfold10_baseline_wd001`, `cdl_mddnet_kfold10_baseline_lr0003`, `cdl_mddnet_loso_baseline`, `cdl_mddnet_loso_dann`) — their checkpoints, fold normalizers, logs, and predictions are already verified and archived in `v1.0.0`.
- **Only run new experiments** for unexecuted downstream studies:
  - Reduced-channel runs (`cdl_mddnet_ch32`, `cdl_mddnet_ch16`, etc.) once Member 5 produces `channel_ranking.csv`.
  - Baseline architecture comparison runs (`ms_mddnet`) once Member 6 implements `src/models/ms_mddnet.py`.
  - Optionally `--folds 47` on `cdl_mddnet_loso_dann` only if `fold_47/model.pt` is needed.

---

## H. Remaining Team Responsibilities

| Task / Deliverable | Owner (per `README.md`) | Current Status in Repository | Action Still Required |
| --- | --- | --- | --- |
| **Merge `m3/splits-and-training` into `main`** | Team / Member 1 | **Not yet merged** (`origin/main` is at `5d5d534`) | Review PR, verify green CI, and merge into `main`. |
| **Channel & Band Interpretability (`src/explain/`)** | Member 5 | **Not implemented** (only empty [`src/explain/__init__.py`](../src/explain/__init__.py) exists) | Implement Captum attribution using saved `fold_<k>/model.pt` + `normalizer_stats.npz`, generate topomaps and `channel_ranking.csv` (`channel,score`). |
| **Comparison Model & Channel Reduction (`src/models/ms_mddnet.py`)** | Member 6 | **Not implemented** (`src/models/ms_mddnet.py` does not exist) | Implement `ms_mddnet` baseline and run reduced-channel `CDLMDDNet` experiments (`--channel-ranking channel_ranking.csv --top-k-channels 32`). |
| **Unified Evaluation Metrics, Plots & Demo (`src/eval/metrics.py`, `evaluate.py`, `src/app/`)** | Member 7 | **Not implemented** (`metrics.py` and `evaluate.py` do not exist; [`src/app/__init__.py`](../src/app/__init__.py) is empty) | Aggregate fold `preds.csv` into `runs/results.csv` with columns `experiment,protocol,level,metric,mean,sd` (fractions `0..1`), generate ROC/confusion-matrix figures in `reports/figures/`, and build CLI/demo predictor if planned. |
| **Efficiency Benchmark & Slide Deck Update** | Member 4 / Team | Scripts exist ([`benchmark.py`](../src/eval/benchmark.py), [`results_table.py`](../src/eval/results_table.py), [`update_slides.py`](../src/eval/update_slides.py)), but `reports/efficiency.csv`, `runs/results.csv`, and `deck.pptx` **do not exist** | Run `python -m src.eval.benchmark --model src.models.cdl_mddnet:CDLMDDNet --channels 128 32 --out reports/efficiency.csv`, generate `reports/results_table.md`, and update presentation slides. |
| **Final Presentation & Course Submission** | All Team Members | **Not verified / Pending** | Integrate final figures, honest cross-subject results, and submit final report/slides. |

---

## I. Limitations, Reproducibility Notes, and References

### I.1 Technical Limitations & Caveats

1. **Colab Paths in Archived `run_config.json`**: The archived `run_config.json` files record Colab paths (`"data_path": "/content/modma_epochs.npz"` and `"runs_dir": "/content/drive/MyDrive/CDL-MDDNet/runs"`). Passing `--resume` locally against extracted Colab runs will raise a `data_path` mismatch error in `_validate_resume_compatibility`; run new experiments in fresh `--experiment` directories.
2. **Schema Difference Between `results/*.csv` and `src/eval/results_table.py`**: [`results/final_verified_results.csv`](../results/final_verified_results.csv) uses columns `Experiment,Folds,Epoch predictions,Participants,Accuracy,Balanced accuracy,ROC-AUC,TN,FP,FN,TP,Integrity`, whereas [`src/eval/results_table.py`](../src/eval/results_table.py) reads `runs/results.csv` with columns `experiment,protocol,level,metric,mean,sd`. Member 7's evaluation script should emit `runs/results.csv` in that contract format.
3. **Post-Processing Script Not Tracked in `src/`**: The Colab post-processing notebook/script that generated `combined_preds.csv`, `participant_level_preds.csv`, and the bootstrap CIs / McNemar p-values in [`results/final_paired_comparison_53_participants.csv`](../results/final_paired_comparison_53_participants.csv) is not checked into `src/`. All participant-level accuracy, balanced accuracy, ROC-AUC, and confusion matrix values in [`results/final_verified_results.csv`](../results/final_verified_results.csv) were independently re-verified directly from `fold_*/preds.csv`; bootstrap CI / McNemar generation script internals are **Not verified** from repo source alone.
4. **Archived File Edge Cases**: `cdl_mddnet_loso_dann/fold_47/model.pt` is missing from `final_CDL_MDDNet_all_experiments.zip` (`52/53` present), and `cdl_mddnet_kfold10_baseline_lr0003/fold_9/train_log.csv` is missing (`9/10` present).

### I.2 Reproducibility & Dependency Notes

- **Seeding & Hardware Nondeterminism**: [`set_seed`](../src/training/config.py) fixes Python, NumPy, and PyTorch CPU/CUDA seeds (`seed=42`, `fold_seed = 42 + fold_idx`, `split_seed=0`), sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and enables `torch.use_deterministic_algorithms(True, warn_only=True)`. As explicitly logged in `run_config.json["reproducibility"]["limitations"]`, `warn_only=True` allows certain GPU reduction kernels to remain non-deterministic across hardware/driver versions.
- **Dependencies**: Installing [`requirements.txt`](../requirements.txt) provides all core libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `mne`, `torch>=2.0`, `captum`, `pytest`, `python-pptx`). Add `h5py` if loading MATLAB v7.3 `.mat` files, and `python-docx` if regenerating `.docx` documentation.

### I.3 Items Explicitly Marked "Not Verified"

- Live GitHub Pull Request creation/merge status or downstream commits on branches other than `main` and `m3/splits-and-training`: **Not verified** (only `main` and `m3/splits-and-training` exist on `origin`).
- Downstream Member 5, Member 6, and Member 7 code not yet pushed to `origin`: **Not verified**.
- External cohort or clinical generalization beyond the 53-participant MODMA resting-state dataset: **Not verified** (and not claimed).

### I.4 Project Links & Repository References

- **GitHub Repository**: https://github.com/Susan5504R/IVP
- **Branch (`m3/splits-and-training`)**: https://github.com/Susan5504R/IVP/tree/m3/splits-and-training
- **Create Pull Request (`m3/splits-and-training` -> `main`)**: https://github.com/Susan5504R/IVP/pull/new/m3/splits-and-training
- **GitHub Release `v1.0.0`**: https://github.com/Susan5504R/IVP/releases/tag/v1.0.0
- **Release Asset (`final_CDL_MDDNet_all_experiments.zip`)**: https://github.com/Susan5504R/IVP/releases/download/v1.0.0/final_CDL_MDDNet_all_experiments.zip
- **MODMA Dataset Portal**: https://modma.lzu.edu.cn
- **Repository Source & Result Paths**:
  - Model: [`src/models/cdl_mddnet.py`](../src/models/cdl_mddnet.py)
  - Data Pipeline: [`src/data/load_modma.py`](../src/data/load_modma.py), [`src/data/preprocess.py`](../src/data/preprocess.py), [`src/data/normalize.py`](../src/data/normalize.py)
  - Training & Splits: [`src/training/config.py`](../src/training/config.py), [`src/training/splits.py`](../src/training/splits.py), [`src/training/trainer.py`](../src/training/trainer.py), [`src/training/run_experiment.py`](../src/training/run_experiment.py)
  - Configuration & Results: [`config/train_config.json`](../config/train_config.json), [`results/final_verified_results.csv`](../results/final_verified_results.csv), [`results/final_experiment_configurations.csv`](../results/final_experiment_configurations.csv), [`results/final_paired_comparison_53_participants.csv`](../results/final_paired_comparison_53_participants.csv)
  - Tests & CI: [`tests/test_splits.py`](../tests/test_splits.py), [`tests/test_training.py`](../tests/test_training.py), [`tests/test_no_subject_leakage.py`](../tests/test_no_subject_leakage.py), [`.github/workflows/tests.yml`](../.github/workflows/tests.yml)
