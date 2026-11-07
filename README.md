# CDL-MDDNet

Lightweight, interpretable EEG model for major depressive disorder detection on MODMA (53 participants: 24 MDD, 29 healthy, 128 channels, 250 Hz, resting state). Goal: under 100k parameters, evaluated on unseen subjects (10-fold and leave-one-subject-out), with channel and band level explanations.

Image and Video Processing course project, 7-person team.

## Setup

Python 3.10+.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Training runs on Google Colab (T4). MODMA is not in the repo: put it under `data/raw/` (git-ignored) and get it from modma.lzu.edu.cn.

## Layout and owners

| Path | What | Owner |
| --- | --- | --- |
| `src/models/cdl_mddnet.py` | CDL-MDDNet network, domain-generalization part | Member 1 |
| `src/data/` | MODMA loading, filtering, ICA, epoching, normalizer | Member 2 |
| `src/training/` | splits, training loop, experiment runner | Member 3 |
| `src/eval/benchmark.py`, `results_table.py`, `update_slides.py`, `tests/`, `.github/` | benchmark, leakage tests, CI, results table, slide update | Member 4 |
| `src/explain/` | Captum channel and band attribution, topomaps | Member 5 |
| `src/models/ms_mddnet.py` | baseline, channel-reduction study | Member 6 |
| `src/eval/metrics.py`, `src/eval/evaluate.py`, `src/app/` | metrics, plots, CLI predictor, demo | Member 7 |

## Contracts

Agree on these before writing against each other. Change them only by telling everyone.

- Processed epochs: `data/processed/modma_epochs.npz` with `X` float32 `(N, C, 1000)`, `y` (0 healthy, 1 MDD), `subject_id` int, `ch_names` list.
- Model: `CDLMDDNet(n_channels, n_samples=1000, n_classes=2)`, input `(B, C, T)`, output logits `(B, 2)`. Any extra dimension is added inside the model.
- Splits (`src/training/splits.py`): `kfold_splits(subject_ids, n_folds=10, seed=0)` and `loso_splits(subject_ids)` yield `(train_idx, test_idx)`, epoch indices into the per-epoch `subject_ids` array.
- Normalizer (`src/data/normalize.py`): `stats = fit_normalizer(train_x)`, `out = apply_normalizer(x, stats)`. Fit on the training fold only.
- Fold outputs: `runs/<experiment>/fold_<k>/preds.csv` with `epoch_idx, subject_id, y_true, p_mdd`, plus `model.pt`.
- Electrode ranking: `channel_ranking.csv` with `channel, score`.
- Results: `runs/results.csv` with `experiment, protocol, level, metric, mean, sd`. Values are fractions (0.87, not 87). Names the results table looks for:
  - `experiment`: `cdl_mddnet` (full 128 channels), `cdl_mddnet_ch<N>` for a reduced montage (`cdl_mddnet_ch32`), plus `ms_mddnet` for the baseline
  - `protocol`: `kfold10` or `loso`
  - `level`: `participant` (majority vote) or `epoch`
  - `metric`: `accuracy`, `sensitivity`, `specificity`, `precision`, `f1`, `auc`

## Workflow

- One branch per task (`m2/ica`, `m3/loso`), pull request into `main`, one review before merge. Member 1 merges.
- CI runs `pytest` on every pull request. Do not merge a red build.
- Never commit data, checkpoints or `runs/`.

## Checks

```
pytest -q                                  # leakage and normalizer tests
python -m src.eval.benchmark --model dummy --channels 32 128
python -m src.eval.benchmark --model src.models.cdl_mddnet:CDLMDDNet --channels 128 32 --out reports/efficiency.csv
python -m src.eval.results_table --results runs/results.csv --eff reports/efficiency.csv
python -m src.eval.update_slides deck.pptx --out deck_measured.pptx
```

`tests/test_no_subject_leakage.py` fails if any participant appears in both train and test in any 10-fold or LOSO split. Tests for splits and the normalizer skip until those modules exist.

## Results table and slides

When Member 7 has produced `runs/results.csv`:

1. Run the benchmark for the final model, writing `reports/efficiency.csv`.
2. Run `results_table`. It writes `reports/results_table.md`: published numbers, deck targets, measured values and met / not met.
3. Run `update_slides` on the deck (needs `python-pptx`). It fills slide 39's table and slide 40's targets and saves a new file. Rows without results are left as they were. Look at the slides before sending them on.

## Status

| Stage | Status |
| --- | --- |
| Repo, CI, leakage tests, benchmark, results table and slide scripts | done (results table and slides wait for real results) |
| Data pipeline | not started |
| Splits and training | not started |
| Model | not started |
| Evaluation | not started |
| Channel reduction | not started |
| Interpretability | not started |
