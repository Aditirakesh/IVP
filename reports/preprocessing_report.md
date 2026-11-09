# CDL-MDDNet: Preprocessing Report (Member 2)

## 1. Preprocessing Configuration

- **Dataset**: MODMA 128-channel Resting-State EEG
- **Sampling Frequency**: 250.0 Hz
- **Powerline Notch Filter**: 50.0 Hz (zero-phase IIR notch, Q=30)
- **Bandpass Filter**: 0.5 - 45.0 Hz (4th-order zero-phase Butterworth SOS)
- **Artifact Removal**: FastICA with Fisher kurtosis thresholding and frontal projection
- **Epoch Length**: 4.0 seconds (exactly 1000 samples at 250 Hz)
- **Epoch Overlap**: 0.0% (non-overlapping)
- **Total Participants Processed**: 53
- **Total Epochs Generated**: 4002

## 2. Participant Preprocessing Breakdown

| Subject ID | Diagnosis | Label | Raw Duration (s) | Epochs Extracted | ICA Excluded Components | Status |
|---|---|---|---|---|---|---|
| `02010002` | MDD | 1 | 300.8s | 75 | 17, 21, 23 | Cleaned & Epoched |
| `02010004` | MDD | 1 | 301.4s | 75 | 2, 21, 14 | Cleaned & Epoched |
| `02010005` | MDD | 1 | 300.9s | 75 | 22, 15, 9 | Cleaned & Epoched |
| `02010006` | MDD | 1 | 314.6s | 78 | 21, 9, 0 | Cleaned & Epoched |
| `02010008` | MDD | 1 | 330.6s | 82 | 24, 10, 15 | Cleaned & Epoched |
| `02010010` | MDD | 1 | 300.4s | 75 | 16, 21, 11 | Cleaned & Epoched |
| `02010011` | MDD | 1 | 300.5s | 75 | 21, 15, 13 | Cleaned & Epoched |
| `02010012` | MDD | 1 | 300.5s | 75 | 15, 22, 10 | Cleaned & Epoched |
| `02010013` | MDD | 1 | 300.7s | 75 | 28, 10, 19 | Cleaned & Epoched |
| `02010015` | MDD | 1 | 302.6s | 75 | 23, 1, 24 | Cleaned & Epoched |
| `02010016` | MDD | 1 | 300.8s | 75 | 22, 8, 29 | Cleaned & Epoched |
| `02010018` | MDD | 1 | 301.4s | 75 | 8, 0, 9 | Cleaned & Epoched |
| `02010019` | MDD | 1 | 301.6s | 75 | 29, 8, 5 | Cleaned & Epoched |
| `02010021` | MDD | 1 | 300.5s | 75 | 0, 14, 4 | Cleaned & Epoched |
| `02010022` | MDD | 1 | 301.1s | 75 | 10, 6, 23 | Cleaned & Epoched |
| `02010023` | MDD | 1 | 300.4s | 75 | 16, 31, 2 | Cleaned & Epoched |
| `02010024` | MDD | 1 | 301.6s | 75 | 28, 17, 8 | Cleaned & Epoched |
| `02010025` | MDD | 1 | 300.7s | 75 | 29, 1, 27 | Cleaned & Epoched |
| `02010026` | MDD | 1 | 300.5s | 75 | 13, 6, 7 | Cleaned & Epoched |
| `02010028` | MDD | 1 | 300.5s | 75 | 14, 24, 6 | Cleaned & Epoched |
| `02010030` | MDD | 1 | 301.6s | 75 | 29, 13, 7 | Cleaned & Epoched |
| `02010033` | MDD | 1 | 301.2s | 75 | 6, 29, 1 | Cleaned & Epoched |
| `02010034` | MDD | 1 | 300.6s | 75 | 31, 10, 29 | Cleaned & Epoched |
| `02010036` | MDD | 1 | 300.5s | 75 | 18, 15, 23 | Cleaned & Epoched |
| `02020008` | HC | 0 | 300.4s | 75 | 8, 13, 20 | Cleaned & Epoched |
| `02020010` | HC | 0 | 304.2s | 76 | 0, 1, 31 | Cleaned & Epoched |
| `02020013` | HC | 0 | 301.0s | 75 | 25, 30, 20 | Cleaned & Epoched |
| `02020014` | HC | 0 | 300.6s | 75 | 2, 21, 26 | Cleaned & Epoched |
| `02020015` | HC | 0 | 300.6s | 75 | 13, 22, 7 | Cleaned & Epoched |
| `02020016` | HC | 0 | 300.4s | 75 | 28, 15, 1 | Cleaned & Epoched |
| `02020018` | HC | 0 | 300.8s | 75 | 5, 21, 28 | Cleaned & Epoched |
| `02020019` | HC | 0 | 300.7s | 75 | 11, 6, 24 | Cleaned & Epoched |
| `02020020` | HC | 0 | 301.5s | 75 | 9, 28, 29 | Cleaned & Epoched |
| `02020021` | HC | 0 | 300.5s | 75 | 1, 3, 2 | Cleaned & Epoched |
| `02020022` | HC | 0 | 300.6s | 75 | 21, 1, 15 | Cleaned & Epoched |
| `02020023` | HC | 0 | 300.8s | 75 | 25, 28, 8 | Cleaned & Epoched |
| `02020025` | HC | 0 | 335.5s | 83 | 26, 12, 3 | Cleaned & Epoched |
| `02020026` | HC | 0 | 300.6s | 75 | 20, 24, 31 | Cleaned & Epoched |
| `02020027` | HC | 0 | 332.7s | 83 | 1, 19, 27 | Cleaned & Epoched |
| `02020029` | HC | 0 | 300.5s | 75 | 31, 21, 29 | Cleaned & Epoched |
| `02030002` | HC | 0 | 302.1s | 75 | 21, 16, 1 | Cleaned & Epoched |
| `02030003` | HC | 0 | 302.1s | 75 | 6, 1, 8 | Cleaned & Epoched |
| `02030004` | HC | 0 | 300.6s | 75 | 24, 22, 1 | Cleaned & Epoched |
| `02030005` | HC | 0 | 300.5s | 75 | 12, 5, 4 | Cleaned & Epoched |
| `02030006` | HC | 0 | 302.8s | 75 | 12, 17, 31 | Cleaned & Epoched |
| `02030007` | HC | 0 | 303.0s | 75 | 20, 17, 2 | Cleaned & Epoched |
| `02030009` | HC | 0 | 300.6s | 75 | 30, 21, 5 | Cleaned & Epoched |
| `02030014` | HC | 0 | 300.8s | 75 | 22, 16, 27 | Cleaned & Epoched |
| `02030017` | HC | 0 | 300.6s | 75 | 3, 12, 22 | Cleaned & Epoched |
| `02030018` | HC | 0 | 300.7s | 75 | 19, 1, 13 | Cleaned & Epoched |
| `02030019` | HC | 0 | 300.5s | 75 | 12, 24, 30 | Cleaned & Epoched |
| `02030020` | HC | 0 | 300.5s | 75 | 17, 12, 2 | Cleaned & Epoched |
| `02030021` | HC | 0 | 301.1s | 75 | 24, 10, 9 | Cleaned & Epoched |

## 3. Data Leakage Prevention & Normalization Contract

- **Subject Independence**: Normalization statistics (mean and standard deviation) are strictly NOT computed globally or saved into the dataset NPZ.
- **Training-Fold-Only Fit**: `fit_normalizer(train_x)` must be invoked independently inside each training fold by Member 3 (`src/training/train.py`).
- **Test Set Transform**: `apply_normalizer(test_x, stats)` applies the training statistics to held-out test subjects, guaranteeing zero subject leakage.

Report automatically generated by Member 2 Data Pipeline (`src/data/preprocess.py`).