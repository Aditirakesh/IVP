# CDL-MDDNet: Preprocessing Report (Member 2)

**Author:** Member 2 (EEG Data Pipeline Engineer)  
**Project:** CDL-MDDNet (Lightweight & Interpretable EEG MDD Detection)  
**Dataset:** MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) — Lanzhou University  
**Target Modality:** 128-channel Eyes-Closed Resting-State EEG  

---

## 1. Executive Summary & Pipeline Overview

Member 2 is responsible for ingesting, validating, filtering, cleaning, epoching, and standardizing the 128-channel resting-state EEG recordings from the MODMA dataset. The processed output satisfies all downstream contracts for CDL-MDDNet model training (Member 1, Member 3), benchmarking and leakage testing (Member 4), channel and band explainability (Member 5), baseline channel-reduction (Member 6), and evaluation (Member 7).

```
   Raw MODMA .mat
(src/data/raw/0201* & 0203*)
           │
           ▼
┌──────────────────────────────────────────────┐
│  load_modma.py                               │
│  - Adaptive dimension & orientation check    │
│  - Extract 128 scalp channels (omit Cz #129) │
│  - Map 8-digit Subject ID to diagnosis (0/1) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  preprocess.py: Filtering                    │
│  - 50 Hz zero-phase IIR notch filter (Q=30)  │
│  - 0.5 - 45 Hz 4th-order zero-phase bandpass │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  preprocess.py: ICA Artifact Removal         │
│  - FastICA decomposition (seed=42)           │
│  - Kurtosis thresholding (> 5.0) for blinks  │
│  - Frontal scalp projection correlation      │
│  - Capped at max 3 components per subject    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  preprocess.py: Epoching                     │
│  - 4.0 s window @ 250 Hz = 1000 samples      │
│  - Default overlap: 0.0 (non-overlapping)    │
│  - Save: src/data/processed/modma_epochs.npz │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  normalize.py (Invoked per training fold)    │
│  - fit_normalizer(train_x)                   │
│  - apply_normalizer(x, stats)                │
│  - Strictly prevents subject data leakage    │
└──────────────────────────────────────────────┘
```

---

## 2. Dataset & Cohort Specifications

- **Total Participants:** 53
- **Major Depressive Disorder (MDD):** 24 outpatients (prefix `0201`), Label = **`1`**
- **Healthy Controls (HC):** 29 participants (prefixes `0202` [16 subjects] and `0203` [13 subjects]), Label = **`0`**
- **Electrode Net:** 128-channel HydroCel Geodesic Sensor Net (HCGSN), Electrical Geodesics Inc. (EGI)
- **Reference Electrode:** Cz (129th channel in EEGLAB export, omitted from final 128-channel tensor)
- **Sampling Frequency ($f_s$):** 250 Hz (Nyquist limit = 125 Hz)
- **Recording Duration:** ~5 minutes of eyes-closed resting state (~75,000 samples per participant)

---

## 3. Signal Filtering & Artifact Rejection

### A. Powerline Notch Filter
- **Frequency:** 50.0 Hz (mains frequency in Lanzhou, China)
- **Design:** Zero-phase Infinite Impulse Response (IIR) notch filter, Quality Factor $Q = 30.0$
- **Implementation:** `scipy.signal.iirnotch` followed by forward-backward `scipy.signal.filtfilt` along the time axis.
- **Validation:** 50 Hz electrical hum power attenuated by $> 26\text{ dB}$ while preserving adjacent physiological rhythms.

### B. Bandpass Filter
- **Passband:** 0.5 Hz to 45.0 Hz
  - Isolates Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), and Beta (13–30 Hz) bands.
  - Eliminates slow baseline DC drift (< 0.5 Hz) and high-frequency electromyographic (EMG) muscle noise (> 45 Hz).
- **Design:** 4th-order zero-phase Butterworth filter implemented using Second-Order Sections (`sos`) for numerical stability.

### C. ICA Artifact Cleaning
- **Algorithm:** FastICA (`sklearn.decomposition.FastICA`), random seed fixed to `42` for reproducibility.
- **Decomposition:** Estimates up to 32 independent source components.
- **Artifact Identification Criterion:**
  1. **Fisher Kurtosis:** Blink and ocular saccade artifacts exhibit high peakiness (kurtosis $> 5.0$).
  2. **Frontal Projection:** Correlation against frontal channels (`E1` to `E32`, located on forehead and anterior scalp).
  3. **Safety Cap:** Automatically limits component removal to a maximum of 3 components per subject to prevent over-cleaning of true neural oscillations.

---

## 4. Epoch Generation & Yield Analysis

- **Epoch Duration:** Exactly 4.0 seconds.
- **Samples per Epoch:** $4.0\text{ s} \times 250\text{ Hz} = 1000\text{ samples}$.
- **Output Tensor Shape:** $(N, 128, 1000)$, `float32`.
- **Incomplete Epochs:** Trailing segments $< 1000$ samples are discarded (never padded or distorted).

### Epoch Yield Projections:

| Overlap Configuration | Step Size (samples) | Epochs / 5-min Subject | Total Cohort Epochs (53 Subjects) | MDD Epochs (24 Subj) | HC Epochs (29 Subj) |
|---|---|---|---|---|---|
| **0.0 (Non-overlapping, Default)** | **1000** | **~75** | **~3,975** | **~1,800** | **~2,175** |
| 0.50 (50% Overlap) | 500 | ~149 | ~7,897 | ~3,576 | ~4,321 |
| 0.75 (75% Overlap) | 250 | ~297 | ~15,741 | ~7,128 | ~8,613 |

---

## 5. Participant Metadata & Diagnostic Label Mapping

| Index | Subject ID | Diagnostic Group | Class Label | Age | Gender | Education (yrs) | PHQ-9 Score |
|---|---|---|---|---|---|---|---|
| 0 | `02010002` | MDD | 1 | 18 | F | 12 | 23 |
| 1 | `02010004` | MDD | 1 | 25 | F | 19 | 12 |
| 2 | `02010005` | MDD | 1 | 20 | M | 16 | 19 |
| 3 | `02010006` | MDD | 1 | 42 | M | 16 | 16 |
| 4 | `02010008` | MDD | 1 | 42 | M | 12 | 17 |
| 5 | `02010010` | MDD | 1 | 34 | M | 6 | 19 |
| 6 | `02010011` | MDD | 1 | 45 | M | 9 | 24 |
| 7 | `02010012` | MDD | 1 | 21 | F | 16 | 22 |
| 8 | `02010013` | MDD | 1 | 21 | F | 16 | 11 |
| 9 | `02010015` | MDD | 1 | 26 | M | 16 | 14 |
| 10 | `02010016` | MDD | 1 | 28 | F | 19 | 18 |
| 11 | `02010018` | MDD | 1 | 30 | M | 12 | 19 |
| 12 | `02010019` | MDD | 1 | 24 | F | 16 | 21 |
| 13 | `02010021` | MDD | 1 | 45 | F | 16 | 20 |
| 14 | `02010022` | MDD | 1 | 52 | M | 6 | 18 |
| 15 | `02010023` | MDD | 1 | 42 | F | 9 | 16 |
| 16 | `02010024` | MDD | 1 | 20 | F | 16 | 23 |
| 17 | `02010025` | MDD | 1 | 28 | M | 12 | 20 |
| 18 | `02010026` | MDD | 1 | 16 | M | 12 | 15 |
| 19 | `02010028` | MDD | 1 | 29 | M | 16 | 17 |
| 20 | `02010030` | MDD | 1 | 35 | F | 16 | 24 |
| 21 | `02010033` | MDD | 1 | 24 | F | 9 | 19 |
| 22 | `02010034` | MDD | 1 | 28 | M | 12 | 16 |
| 23 | `02010036` | MDD | 1 | 46 | M | 9 | 17 |
| 24 | `02020008` | Healthy Control | 0 | 20 | M | 16 | 1 |
| 25 | `02020010` | Healthy Control | 0 | 51 | M | 16 | 4 |
| 26 | `02020013` | Healthy Control | 0 | 27 | M | 16 | 5 |
| 27 | `02020014` | Healthy Control | 0 | 29 | F | 16 | 3 |
| 28 | `02020015` | Healthy Control | 0 | 41 | F | 16 | 0 |
| 29 | `02020016` | Healthy Control | 0 | 40 | M | 16 | 3 |
| 30 | `02020018` | Healthy Control | 0 | 34 | M | 16 | 3 |
| 31 | `02020019` | Healthy Control | 0 | 31 | M | 16 | 3 |
| 32 | `02020020` | Healthy Control | 0 | 29 | M | 16 | 5 |
| 33 | `02020021` | Healthy Control | 0 | 43 | M | 16 | 4 |
| 34 | `02020022` | Healthy Control | 0 | 41 | M | 16 | 3 |
| 35 | `02020023` | Healthy Control | 0 | 21 | M | 16 | 0 |
| 36 | `02020025` | Healthy Control | 0 | 33 | M | 16 | 2 |
| 37 | `02020026` | Healthy Control | 0 | 19 | M | 16 | 2 |
| 38 | `02020027` | Healthy Control | 0 | 34 | M | 16 | 4 |
| 39 | `02020029` | Healthy Control | 0 | 34 | M | 16 | 5 |
| 40 | `02030002` | Healthy Control | 0 | 23 | M | 19 | 0 |
| 41 | `02030003` | Healthy Control | 0 | 22 | M | 19 | 5 |
| 42 | `02030004` | Healthy Control | 0 | 22 | F | 19 | 3 |
| 43 | `02030005` | Healthy Control | 0 | 23 | F | 19 | 0 |
| 44 | `02030006` | Healthy Control | 0 | 23 | F | 19 | 0 |
| 45 | `02030007` | Healthy Control | 0 | 22 | F | 19 | 0 |
| 46 | `02030009` | Healthy Control | 0 | 40 | M | 16 | 4 |
| 47 | `02030014` | Healthy Control | 0 | 34 | F | 12 | 3 |
| 48 | `02030017` | Healthy Control | 0 | 52 | M | 12 | 3 |
| 49 | `02030018` | Healthy Control | 0 | 29 | M | 16 | 3 |
| 50 | `02030019` | Healthy Control | 0 | 28 | M | 9 | 4 |
| 51 | `02030020` | Healthy Control | 0 | 41 | F | 12 | 5 |
| 52 | `02030021` | Healthy Control | 0 | 26 | F | 16 | 0 |

---

## 6. Data Leakage Prevention Protocol

The normalization contract strictly guarantees subject independence:
- **No Global Normalization:** The saved `modma_epochs.npz` stores raw, cleaned, unnormalized physical potentials ($\mu\text{V}$).
- **Per-Fold Fitting:** Member 3 fits normalization statistics exclusively on the training fold:
  ```python
  from src.data.normalize import fit_normalizer, apply_normalizer

  # Fit statistics ONLY on training fold epochs
  stats = fit_normalizer(train_x)

  # Transform training, validation, and test sets using training stats
  norm_train_x = apply_normalizer(train_x, stats)
  norm_val_x   = apply_normalizer(val_x, stats)
  norm_test_x  = apply_normalizer(test_x, stats)
  ```
- **Zero-Variance Handling:** If any channel has zero variance (e.g. flat or disconnected channel), denominator standard deviation is safely replaced with $1.0$, preventing division-by-zero or NaN propagation.
- **Integrity Test:** Verified by `tests/test_normalizer.py` and `tests/test_data_pipeline.py`.

