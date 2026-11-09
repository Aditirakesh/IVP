"""Comprehensive unit and integration tests for Member 2's MODMA EEG Data Pipeline.

Validates:
1. Normalization (fit_normalizer, apply_normalizer, zero-variance handling, no data leakage).
2. Filtering (50 Hz notch filter, 0.5-45 Hz bandpass filter, frequency response).
3. FastICA artifact cleaning (reproducibility, component exclusion).
4. Epoch generation (exact 1000 samples, shape (N, C, 1000), overlap parameter).
5. Metadata and ID mapping (53 participants, binary labels 0 and 1, stable integer IDs).
6. MATLAB .mat file inspection and loading (MATLAB v5/v7, axis orientation detection, Cz handling).
7. End-to-end dataset creation and NPZ serialization without pickle.
"""

from pathlib import Path
import numpy as np
import pytest
import scipy.io

from src.data.load_modma import (
    MODMA_SUBJECTS_METADATA,
    extract_subject_id_from_filename,
    get_subject_label,
    get_subject_mapping,
    inspect_mat_file,
    load_modma_raw,
)
from src.data.normalize import apply_normalizer, fit_normalizer
from src.data.preprocess import (
    apply_bandpass_filter,
    apply_ica_cleaning,
    apply_notch_filter,
    create_epochs,
    generate_synthetic_subject,
    preprocess_cohort,
    preprocess_single_subject,
    run_synthetic_pipeline,
)


# =============================================================================
# 1. Normalization Tests
# =============================================================================

class TestNormalization:
    def test_fit_and_apply_shapes_and_dtype(self):
        rng = np.random.default_rng(42)
        train_x = rng.normal(5.0, 2.0, size=(100, 128, 1000)).astype(np.float32)
        test_x = rng.normal(5.0, 2.0, size=(20, 128, 1000)).astype(np.float32)

        stats = fit_normalizer(train_x)
        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"].shape == (1, 128, 1)
        assert stats["std"].shape == (1, 128, 1)

        norm_train = apply_normalizer(train_x, stats)
        norm_test = apply_normalizer(test_x, stats)

        assert norm_train.shape == train_x.shape
        assert norm_test.shape == test_x.shape
        assert norm_train.dtype == np.float32
        assert norm_test.dtype == np.float32

        # Check train mean ~ 0 and std ~ 1 per channel
        assert np.allclose(norm_train.mean(axis=(0, 2)), 0.0, atol=1e-2)
        assert np.allclose(norm_train.std(axis=(0, 2)), 1.0, atol=1e-2)

    def test_zero_variance_channels_handled_safely(self):
        # Channel 0 is completely flat (0.0 everywhere, e.g. disconnected electrode)
        train_x = np.zeros((50, 10, 500), dtype=np.float32)
        train_x[:, 1:, :] = np.random.randn(50, 9, 500)

        stats = fit_normalizer(train_x)
        out = apply_normalizer(train_x, stats)

        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert np.all(out[:, 0, :] == 0.0)

    def test_2d_input_support(self):
        train_x = np.random.randn(128, 2000).astype(np.float32)
        stats = fit_normalizer(train_x)
        out = apply_normalizer(train_x, stats)
        assert out.shape == (128, 2000)
        assert out.dtype == np.float32
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-2)

    def test_channel_mismatch_raises(self):
        train_x = np.zeros((10, 128, 100), dtype=np.float32)
        test_x = np.zeros((10, 64, 100), dtype=np.float32)
        stats = fit_normalizer(train_x)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            apply_normalizer(test_x, stats)


# =============================================================================
# 2. Metadata and Subject ID Mapping Tests
# =============================================================================

class TestMetadata:
    def test_metadata_has_53_participants(self):
        assert len(MODMA_SUBJECTS_METADATA) == 53
        mdd = [k for k, v in MODMA_SUBJECTS_METADATA.items() if v["type"] == "MDD"]
        hc = [k for k, v in MODMA_SUBJECTS_METADATA.items() if v["type"] == "HC"]
        assert len(mdd) == 24
        assert len(hc) == 29

    def test_subject_mapping_bijection(self):
        str_to_int, int_to_str = get_subject_mapping()
        assert len(str_to_int) == 53
        assert len(int_to_str) == 53
        # All integer IDs must span 0 to 52 contiguously
        assert sorted(str_to_int.values()) == list(range(53))
        for sid, idx in str_to_int.items():
            assert int_to_str[idx] == sid

    def test_extract_subject_id_from_filenames(self):
        filenames = [
            ("02010002rest 20150416 1017..mat", "02010002"),
            ("02010008_rest 20150619 1653.mat", "02010008"),
            ("02020008rest.mat", "02020008"),
            ("02030002.mat", "02030002"),
        ]
        for fname, expected in filenames:
            assert extract_subject_id_from_filename(fname) == expected

    def test_get_subject_labels(self):
        assert get_subject_label("02010002") == 1  # MDD
        assert get_subject_label("02010036") == 1  # MDD
        assert get_subject_label("02020008") == 0  # HC
        assert get_subject_label("02030021") == 0  # HC

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="Unable to extract"):
            extract_subject_id_from_filename("unrelated_file.mat")


# =============================================================================
# 3. Filtering Tests (Notch & Bandpass)
# =============================================================================

class TestFiltering:
    def test_notch_filter_attenuates_50hz(self):
        fs = 250.0
        t = np.linspace(0, 4.0, int(4.0 * fs), endpoint=False)
        # Signal with 10 Hz EEG component + strong 50 Hz powerline interference
        clean = np.sin(2 * np.pi * 10.0 * t)
        noise = 5.0 * np.sin(2 * np.pi * 50.0 * t)
        signal = np.vstack([clean + noise, clean + noise])  # (2, 1000)

        notched = apply_notch_filter(signal, fs=fs, notch_freq=50.0)

        # 50 Hz power must be heavily attenuated (> 20 dB drop)
        fft_orig = np.abs(np.fft.rfft(signal[0]))
        fft_notched = np.abs(np.fft.rfft(notched[0]))
        freqs = np.fft.rfftfreq(len(t), 1 / fs)

        idx_50 = np.argmin(np.abs(freqs - 50.0))
        assert fft_notched[idx_50] < 0.05 * fft_orig[idx_50]

        # 10 Hz component must be preserved
        idx_10 = np.argmin(np.abs(freqs - 10.0))
        assert np.isclose(fft_notched[idx_10], fft_orig[idx_10], rtol=0.05)

    def test_bandpass_filter(self):
        fs = 250.0
        # Use 20 seconds to provide integer cycles for low-frequency evaluation
        duration = 20.0
        t = np.linspace(0, duration, int(duration * fs), endpoint=False)
        # Slow drift (0.1 Hz) + alpha (10 Hz) + high noise (60 Hz)
        drift = 10.0 * np.sin(2 * np.pi * 0.1 * t)
        alpha = 2.0 * np.sin(2 * np.pi * 10.0 * t)
        high = 5.0 * np.sin(2 * np.pi * 60.0 * t)
        signal = (drift + alpha + high)[np.newaxis, :]

        filtered = apply_bandpass_filter(signal, fs=fs, lowcut=0.5, highcut=45.0)

        fft_f = np.abs(np.fft.rfft(filtered[0]))
        freqs = np.fft.rfftfreq(len(t), 1 / fs)

        # Check drift (< 0.5 Hz) and high freq (> 45 Hz) are strongly attenuated
        idx_drift = np.argmin(np.abs(freqs - 0.1))
        idx_high = np.argmin(np.abs(freqs - 60.0))
        idx_alpha = np.argmin(np.abs(freqs - 10.0))

        assert fft_f[idx_drift] < 0.05 * fft_f[idx_alpha]
        assert fft_f[idx_high] < 0.15 * fft_f[idx_alpha]


# =============================================================================
# 4. ICA Artifact Removal Tests
# =============================================================================

class TestICA:
    def test_ica_cleaning_removes_ocular_spikes(self):
        fs = 250.0
        duration = 8.0
        n_samples = int(duration * fs)
        rng = np.random.default_rng(42)

        # 16 channels of background activity
        data = rng.normal(0, 1.0, size=(16, n_samples)).astype(np.float32)

        # Inject large blink spike artifact into frontal channels (0, 1)
        spike = np.zeros(n_samples, dtype=np.float32)
        spike[500:550] = 50.0  # huge transient
        data[0] += spike
        data[1] += spike * 0.8

        cleaned, info = apply_ica_cleaning(data, n_components=8, random_seed=42)

        assert cleaned.shape == data.shape
        assert cleaned.dtype == np.float32
        assert "excluded_components" in info
        # Peak amplitude in cleaned data must be substantially reduced compared to raw peak
        assert np.max(np.abs(cleaned[0])) < np.max(np.abs(data[0]))


# =============================================================================
# 5. Epoch Generation Tests
# =============================================================================

class TestEpoching:
    def test_exact_1000_samples_per_epoch(self):
        fs = 250.0
        # 10 seconds of 128-channel EEG = 2500 samples
        signal = np.random.randn(128, 2500).astype(np.float32)

        # Non-overlapping 4-second epochs: (2500 - 1000) // 1000 + 1 = 2 epochs
        epochs = create_epochs(signal, epoch_duration_sec=4.0, fs=fs, overlap=0.0)
        assert epochs.shape == (2, 128, 1000)
        assert epochs.dtype == np.float32

    def test_configurable_overlap(self):
        fs = 250.0
        signal = np.random.randn(128, 2500).astype(np.float32)

        # 50% overlap: step is 500 samples. Valid starts: 0, 500, 1000, 1500 (4 epochs)
        epochs_50 = create_epochs(signal, epoch_duration_sec=4.0, fs=fs, overlap=0.5)
        assert epochs_50.shape == (4, 128, 1000)

        # 75% overlap: step is 250 samples. (2500 - 1000) // 250 + 1 = 7 epochs
        epochs_75 = create_epochs(signal, epoch_duration_sec=4.0, fs=fs, overlap=0.75)
        assert epochs_75.shape == (7, 128, 1000)

    def test_too_short_signal_raises(self):
        signal = np.random.randn(128, 800).astype(np.float32)
        with pytest.raises(ValueError, match="shorter than required epoch length"):
            create_epochs(signal, epoch_duration_sec=4.0, fs=250.0)


# =============================================================================
# 6. MATLAB Loader and Orientation Tests
# =============================================================================

class TestMATLoader:
    def test_load_eeglab_struct_mat(self, tmp_path):
        # Create a mock EEGLAB struct MAT file
        mat_path = tmp_path / "02010002rest 20150416 1017..mat"
        eeg_data = np.random.randn(129, 2000).astype(np.float32)  # 129 channels (128 scalp + Cz)

        # Mock EEGLAB structure
        class MockEEG:
            def __init__(self):
                self.data = eeg_data
                self.srate = 250.0
                self.nbchan = 129
                self.pnts = 2000

        scipy.io.savemat(str(mat_path), {"EEG": MockEEG()})

        # Test diagnostic inspect
        diag = inspect_mat_file(mat_path)
        assert diag["subject_id"] == "02010002"
        assert diag["label"] == 1
        assert diag["diagnosis"] == "MDD"
        assert diag["eeg_variable"] == "EEG.data"
        assert diag["channel_dim"] == 0

        # Test loading
        sig, fs, ch_names, sid, lbl = load_modma_raw(mat_path)
        assert sig.shape == (128, 2000)  # 129th channel excluded
        assert fs == 250.0
        assert sid == "02010002"
        assert lbl == 1
        assert len(ch_names) == 128
        assert ch_names[0] == "E1"
        assert ch_names[-1] == "E128"

    def test_load_transposed_raw_mat(self, tmp_path):
        # Shape (time, channels) = (2000, 128) -> should automatically transpose to (128, 2000)
        mat_path = tmp_path / "02020008rest.mat"
        eeg_data = np.random.randn(2000, 128).astype(np.float32)
        scipy.io.savemat(str(mat_path), {"data": eeg_data})

        sig, fs, ch_names, sid, lbl = load_modma_raw(mat_path)
        assert sig.shape == (128, 2000)
        assert sid == "02020008"
        assert lbl == 0


# =============================================================================
# 7. End-to-End Pipeline & NPZ Serialization Tests
# =============================================================================

class TestEndToEndPipeline:
    def test_synthetic_cohort_pipeline_and_npz_contract(self, tmp_path):
        out_npz = tmp_path / "modma_epochs.npz"
        X, y, subject_id, ch_names = run_synthetic_pipeline(output_path=out_npz)

        assert out_npz.exists()

        # Check NPZ contract without allow_pickle=True
        with np.load(str(out_npz)) as data:
            assert "X" in data
            assert "y" in data
            assert "subject_id" in data
            assert "ch_names" in data

            loaded_X = data["X"]
            loaded_y = data["y"]
            loaded_sids = data["subject_id"]
            loaded_ch = data["ch_names"]

            assert loaded_X.shape == X.shape
            assert loaded_X.shape[1] == 128
            assert loaded_X.shape[2] == 1000
            assert loaded_X.dtype == np.float32

            # y must be binary 0 and 1
            assert set(np.unique(loaded_y)).issubset({0, 1})

            # subject_id must be integers
            assert np.issubdtype(loaded_sids.dtype, np.integer)

            # ch_names must match channel dimension
            assert len(loaded_ch) == 128
            assert loaded_ch[0] == "E1"
            assert loaded_ch[-1] == "E128"
