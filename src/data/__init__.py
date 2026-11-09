"""EEG Data Pipeline package for CDL-MDDNet (Member 2 Contribution).

Exports key loading, preprocessing, and normalization interfaces:
- fit_normalizer, apply_normalizer
- load_modma_raw, inspect_mat_file, diagnose_raw_directory, scan_raw_dataset
- extract_subject_id_from_filename, get_subject_label, get_subject_mapping
- apply_notch_filter, apply_bandpass_filter, apply_ica_cleaning, create_epochs
- preprocess_single_subject, preprocess_cohort, run_synthetic_pipeline
- MODMA_SUBJECTS_METADATA
"""

__all__ = [
    # Normalization contract
    "fit_normalizer",
    "apply_normalizer",
    # Data loading and inspection
    "load_modma_raw",
    "inspect_mat_file",
    "diagnose_raw_directory",
    "scan_raw_dataset",
    "extract_subject_id_from_filename",
    "get_subject_label",
    "get_subject_mapping",
    "MODMA_SUBJECTS_METADATA",
    # Preprocessing
    "apply_notch_filter",
    "apply_bandpass_filter",
    "apply_ica_cleaning",
    "resample_eeg",
    "create_epochs",
    "preprocess_single_subject",
    "preprocess_cohort",
    "generate_synthetic_subject",
    "run_synthetic_pipeline",
]


def __getattr__(name: str):
    """Lazy module attribute loader to avoid eager submodule circularity during python -m execution."""
    if name in ("fit_normalizer", "apply_normalizer"):
        from src.data import normalize
        return getattr(normalize, name)

    if name in (
        "load_modma_raw",
        "inspect_mat_file",
        "diagnose_raw_directory",
        "scan_raw_dataset",
        "extract_subject_id_from_filename",
        "get_subject_label",
        "get_subject_mapping",
        "MODMA_SUBJECTS_METADATA",
    ):
        from src.data import load_modma
        return getattr(load_modma, name)

    if name in (
        "apply_notch_filter",
        "apply_bandpass_filter",
        "apply_ica_cleaning",
        "resample_eeg",
        "create_epochs",
        "preprocess_single_subject",
        "preprocess_cohort",
        "generate_synthetic_subject",
        "run_synthetic_pipeline",
    ):
        from src.data import preprocess
        return getattr(preprocess, name)

    raise AttributeError(f"module 'src.data' has no attribute '{name}'")
