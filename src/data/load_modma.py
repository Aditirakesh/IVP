"""MODMA 128-channel Resting-State EEG Data Loader and Diagnostic Tool.

This module is responsible for:
1. Scanning and identifying MODMA EEG recordings from `src/data/raw/`.
2. Extracting participant IDs from recording filenames (e.g. '02010002rest...mat').
3. Associating recordings with diagnostic labels (0 = Healthy Control, 1 = MDD)
   using the official MODMA metadata.
4. Loading MATLAB .mat files (both standard MAT v5/v7 and MATLAB v7.3 / HDF5).
5. Determining channel vs. time dimensions adaptively without hardcoded assumptions.
6. Extracting 128 scalp channels (handling the 129th Cz reference channel per MODMA paper).
7. Extracting or generating consistent channel names (E1 to E128).
8. Identifying or verifying the 250 Hz sampling frequency.
9. Providing a command-line and programmatic diagnostic tool to inspect local .mat files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io


# -----------------------------------------------------------------------------
# Official MODMA 128-channel Resting-State Participant Metadata
# Source: subjects_information_EEG_128channels_resting_lanzhou_2015
# Total: 53 subjects (24 MDD, 29 Healthy Controls)
# -----------------------------------------------------------------------------

MODMA_SUBJECTS_METADATA: Dict[str, Dict[str, Any]] = {
    # 24 Major Depressive Disorder (MDD) participants: prefix 0201
    "02010002": {"type": "MDD", "label": 1, "age": 18, "gender": "F", "education": 12, "PHQ-9": 23},
    "02010004": {"type": "MDD", "label": 1, "age": 25, "gender": "F", "education": 19, "PHQ-9": 12},
    "02010005": {"type": "MDD", "label": 1, "age": 20, "gender": "M", "education": 16, "PHQ-9": 19},
    "02010006": {"type": "MDD", "label": 1, "age": 42, "gender": "M", "education": 16, "PHQ-9": 16},
    "02010008": {"type": "MDD", "label": 1, "age": 42, "gender": "M", "education": 12, "PHQ-9": 17},
    "02010010": {"type": "MDD", "label": 1, "age": 34, "gender": "M", "education": 6,  "PHQ-9": 19},
    "02010011": {"type": "MDD", "label": 1, "age": 45, "gender": "M", "education": 9,  "PHQ-9": 24},
    "02010012": {"type": "MDD", "label": 1, "age": 21, "gender": "F", "education": 16, "PHQ-9": 22},
    "02010013": {"type": "MDD", "label": 1, "age": 21, "gender": "F", "education": 16, "PHQ-9": 11},
    "02010015": {"type": "MDD", "label": 1, "age": 26, "gender": "M", "education": 16, "PHQ-9": 14},
    "02010016": {"type": "MDD", "label": 1, "age": 28, "gender": "F", "education": 19, "PHQ-9": 18},
    "02010018": {"type": "MDD", "label": 1, "age": 30, "gender": "M", "education": 12, "PHQ-9": 19},
    "02010019": {"type": "MDD", "label": 1, "age": 24, "gender": "F", "education": 16, "PHQ-9": 21},
    "02010021": {"type": "MDD", "label": 1, "age": 45, "gender": "F", "education": 16, "PHQ-9": 20},
    "02010022": {"type": "MDD", "label": 1, "age": 52, "gender": "M", "education": 6,  "PHQ-9": 18},
    "02010023": {"type": "MDD", "label": 1, "age": 42, "gender": "F", "education": 9,  "PHQ-9": 16},
    "02010024": {"type": "MDD", "label": 1, "age": 20, "gender": "F", "education": 16, "PHQ-9": 23},
    "02010025": {"type": "MDD", "label": 1, "age": 28, "gender": "M", "education": 12, "PHQ-9": 20},
    "02010026": {"type": "MDD", "label": 1, "age": 16, "gender": "M", "education": 12, "PHQ-9": 15},
    "02010028": {"type": "MDD", "label": 1, "age": 29, "gender": "M", "education": 16, "PHQ-9": 17},
    "02010030": {"type": "MDD", "label": 1, "age": 35, "gender": "F", "education": 16, "PHQ-9": 24},
    "02010033": {"type": "MDD", "label": 1, "age": 24, "gender": "F", "education": 9,  "PHQ-9": 19},
    "02010034": {"type": "MDD", "label": 1, "age": 28, "gender": "M", "education": 12, "PHQ-9": 16},
    "02010036": {"type": "MDD", "label": 1, "age": 46, "gender": "M", "education": 9,  "PHQ-9": 17},

    # 29 Healthy Control (HC) participants: prefix 0202 (16 subjects) and 0203 (13 subjects)
    "02020008": {"type": "HC", "label": 0, "age": 20, "gender": "M", "education": 16, "PHQ-9": 1},
    "02020010": {"type": "HC", "label": 0, "age": 51, "gender": "M", "education": 16, "PHQ-9": 4},
    "02020013": {"type": "HC", "label": 0, "age": 27, "gender": "M", "education": 16, "PHQ-9": 5},
    "02020014": {"type": "HC", "label": 0, "age": 29, "gender": "F", "education": 16, "PHQ-9": 3},
    "02020015": {"type": "HC", "label": 0, "age": 41, "gender": "F", "education": 16, "PHQ-9": 0},
    "02020016": {"type": "HC", "label": 0, "age": 40, "gender": "M", "education": 16, "PHQ-9": 3},
    "02020018": {"type": "HC", "label": 0, "age": 34, "gender": "M", "education": 16, "PHQ-9": 3},
    "02020019": {"type": "HC", "label": 0, "age": 31, "gender": "M", "education": 16, "PHQ-9": 3},
    "02020020": {"type": "HC", "label": 0, "age": 29, "gender": "M", "education": 16, "PHQ-9": 5},
    "02020021": {"type": "HC", "label": 0, "age": 43, "gender": "M", "education": 16, "PHQ-9": 4},
    "02020022": {"type": "HC", "label": 0, "age": 41, "gender": "M", "education": 16, "PHQ-9": 3},
    "02020023": {"type": "HC", "label": 0, "age": 21, "gender": "M", "education": 16, "PHQ-9": 0},
    "02020025": {"type": "HC", "label": 0, "age": 33, "gender": "M", "education": 16, "PHQ-9": 2},
    "02020026": {"type": "HC", "label": 0, "age": 19, "gender": "M", "education": 16, "PHQ-9": 2},
    "02020027": {"type": "HC", "label": 0, "age": 34, "gender": "M", "education": 16, "PHQ-9": 4},
    "02020029": {"type": "HC", "label": 0, "age": 34, "gender": "M", "education": 16, "PHQ-9": 5},
    "02030002": {"type": "HC", "label": 0, "age": 23, "gender": "M", "education": 19, "PHQ-9": 0},
    "02030003": {"type": "HC", "label": 0, "age": 22, "gender": "M", "education": 19, "PHQ-9": 5},
    "02030004": {"type": "HC", "label": 0, "age": 22, "gender": "F", "education": 19, "PHQ-9": 3},
    "02030005": {"type": "HC", "label": 0, "age": 23, "gender": "F", "education": 19, "PHQ-9": 0},
    "02030006": {"type": "HC", "label": 0, "age": 23, "gender": "F", "education": 19, "PHQ-9": 0},
    "02030007": {"type": "HC", "label": 0, "age": 22, "gender": "F", "education": 19, "PHQ-9": 0},
    "02030009": {"type": "HC", "label": 0, "age": 40, "gender": "M", "education": 16, "PHQ-9": 4},
    "02030014": {"type": "HC", "label": 0, "age": 34, "gender": "F", "education": 12, "PHQ-9": 3},
    "02030017": {"type": "HC", "label": 0, "age": 52, "gender": "M", "education": 12, "PHQ-9": 3},
    "02030018": {"type": "HC", "label": 0, "age": 29, "gender": "M", "education": 16, "PHQ-9": 3},
    "02030019": {"type": "HC", "label": 0, "age": 28, "gender": "M", "education": 9,  "PHQ-9": 4},
    "02030020": {"type": "HC", "label": 0, "age": 41, "gender": "F", "education": 12, "PHQ-9": 5},
    "02030021": {"type": "HC", "label": 0, "age": 26, "gender": "F", "education": 16, "PHQ-9": 0},
}


def get_subject_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return stable mappings between 8-digit subject ID strings and 0-indexed integers.

    Sorting all 53 participant IDs alphabetically guarantees a deterministic,
    reproducible bijection between string IDs and integer subject IDs (0..52).
    """
    sorted_ids = sorted(MODMA_SUBJECTS_METADATA.keys())
    str_to_int = {sid: idx for idx, sid in enumerate(sorted_ids)}
    int_to_str = {idx: sid for idx, sid in enumerate(sorted_ids)}
    return str_to_int, int_to_str


def extract_subject_id_from_filename(filename_or_path: Union[str, Path]) -> str:
    """Extract the 8-character MODMA subject ID from a filename or path.

    Handles naming variations observed in MODMA:
    - '02010002rest 20150416 1017..mat'
    - '02010008_rest 20150619 1653.mat'
    - '02020008rest.mat'
    - '02030002.mat'

    Parameters
    ----------
    filename_or_path : str or Path
        Filename or path to inspect.

    Returns
    -------
    str
        The 8-digit subject ID string.

    Raises
    ------
    ValueError
        If no valid subject ID can be identified.
    """
    stem = Path(filename_or_path).name
    # MODMA participant ID pattern: 8 digits starting with 020
    match = re.search(r"\b(020[123]\d{4})\b", stem)
    if not match:
        match = re.search(r"(020\d{5})", stem)
    if not match:
        match = re.search(r"(\d{8})", stem)

    if match:
        return match.group(1)

    raise ValueError(
        f"Unable to extract an 8-digit MODMA subject ID from filename: '{stem}'. "
        f"Expected filenames starting with or containing IDs like '02010002', '02020008', '02030002'."
    )


def get_subject_label(
    subject_id: str,
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """Determine the binary diagnostic label for a participant.

    Labels:
        0: Healthy Control (HC / NC)
        1: Major Depressive Disorder (MDD)

    Resolution order:
    1. Lookup in supplied metadata or built-in MODMA_SUBJECTS_METADATA.
    2. Rule-based deduction from prefix per MODMA paper:
       - '0201' -> MDD (1)
       - '0202' or '0203' -> Healthy Control (0)
    """
    meta = metadata if metadata is not None else MODMA_SUBJECTS_METADATA

    if subject_id in meta:
        return int(meta[subject_id]["label"])

    # Fallback to MODMA prefix rules
    if subject_id.startswith("0201"):
        return 1
    elif subject_id.startswith("0202") or subject_id.startswith("0203"):
        return 0

    raise ValueError(
        f"Unknown subject ID '{subject_id}': cannot determine diagnostic label. "
        f"Subject not found in MODMA metadata and prefix does not match 0201 (MDD) or 0202/0203 (HC)."
    )


# -----------------------------------------------------------------------------
# MATLAB File Inspection and Loading
# -----------------------------------------------------------------------------

def inspect_mat_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect the internal structure of a MATLAB .mat recording without loading huge arrays.

    Provides complete diagnostic insight into:
    - MATLAB format version (v5/v7 vs v7.3 HDF5)
    - Available variables, types, and array dimensions
    - Presence of EEGLAB struct ('EEG') or raw arrays
    - Channel dimension candidates (128 or 129)
    - Sampling frequency metadata
    - Participant ID and label resolution

    Parameters
    ----------
    file_path : str or Path
        Path to the .mat file.

    Returns
    -------
    dict
        Detailed diagnostic dictionary.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    report: Dict[str, Any] = {
        "file_name": p.name,
        "file_path": str(p.resolve()),
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "mat_version": "unknown",
        "variables": {},
        "eeg_variable": None,
        "detected_shape": None,
        "channel_dim": None,
        "time_dim": None,
        "sampling_rate": None,
        "subject_id": None,
        "label": None,
        "diagnosis": None,
        "notes": [],
    }

    # Attempt subject ID extraction
    try:
        sid = extract_subject_id_from_filename(p.name)
        report["subject_id"] = sid
        lbl = get_subject_label(sid)
        report["label"] = lbl
        report["diagnosis"] = "MDD" if lbl == 1 else "Healthy Control"
    except Exception as e:
        report["notes"].append(f"Subject extraction note: {e}")

    # Try standard scipy.io.loadmat (MATLAB v5 / v7 <= 7.2)
    try:
        mat_dict = scipy.io.loadmat(
            str(p),
            struct_as_record=False,
            squeeze_me=True,
        )
        report["mat_version"] = "MATLAB v5/v7 (standard)"

        for k, v in mat_dict.items():
            if k.startswith("__"):
                continue

            var_info: Dict[str, Any] = {
                "type": type(v).__name__,
            }

            if isinstance(v, np.ndarray):
                var_info["shape"] = v.shape
                var_info["dtype"] = str(v.dtype)
            elif hasattr(v, "_fieldnames"):
                # MATLAB struct loaded by scipy with struct_as_record=False
                fields = list(v._fieldnames)
                var_info["is_struct"] = True
                var_info["fields"] = fields

                # Inspect struct fields like 'data', 'srate', 'chanlocs'
                struct_details = {}
                for field in fields:
                    try:
                        val = getattr(v, field)
                        if isinstance(val, np.ndarray):
                            struct_details[field] = {"shape": val.shape, "dtype": str(val.dtype)}
                        else:
                            struct_details[field] = {"type": type(val).__name__, "val": str(val)[:50]}
                    except Exception:
                        pass
                var_info["struct_details"] = struct_details

            report["variables"][k] = var_info

    except NotImplementedError:
        # File is MATLAB v7.3 (HDF5 format)
        report["mat_version"] = "MATLAB v7.3 (HDF5)"
        try:
            import h5py
            with h5py.File(str(p), "r") as h5f:
                for k in h5f.keys():
                    item = h5f[k]
                    if isinstance(item, h5py.Dataset):
                        report["variables"][k] = {
                            "type": "HDF5 Dataset",
                            "shape": item.shape,
                            "dtype": str(item.dtype),
                        }
                    elif isinstance(item, h5py.Group):
                        report["variables"][k] = {
                            "type": "HDF5 Group / Struct",
                            "keys": list(item.keys()),
                        }
        except ImportError:
            report["notes"].append(
                "MATLAB v7.3 file detected, but 'h5py' is not installed. Run 'pip install h5py' to inspect."
            )
        except Exception as e:
            report["notes"].append(f"HDF5 reading error: {e}")

    except Exception as e:
        report["notes"].append(f"Loadmat error: {e}")

    # Analyze candidate EEG variables and channel orientations
    _analyze_eeg_candidate(report)

    return report


def _analyze_eeg_candidate(report: Dict[str, Any]) -> None:
    """Analyze variable shapes to determine candidate EEG data array and channel orientation."""
    vars_dict = report.get("variables", {})

    # Check for EEGLAB struct 'EEG'
    if "EEG" in vars_dict and vars_dict["EEG"].get("is_struct"):
        s_details = vars_dict["EEG"].get("struct_details", {})
        if "data" in s_details:
            shape = s_details["data"].get("shape")
            report["eeg_variable"] = "EEG.data"
            report["detected_shape"] = shape
            if "srate" in s_details:
                report["sampling_rate"] = s_details["srate"].get("val")

    # If not identified in EEGLAB struct, check top-level arrays
    if report["eeg_variable"] is None:
        for name, info in vars_dict.items():
            shape = info.get("shape")
            if shape and len(shape) >= 2:
                # Look for 128 or 129 channels
                if 128 in shape or 129 in shape:
                    report["eeg_variable"] = name
                    report["detected_shape"] = shape
                    break

    # If still not found, check any 2D numeric array
    if report["eeg_variable"] is None:
        for name, info in vars_dict.items():
            shape = info.get("shape")
            if shape and len(shape) == 2:
                report["eeg_variable"] = name
                report["detected_shape"] = shape
                break

    # Determine channel orientation
    shape = report.get("detected_shape")
    if shape and len(shape) >= 2:
        d0, d1 = shape[0], shape[1]
        if d0 in (128, 129) and d1 > d0:
            report["channel_dim"] = 0
            report["time_dim"] = 1
            report["notes"].append(f"Channel dimension is Axis 0 ({d0} channels), time is Axis 1 ({d1} samples).")
        elif d1 in (128, 129) and d0 > d1:
            report["channel_dim"] = 1
            report["time_dim"] = 0
            report["notes"].append(f"Channel dimension is Axis 1 ({d1} channels), time is Axis 0 ({d0} samples). Will transpose.")
        elif d0 < d1 and d0 <= 256:
            report["channel_dim"] = 0
            report["time_dim"] = 1
            report["notes"].append(f"Inferred Axis 0 as channels ({d0} <= 256) and Axis 1 as time ({d1} samples).")
        elif d1 < d0 and d1 <= 256:
            report["channel_dim"] = 1
            report["time_dim"] = 0
            report["notes"].append(f"Inferred Axis 1 as channels ({d1} <= 256) and Axis 0 as time ({d0} samples). Will transpose.")


def load_modma_raw(
    file_path: Union[str, Path],
    expected_channels: int = 128,
    default_fs: float = 250.0,
) -> Tuple[np.ndarray, float, List[str], str, int]:
    """Load an original MODMA resting-state EEG recording from a .mat file.

    Handles:
    - MATLAB v5/v7 and MATLAB v7.3 (HDF5) files.
    - EEGLAB structures (`EEG.data`) and standalone 2D arrays.
    - Channel orientation detection (does NOT assume axis 0 or axis 1).
    - 129-channel recordings: extracts the 128 scalp channels (E1-E128) and
      omits the 129th reference channel (Cz) per MODMA documentation.
    - Extraction of channel names (or generation of E1-E128).
    - Sampling frequency identification (defaults to 250 Hz if unrecorded).
    - Participant ID and binary diagnostic label (0 = HC, 1 = MDD).

    Parameters
    ----------
    file_path : str or Path
        Path to the MATLAB recording (.mat).
    expected_channels : int, default=128
        Expected number of scalp EEG channels.
    default_fs : float, default=250.0
        Fallback sampling rate in Hz.

    Returns
    -------
    signals : np.ndarray
        Raw EEG signals of shape (128, n_samples), dtype float32.
    fs : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel names of length 128 (e.g. ['E1', 'E2', ..., 'E128']).
    subject_id : str
        8-digit participant ID string (e.g. '02010002').
    label : int
        Diagnostic label (0 = Healthy Control, 1 = MDD).
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Recording file not found: {p}")

    subject_id = extract_subject_id_from_filename(p.name)
    label = get_subject_label(subject_id)

    signals: Optional[np.ndarray] = None
    fs: float = default_fs
    ch_names: Optional[List[str]] = None

    # Strategy 1: scipy.io.loadmat for standard MATLAB files
    try:
        mat = scipy.io.loadmat(str(p), struct_as_record=False, squeeze_me=True)

        # Check for EEGLAB structure
        if "EEG" in mat:
            eeg_obj = mat["EEG"]
            if hasattr(eeg_obj, "data"):
                raw_data = getattr(eeg_obj, "data")
                if isinstance(raw_data, np.ndarray):
                    signals = raw_data
            if hasattr(eeg_obj, "srate"):
                try:
                    srate_val = float(getattr(eeg_obj, "srate"))
                    if srate_val > 0:
                        fs = srate_val
                except (ValueError, TypeError):
                    pass
            if hasattr(eeg_obj, "chanlocs"):
                chanlocs = getattr(eeg_obj, "chanlocs")
                if isinstance(chanlocs, np.ndarray):
                    extracted_names = []
                    for item in chanlocs:
                        if hasattr(item, "labels"):
                            extracted_names.append(str(getattr(item, "labels")))
                    if len(extracted_names) >= expected_channels:
                        ch_names = extracted_names[:expected_channels]

        # If not inside EEG struct, check direct variables
        if signals is None:
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    if 128 in v.shape or 129 in v.shape or v.shape[0] < v.shape[1]:
                        signals = v
                        break

    except NotImplementedError:
        # Strategy 2: MATLAB v7.3 HDF5 file
        try:
            import h5py
        except ImportError:
            raise ImportError(
                f"File '{p.name}' is in MATLAB v7.3 (HDF5) format, which requires the 'h5py' package. "
                f"Please install it with: pip install h5py"
            )

        with h5py.File(str(p), "r") as h5f:
            # Check for EEG dataset or group
            if "EEG" in h5f:
                eeg_grp = h5f["EEG"]
                if isinstance(eeg_grp, h5py.Group) and "data" in eeg_grp:
                    signals = np.array(eeg_grp["data"])
                elif isinstance(eeg_grp, h5py.Dataset):
                    signals = np.array(eeg_grp)

            if signals is None and "data" in h5f:
                signals = np.array(h5f["data"])

            if signals is None:
                # Search for any 2D dataset
                for k in h5f.keys():
                    ds = h5f[k]
                    if isinstance(ds, h5py.Dataset) and ds.ndim >= 2:
                        signals = np.array(ds)
                        break

    if signals is None:
        raise ValueError(
            f"Failed to extract EEG signal array from file '{p.name}'. "
            f"Run 'python -m src.data.load_modma --inspect \"{p}\"' to inspect available variables."
        )

    # Ensure array is 2D
    if signals.ndim > 2:
        signals = np.squeeze(signals)
        if signals.ndim != 2:
            raise ValueError(
                f"Expected 2D EEG array from '{p.name}', but after squeeze got shape {signals.shape}."
            )

    # Determine channel orientation adaptively
    d0, d1 = signals.shape
    if d0 in (128, 129) and d1 > d0:
        # Channels along Axis 0
        pass
    elif d1 in (128, 129) and d0 > d1:
        # Channels along Axis 1: transpose to (channels, time)
        signals = signals.T
    elif d0 < d1 and d0 <= 256:
        # Presumed (channels, time)
        pass
    elif d1 < d0 and d1 <= 256:
        # Presumed (time, channels): transpose
        signals = signals.T
    else:
        # Ambiguous shape
        raise ValueError(
            f"Ambiguous EEG signal dimensions in '{p.name}': shape {signals.shape}. "
            f"Cannot reliably determine which axis represents EEG channels."
        )

    n_chan, n_samples = signals.shape

    # Handle 129 channels (128 scalp channels + Cz reference) per MODMA paper
    if n_chan == 129:
        signals = signals[:expected_channels, :]
    elif n_chan > expected_channels:
        signals = signals[:expected_channels, :]
    elif n_chan < expected_channels:
        raise ValueError(
            f"Recording in '{p.name}' has only {n_chan} channels; {expected_channels} channels required."
        )

    # Validate signal length: at 250 Hz, 4 seconds requires at least 1000 samples
    if n_samples < 1000:
        raise ValueError(
            f"Recording in '{p.name}' is too short ({n_samples} samples). "
            f"At least 1000 samples (4 seconds at 250 Hz) are required for epoch generation."
        )

    # Ensure consistent channel names
    if ch_names is None or len(ch_names) != expected_channels:
        ch_names = [f"E{i+1}" for i in range(expected_channels)]

    return signals.astype(np.float32, copy=False), float(fs), ch_names, subject_id, label


def scan_raw_dataset(
    raw_dir: Union[str, Path] = "src/data/raw",
) -> List[Path]:
    """Scan raw directory (and its subdirectories) for MODMA resting-state .mat files.

    Excludes 3-channel and ERP directories to preserve experimental integrity.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []

    mat_files: List[Path] = []
    for root, dirs, files in os.walk(raw_path):
        root_path = Path(root)
        folder_name = root_path.name.lower()

        # Reject ERP and 3-channel folders per project distinction requirements
        if "erp" in folder_name or "3channel" in folder_name:
            continue

        for f in sorted(files):
            # Check for .mat file
            if f.endswith(".mat") or ".mat" in f.lower():
                # Check for resting-state naming or subject pattern
                f_lower = f.lower()
                if "dot" in f_lower or "erp" in f_lower:
                    continue  # skip ERP / task recordings
                mat_files.append(root_path / f)

    return sorted(mat_files)


def diagnose_raw_directory(
    raw_dir: Union[str, Path] = "src/data/raw",
) -> None:
    """Print a diagnostic report for all recordings in the raw directory."""
    raw_path = Path(raw_dir)
    print("=" * 80)
    print(f"CDL-MDDNet: MODMA Local EEG Recording Diagnostic Tool")
    print(f"Scanning directory: {raw_path.resolve()}")
    print("=" * 80)

    if not raw_path.exists():
        print(f"\n[!] Directory does not exist: {raw_path}")
        print("    Please create the directory and place the MODMA .mat recordings inside.")
        return

    files = scan_raw_dataset(raw_path)
    if not files:
        print(f"\n[!] No resting-state .mat recordings found in {raw_path}.")
        print("    Instructions:")
        print("    1. Download 'EEG_128channels_resting_lanzhou_2015' from the MODMA dataset.")
        print(f"    2. Copy or extract the .mat files into: {raw_path.resolve()}")
        print("    3. Expected filename format: '02010002rest 20150416 1017..mat', etc.")
        print("=" * 80)
        return

    print(f"\nFound {len(files)} candidate resting-state .mat recording(s):\n")

    mdd_count = 0
    hc_count = 0
    errors = 0

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] File: {f.name}")
        try:
            info = inspect_mat_file(f)
            print(f"    Size: {info['file_size_mb']} MB | Format: {info['mat_version']}")
            print(f"    Subject ID: {info['subject_id']} | Label: {info['label']} ({info['diagnosis']})")
            if info["diagnosis"] == "MDD":
                mdd_count += 1
            elif info["diagnosis"] == "Healthy Control":
                hc_count += 1

            print(f"    EEG Variable: {info['eeg_variable']} | Shape: {info['detected_shape']}")
            print(f"    Channels: {info['channel_dim']} | Time: {info['time_dim']}")
            if info["notes"]:
                for n in info["notes"]:
                    print(f"    * Note: {n}")
        except Exception as e:
            errors += 1
            print(f"    [!] Error inspecting file: {e}")
        print("-" * 80)

    print(f"\nSummary:")
    print(f"  Total files inspected: {len(files)}")
    print(f"  MDD participants identified: {mdd_count}")
    print(f"  Healthy Control participants identified: {hc_count}")
    print(f"  Inspection errors: {errors}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for MODMA recording diagnostics."""
    parser = argparse.ArgumentParser(
        description="Inspect and diagnose local MODMA MATLAB EEG recordings."
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Directory or .mat file to inspect.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific .mat file to inspect.",
    )
    args = parser.parse_args()

    target = args.file or args.inspect or "src/data/raw"
    target_path = Path(target)

    if target_path.is_file():
        info = inspect_mat_file(target_path)
        print("=" * 80)
        print(f"Inspection Report: {target_path.name}")
        print("=" * 80)
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("=" * 80)
    else:
        diagnose_raw_directory(target_path)


if __name__ == "__main__":
    main()

