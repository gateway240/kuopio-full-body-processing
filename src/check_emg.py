from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample

# max_workers: The maximum number of processes that can be used to
#     execute the given calls. If None or not given then as many
#     worker processes will be created as the machine has processors.
MAX_WORKERS = 12

EMG_SENSORS = {
    "LD_Right",
    "ST_Left",
    "LD_Left",
    "VM_Left",
    "RF_Left",
    "VL_Right",
    "VM_Right",
    "BF_Left",
    "VL_Left",
    "RF_Right",
    "ST_Right",
    "BF_Right",
    "GM_Right",
    "GM_Left",
    "TT_Left",
    "DM_Left",
    "TA_Left",
    "TD_Left",	
    "TT_Right",
    "DM_Right",
    "TA_Right",	
    "TD_Right"
}
KNOWN_TRIALS = {
    "back_fly",
    "box_combos",
    "box_jabs",
    "crab_walking",
    "crouch_lift",
    "crouch_rotate",
    "curls",
    "dyn_sara",
    "dyn_score",
    "half_jacks",
    "heavy_lift",
    "jogging",
    "kettlebell",
    "side_fly",
    "squat_jumps",
    "static_cal",
    "walking"
}

def _read_file_without_header(
    file_path: Path, sep: str = "\t"
) -> pd.DataFrame:
    print("Starting on: ",file_path)
    with open(file_path, "r") as file:
        # Skip header
        for line in file:
            if line.strip() == "endheader":
                break

        # Read remaining data
        df = pd.read_csv(file, sep=sep, index_col=0, low_memory=False)

    # --- Split vector-like columns ---
    def is_vector_column(series: pd.Series) -> bool:
        # Check first non-null value
        sample = series.dropna().astype(str).head(1)
        if sample.empty:
            return False
        return "," in sample.iloc[0]

    new_cols = {}
    cols_to_drop = []

    for col in df.columns:
        if not is_vector_column(df[col]):
            continue

        # Parse column into array
        arr = np.vstack(
            df[col]
            .astype(str)
            .apply(lambda x: np.fromstring(x, sep=","))
            .values
        )
        if arr.shape[1] == 3:
            # Create new columns
            new_cols[f"{col}_x"] = arr[:, 0]
            new_cols[f"{col}_y"] = arr[:, 1]
            new_cols[f"{col}_z"] = arr[:, 2]
        elif arr.shape[1] == 4:
            new_cols[f"{col}_w"] = arr[:, 0]
            new_cols[f"{col}_x"] = arr[:, 1]
            new_cols[f"{col}_y"] = arr[:, 2]
            new_cols[f"{col}_z"] = arr[:, 3]
        else:
            raise ValueError(f"Column {col}: unknown column shape {arr.shape}")
        cols_to_drop.append(col)

    # Replace columns
    df = df.drop(columns=cols_to_drop)
    for k, v in new_cols.items():
        df[k] = v

    return df.apply(pd.to_numeric, errors="coerce")


# ---------------------------
# 1. PAIR FILES BY TRIAL
# ---------------------------
def filter_motion_trials(trials: dict, known_trials: set):
    """
    Filters collected motion trials to only include known trial names.

    Args:
        trials: output of collect_motion_files()
        known_trials: list of allowed trial names

    Returns:
        filtered dict with only matching trials
    """


    known_set = set(known_trials)  # ensure fast lookup

    filtered = {
        key: data
        for key, data in trials.items()
        if key[1] in known_set
    }

    return filtered

def is_versioned_trial(filename: str, suffix: str) -> bool:
    """
    Detects files like:
        dyn_score_hip-1_accelerations.sto -> True
        dyn_score_hip_accelerations.sto   -> False
    """

    # remove suffix first
    base = filename.replace(suffix, "")

    # check for "-number" at the end
    if "-" not in base:
        return False

    last_part = base.split("-")[-1]

    return last_part.isdigit()

def collect_motion_files(root_dir: str):
    trials = {}

    for participant in os.listdir(root_dir):
        imu_dir = os.path.join(root_dir, participant, "imu")
        mocap_dir = os.path.join(root_dir, participant, "mocap")
        print("IMU dir: ", imu_dir, " Mocap dir: ", mocap_dir)
        if not os.path.isdir(imu_dir) or not os.path.isdir(mocap_dir):
            print("ERROR in dir!")
            continue

        # index mocap
        analog_files = {
            f.replace("_analog.sto", ""): os.path.join(mocap_dir, f)
            for f in os.listdir(mocap_dir)
            if f.endswith("_analog.sto")
        }

        # match
        for trial_name, path in analog_files.items():
            # if trial_name in sto_acceleration_files and trial_name in sto_orientation_files:
            trials[(participant,trial_name)] = {
                "participant": participant,
                "analog": path
            }
    # print(trials)
    return trials

# ---------------------------
# 3. SIGNAL PROCESSING
# ---------------------------

def butter_bandpass_filter(
    data: pd.DataFrame,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    order: int = 4,
) -> pd.DataFrame:

    if len(data) < 2:
        raise ValueError("Not enough samples to compute sampling rate")

    # ---- SAMPLING RATE FROM INDEX ----
    dt = float(data.index[-1]) - float(data.index[0])
    if dt <= 0:
        raise ValueError(f"Invalid time range: dt={dt}")

    sampling_rate = len(data) / dt
    print(f"Sampling rate: {sampling_rate:.2f} Hz")

    nyquist = sampling_rate / 2

    if highcut >= nyquist:
        raise ValueError(
            f"highcut ({highcut}) must be < Nyquist ({nyquist})"
        )

    # ---- NORMALIZED FREQUENCIES ----
    low = lowcut / nyquist
    high = highcut / nyquist

    # ---- FILTER COEFFICIENTS ----
    b, a = butter(order, [low, high], btype="band")

    # ---- SELECT COLUMNS ----
    exclude_cols = ["Frame#", "time", "Time"]
    cols_to_filter = [c for c in data.columns if c not in exclude_cols]

    # ---- HANDLE MISSING VALUES ----
    data_interp = data.copy()
    data_interp[cols_to_filter] = (
        data_interp[cols_to_filter]
        .interpolate(method="linear", limit_direction="both")
    )

    # ---- APPLY FILTER (vectorized) ----
    filtered_values = filtfilt(
        b, a, data_interp[cols_to_filter].values, axis=0
    )

    # ---- RETURN ----
    filtered_df = data.copy()
    filtered_df[cols_to_filter] = filtered_values

    return filtered_df

def downsample(df: pd.DataFrame, target_fs: float, current_fs: float):
    df = df.copy()
    # print(df.index)
    target_dt = pd.to_timedelta(1 / target_fs, unit="s")
    return df.resample(rule = pd.to_timedelta(target_dt)).mean()

def downsample_np(x: np.ndarray, target_fs: float, current_fs: float):
    n_samples = int(len(x) * target_fs / current_fs)
    return resample(x, n_samples)



# ---------------------------
# 4. SINGLE TRIAL PROCESSING
# ---------------------------

def compute_snr(signal, baseline_len=5000, window_size=1000, step=200):
    signal = np.asarray(signal)

    if len(signal) < baseline_len:
        raise ValueError("Signal shorter than baseline length")

    # ---- BASELINE (NOISE) ----
    baseline = signal[:baseline_len]
    noise_power = np.mean(baseline ** 2)

    if noise_power == 0:
        return np.inf

    # ---- FIND WINDOW WITH MAX POWER ----
    max_power = -np.inf
    best_start = 0

    for start in range(baseline_len, len(signal) - window_size + 1, step):
        window = signal[start:start + window_size]
        power = np.mean(window ** 2)

        if power > max_power:
            max_power = power
            best_start = start

    signal_power = max_power

    # ---- SNR ----
    snr = 10 * np.log10(signal_power / noise_power)

    return snr, best_start


def plot_emg_signals(df, snr_dict, best_windows,baseline_size, window_size, save_path):
    n_cols = len(df.columns)
    fig, axs = plt.subplots(n_cols, 1, figsize=(12, 3 * n_cols), squeeze=False)

    for i, col in enumerate(df.columns):
        ax = axs[i, 0]
        signal = df[col].values

        ax.plot(signal, label=col)

        start = 0
        end = baseline_size
        ax.axvspan(start, end, color='green', alpha=0.3, label="Min power window")

        # Highlight best window
        start = best_windows[col]
        end = start + window_size
        ax.axvspan(start, end, color='red', alpha=0.3, label="Max power window")

        ax.set_title(f"{col} (SNR={snr_dict[col]:.2f} dB)")
        ax.set_xlabel("Samples")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved EMG plot to: {save_path}")


def _process_single_trial(args):
    participant, trial_name, info, output_dir = args
    error = -1234
    missing = {}
    try:
        analog = _read_file_without_header(Path(info["analog"]))

        # ---- CHECK REQUIRED COLUMNS ----
        missing = EMG_SENSORS - set(analog.columns)
        if missing:
            print(f"Missing required EMG columns: {missing}")

        # Keep only EMG columns
        emg_df = analog[list(EMG_SENSORS)]
        # ---- APPLY BANDPASS FILTER ----
        # https://wiki.has-motion.com/doku.php?id=visual3d:tutorials:emg:typical_emg_processing
        emg_df = butter_bandpass_filter(emg_df, lowcut=50, highcut=500, order=4)

        # ---- COMPUTE SNR ----
        snr_dict = {}
        best_windows = {}

        baseline_size = 2500
        window_size = 2500
        for col in emg_df.columns:
            snr, best_start = compute_snr(emg_df[col].values, baseline_size, window_size)
            snr_dict[col] = snr
            best_windows[col] = best_start

        # ---- PLOT ----
        save_path = Path(output_dir) / f"{participant}_{trial_name}_emg.png"
        plot_emg_signals(emg_df, snr_dict, best_windows,baseline_size, window_size, save_path)

    except Exception as e:
        print("ERROR: ", info, e)
        snr_dict = {}

    result = {
        "participant": participant,
        "trial": trial_name,
        "error_mean": float(np.mean(error)),
        "error_std": float(np.std(error)),
        "missing": missing,
        "snr": snr_dict,
    }

    return result


# ---------------------------
# 5. PIPELINE
# ---------------------------
def process_motion_files(
    motions: Dict,
    output_dir: Path,
):
    results = []

    tasks = [
        (participant, trial, info, output_dir)
        for (participant, trial), info in motions.items()
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_process_single_trial, t) for t in tasks]

        for future in as_completed(futures):
            results.append(future.result())

    return pd.DataFrame(results)


def main() -> None:
    # This makes a non-interactive backend to prevent memory leak
    # see https://github.com/matplotlib/matplotlib/issues/20300
    plt.switch_backend("agg")
    parser = argparse.ArgumentParser(description="Check files")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        help="Directory to save output CSV (default: current directory)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="If set, do not write any output files."
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    Path.mkdir(output_dir, parents=True,exist_ok=True)
    motions_raw = collect_motion_files(args.source_dir)
    print(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    summary_df = process_motion_files(motions, output_dir)
    # summary_df = summary_df.drop("file", axis=1)
    # summary_df = summary_df.drop("df", axis=1)
    summary_df = summary_df.sort_values(["participant", "trial"])
    print(summary_df)
    output_file = output_dir / "emg-check.csv"
    summary_df.to_csv(output_file, index=False)


    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
