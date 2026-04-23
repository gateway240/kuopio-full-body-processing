from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

KNOWN_TRIALS = {
    # "back_fly",
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
    # "side_fly",
    "squat_jumps",
    "static_cal",
    "walking"
}

def _read_imu_file_without_header(
    file_path: Path, sep: str = "\t"
) -> pd.DataFrame:
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

        # Parse column into Nx3 array
        arr = np.vstack(
            df[col]
            .astype(str)
            .apply(lambda x: np.fromstring(x, sep=","))
            .values
        )

        if arr.shape[1] != 3:
            raise ValueError(f"Column {col}: expected 3 values, got {arr.shape}")

        # Create new columns
        new_cols[f"{col}_x"] = arr[:, 0]
        new_cols[f"{col}_y"] = arr[:, 1]
        new_cols[f"{col}_z"] = arr[:, 2]

        cols_to_drop.append(col)

    # Replace columns
    df = df.drop(columns=cols_to_drop)
    for k, v in new_cols.items():
        df[k] = v

    return df.apply(pd.to_numeric, errors="coerce")


def read_opensim_marker_file(
    file_path: Path,
    index_col: str | int = 0,
    skip: int = 7,
    sep: str = "\t",
) -> pd.DataFrame:

    # Read with multi-level header (two rows)
    raw = pd.read_csv(file_path, sep=sep, header=None, skiprows=skip, low_memory=False)

    # Extract the two header rows
    header1 = raw.iloc[0].ffill()  # marker names (forward fill!)
    # print(header1)
    header2 = raw.iloc[1]  # X1, Y1, Z1...

    # Build clean column names
    cols = []
    for h1, h2 in zip(header1, header2, strict=False):
        if pd.isna(h1):
            cols.append(str(h2))
            continue

        if isinstance(h2, str):
            if h2.startswith("X"):
                suffix = "x"
            elif h2.startswith("Y"):
                suffix = "y"
            elif h2.startswith("Z"):
                suffix = "z"
            else:
                suffix = h2.lower()
        else:
            suffix = ""

        cols.append(f"{h1}_{suffix}".rstrip("_"))

    # Build dataframe (skip header rows)
    df = raw.iloc[2:].copy()
    df.columns = cols

    if isinstance(index_col, int):
        df = df.set_index(df.columns[index_col])

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
    """
    Returns:
        Dict[trial_name] = {
            "participant": str,
            "trc": path,
            "sto": path
        }
    """
    trials = {}

    for participant in os.listdir(root_dir):
        imu_dir = os.path.join(root_dir, participant, "imu")
        mocap_dir = os.path.join(root_dir, participant, "mocap")
        print("IMU dir: ", imu_dir, " Mocap dir: ", mocap_dir)
        if not os.path.isdir(imu_dir) or not os.path.isdir(mocap_dir):
            print("ERROR in dir!")
            continue

        # index mocap
        trc_files = {
            f.replace("_markers.trc", ""): os.path.join(mocap_dir, f)
            for f in os.listdir(mocap_dir)
            if f.endswith("_markers.trc")
            and "-" in f.replace("_markers.trc", "")
            and not f.replace("_markers.trc", "").split("-")[-1].isdigit()
            or "-" not in f.replace("_markers.trc", "")
        }
        # print(trc_files)

        # index imu
        sto_files = {
            f.replace("_accelerations.sto", ""): os.path.join(imu_dir, f)
            for f in os.listdir(imu_dir)
            if f.endswith("_accelerations.sto")
            and "-" in f.replace("_accelerations.sto", "")
            and not f.replace("_accelerations.sto", "").split("-")[-1].isdigit()
            or "-" not in f.replace("_accelerations.sto", "")
        }
        print("STO: ")
        for trial_name in sto_files.keys():
            print(repr(trial_name))
        print("END")
        # match
        for trial_name, trc_path in trc_files.items():
            print("TRIAL:" ,repr(trial_name))
            print("TRC:", repr(trial_name))
            print("IN STO KEYS?:", trial_name in sto_files)
            if trial_name in sto_files:
                print("MATCHED KEY:", repr(trial_name))
            if trial_name in sto_files:
                trials[(participant,trial_name)] = {
                    "participant": participant,
                    "trc": trc_path,
                    "sto": sto_files[trial_name],
                }
        # raise ValueError("bads")
    # print(trials)
    return trials

# ---------------------------
# 3. SIGNAL PROCESSING
# ---------------------------
def butter_lowpass_filter(
    data: pd.DataFrame, cutoff: float, order: float
) -> pd.DataFrame:
    # print(data.index)
    if len(data) < 2:
        raise ValueError("Not enough samples to compute sampling rate")

    dt = float(data.index[-1]) - float(data.index[0])
    if dt <= 0:
        raise ValueError(f"Invalid time range: dt={dt}")
    sampling_rate = len(data) / dt
    print(f"Sampling rate: {sampling_rate} Hz")

    normalized_cutoff = cutoff / (sampling_rate / 2)
    # print(normalized_cutoff)
    # Get the filter coefficients
    b, a = butter(order, normalized_cutoff, btype="low")

    exclude_cols = ["Frame#", "time", "Time"]
    cols_to_filter = [c for c in data.columns if c not in exclude_cols]
    # print(cols_to_filter)
    # logger.info(cols_to_filter)
    filtered_values = filtfilt(b, a, data[cols_to_filter], axis=0)

    filtered_df = data.copy()
    filtered_df[cols_to_filter] = filtered_values

    return filtered_df

def downsample(df: pd.DataFrame, target_fs: float, current_fs: float):
    df = df.copy()
    # print(df.index)
    target_dt = pd.to_timedelta(1 / target_fs, unit="s")
    return df.resample(rule = pd.to_timedelta(target_dt)).mean()


def marker_acc_norm(trc: pd.DataFrame, marker="IMU_PELVIS", fps=100):
    coords = trc[[f"{marker}_x", f"{marker}_y", f"{marker}_z"]].values
    coords = coords / 1000 # mm to m
    dt = 1.0 / fps

    # velocity (first derivative)
    vel = np.gradient(coords, dt, axis=0)

    # acceleration (second derivative)
    acc = np.gradient(vel, dt, axis=0)
        # gravity vector (Z-up convention)
    g = np.array([0, 0, -9.81])

    # add gravity → simulated accelerometer signal
    acc_with_gravity = acc + g

    # magnitude of acceleration vector
    acc_norm = np.linalg.norm(acc_with_gravity, axis=1, ord=2)

    return acc_norm
    # norm = np.linalg.norm(coords, axis=1, ord=2)
    # print("MARKER NORM: ", norm)
    # return norm
    # return (norm - norm.min()) / (norm.max() - norm.min())

def imu_acc_norm(sto: pd.DataFrame, acc_col: str):
    print(sto.columns.to_list())
    acc = sto[[f"{acc_col}_x", f"{acc_col}_y", f"{acc_col}_z"]].values
    # attempt gravity removal
    # print(acc)
    # acc = acc - np.mean(acc, axis=0)
    norm = np.linalg.norm(acc, axis=1, ord=2)
    # print("IMU NORM: ", norm)
    # mean = np.mean(norm, axis=1)
    # print("IMU MEAN: ", mean)
    # return mean
    return norm
    # return (norm - norm.min()) / (norm.max() - norm.min())

# ---------------------------
# 4. SINGLE TRIAL PROCESSING
# ---------------------------
def plot_correlation(marker_signal,
                     imu_signal,
                     corr,
                     lags,
                     best_corr,
                     best_lag,
                     save_path="correlation_plot.png"):
    fs = 60.0
    n = len(marker_signal)
    time = np.arange(n) / fs
    # ---- PLOT ----
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    axs[0].plot(time, marker_signal, label="Marker")
    axs[0].plot(time, imu_signal, label="IMU")
    axs[0].set_title(f"Signals (Best Lag={best_lag}, Best Corr={best_corr:.3f})")
    axs[0].set_xlabel("Time (s)")
    axs[0].legend()
    axs[0].grid()

    print("LAGS: ", lags.size)
    print("CORR: ", corr.size)
    axs[1].plot(lags / fs, corr)
    axs[1].axvline(best_lag / fs, color="r", linestyle="--", label="Best lag")
    axs[1].set_title("Cross-correlation")
    axs[1].set_xlabel("Lag (s)")
    axs[1].legend()
    axs[1].grid()


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {save_path}")
def _process_single_trial(args):
    participant, trial_name, info, output_dir = args
    error = -1234
    best_lag = -1234
    best_corr = -1234
    try:
        trc = read_opensim_marker_file(Path(info["trc"]),skip=3, index_col=1)
        sto = _read_imu_file_without_header(Path(info["sto"]))

        # assume sampling rates known
        TRC_FS = 100.0
        IMU_FS = 60.0

        # filter both
        CUTOFF = 30.0
        trc = butter_lowpass_filter(trc, cutoff=CUTOFF, order=4)
        sto = butter_lowpass_filter(sto, cutoff=CUTOFF, order=4)
        # print("Lengths before: ",len(trc), len(sto))
        # downsample TRC to 60 Hz
        sto.index = pd.to_timedelta(sto.index.astype(float), unit="s")
        trc.index = pd.to_timedelta(trc.index.astype(float), unit="s")
        # print("INDEX TEST:")
        # print(sto.index)
        # print(trc.index)
        trc = downsample(trc, target_fs=IMU_FS, current_fs=TRC_FS)
        # print("INDEX TEST:")
        # print(sto.index)
        # print(trc.index)

        # compute signals
        # print(trc)
        # print(sto)


        marker_signal = marker_acc_norm(trc, "IMU_PELVIS", IMU_FS)
        imu_signal = imu_acc_norm(sto, "pelvis_imu")
        n = min(len(marker_signal), len(imu_signal))
        marker_signal = marker_signal[:n]
        imu_signal = imu_signal[:n]

        # align lengths
        print("Lengths: ",len(marker_signal), len(imu_signal))
        # Normalized cross correlation
        print("marker:", marker_signal.shape)
        print("imu:", imu_signal.shape)
        x = marker_signal
        y = imu_signal
        # x = (marker_signal - np.mean(marker_signal)) / np.std(marker_signal)
        # y = (imu_signal - np.mean(imu_signal)) / np.std(imu_signal)

        # print("x:", x.shape)
        # print("y:", y.shape)
        corr = np.correlate(x, y, mode="full")
        # # lag axis (optional but useful)
        # lags = np.arange(-n + 1, n)

        # MATLAB 'coeff' normalization
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        corr = corr / norm

        # lag axis
        lags = np.arange(-n + 1, n)
        # best alignment
        best_lag = lags[np.argmax(corr)]
        best_corr = np.max(corr)
        # corr = best_corr
        print("Best lag:", best_lag)
        print("Max correlation:", best_corr)
        plot_correlation(
            x,
            y,
            corr,
            lags,
            best_corr,
            best_lag,
            save_path= output_dir / f"{participant}-{trial_name}-corr.png"
        )

        error = np.abs(marker_signal - imu_signal)
    except Exception as e:
        print("ERROR: ",e)
    
    result = {
        "trial": trial_name,
        "participant": participant,
        "error_mean": float(np.mean(error)),
        "error_std": float(np.std(error)),
        "best_lag" : float(best_lag),
        "best_corr": float(best_corr)
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

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_single_trial, t) for t in tasks]

        for future in as_completed(futures):
            results.append(future.result())

    return pd.DataFrame(results)


def main() -> None:
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
    os.makedirs(output_dir,exist_ok=True)

    motions_raw = collect_motion_files(args.source_dir)
    print(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    summary_df = process_motion_files(motions, output_dir)
    # summary_df = summary_df.drop("file", axis=1)
    # summary_df = summary_df.drop("df", axis=1)
    # summary_df = summary_df.sort_values(["participant", "motion"])
    print(summary_df)
    output_file = output_dir / "imu-marker-correlation.csv"
    summary_df.to_csv(output_file, index=False)


    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
