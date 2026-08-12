from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from scipy.spatial.transform import Rotation as R

# max_workers: The maximum number of processes that can be used to
#     execute the given calls. If None or not given then as many
#     worker processes will be created as the machine has processors.
# MAX_WORKERS = 12
MAX_WORKERS = None
# Latitude: 62.8929513
# Height: 85 m above sea level
# https://www.sensorsone.com/local-gravity-calculator/
GRAVITY = 9.82112

# Known for data
TRC_FS = 100.0
IMU_FS = 60.0
# For Low Pass Filtering
CUTOFF = 6.0

PLOT_RESULTS = False

KNOWN_TRIALS = {
    "static_cal",
    "dyn_sara",
    "dyn_score_hip",
    "dyn_score_ankle",
    "crouch_lift",
    "crouch_rotate",
    "curls",
    "kettlebell",
    "squats_deep",
    "half_jacks",
    "squat_jumps",
    "box_jabs",
    "box_combos",
    # "chair_push_right",
    # "chair_push_left",
    # "arm_hang",
    # "heavy_lift",
    # "back_fly",
    # "side_fly",
    "walking",
    "jogging",
    "crab_walking",
}


MARKER_PAIRS = {
    ("IMU_PELVIS", "pelvis_imu"),
    ("LTHI", "femur_l_imu"),
    ("RTHI", "femur_r_imu"),
    ("IMU_LTIB", "tibia_l_imu"),
    ("IMU_RTIB", "tibia_r_imu"),
    ("LTOE", "calcn_l_imu"),
    ("RTOE", "calcn_r_imu"),
    ("CLAV", "torso_imu"),
    ("LUPA", "humerus_l_imu"),
    ("RUPA", "humerus_r_imu"),
    ("LFRM", "radius_l_imu"),
    ("RFRM", "radius_r_imu"),
    ("RBHD", "head_imu"),
}


def _read_imu_file_without_header(file_path: Path, sep: str = "\t") -> pd.DataFrame:
    print("Starting on: ", file_path)
    with Path(file_path).open("r") as file:
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
            df[col].astype(str).apply(lambda x: np.fromstring(x, sep=",")).values,
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
        df.index = df.index.astype(float)

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

    filtered = {key: data for key, data in trials.items() if key[1] in known_set}

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
        if not Path(imu_dir).is_dir() or not Path(mocap_dir).is_dir():
            print("ERROR in dir!")
            continue

        # index mocap
        trc_files = {
            f.replace("_markers.trc", ""): os.path.join(mocap_dir, f)
            for f in os.listdir(mocap_dir)
            if f.endswith("_markers.trc")
        }
        # index imu
        sto_acceleration_files = {
            f.replace("_accelerations.sto", ""): os.path.join(imu_dir, f)
            for f in os.listdir(imu_dir)
            if f.endswith("_accelerations.sto")
        }
        sto_orientation_files = {
            f.replace("_orientations.sto", ""): os.path.join(imu_dir, f)
            for f in os.listdir(imu_dir)
            if f.endswith("_orientations.sto")
        }

        # match
        for trial_name, trc_path in trc_files.items():
            if (
                trial_name in sto_acceleration_files
                and trial_name in sto_orientation_files
            ):
                trials[participant, trial_name] = {
                    "participant": participant,
                    "trc": trc_path,
                    "sto_acceleration": sto_acceleration_files[trial_name],
                    "sto_orientation": sto_orientation_files[trial_name],
                }
    # print(trials)
    return trials


# ---------------------------
# 3. SIGNAL PROCESSING
# ---------------------------
def butter_lowpass_filter(
    data: pd.DataFrame,
    cutoff: float,
    sampling_rate: float,
    order: float,
) -> pd.DataFrame:
    # print(data.index)
    if len(data) < 2:
        raise ValueError("Not enough samples to compute sampling rate")

    df = data.copy()
    # Drop columns with exclusively NAN values
    df = df.dropna(axis=1, how="all")
    print(f"Sampling rate: {sampling_rate} Hz")

    normalized_cutoff = cutoff / (sampling_rate / 2)
    # print(normalized_cutoff)
    # Get the filter coefficients
    b, a = butter(order, normalized_cutoff, btype="low")

    exclude_cols = ["Frame#", "time", "Time"]
    cols_to_filter = [c for c in df.columns if c not in exclude_cols]

    # print(cols_to_filter)
    # logger.info(cols_to_filter)
    # nan_percent = df[cols_to_filter].isna().mean() * 100

    # high_nan_cols = nan_percent[nan_percent > nan_threshold].index
    # low_nan_cols = nan_percent[nan_percent <= threshold].index
    # print(high_nan_cols)
    # First linear interpolation
    df[cols_to_filter] = df[cols_to_filter].interpolate(
        method="linear",
        limit_direction="both",
    )

    # Butterworth low-pass
    df[cols_to_filter] = filtfilt(b, a, df[cols_to_filter], axis=0)

    return df


def downsample(df: pd.DataFrame, target_fs: float, current_fs: float):
    df = df.copy()
    # print(df.index)
    target_dt = pd.to_timedelta(1 / target_fs, unit="s")
    return df.resample(rule=pd.to_timedelta(target_dt)).mean()


def downsample_np(x: np.ndarray, target_fs: float, current_fs: float):
    n_samples = int(round(len(x) * (target_fs / current_fs)))
    return resample(x, n_samples)


def marker_acc_norm(coords: np.ndarray, fps=100):
    coords = coords / 1000  # mm to m
    dt = 1.0 / fps

    # velocity (first derivative)
    vel = np.gradient(coords, dt, axis=0)

    # acceleration (second derivative)
    acc = np.gradient(vel, dt, axis=0)

    # magnitude of acceleration vector
    acc_norm = np.linalg.norm(acc, axis=1, ord=2)

    return acc_norm


def imu_acc_norm(sto_accel: pd.DataFrame, sto_ori: pd.DataFrame, acc_col: str):
    # print(sto_accel.columns.to_list())
    acc = sto_accel[[f"{acc_col}_x", f"{acc_col}_y", f"{acc_col}_z"]].values
    ori = sto_ori[
        [f"{acc_col}_w", f"{acc_col}_x", f"{acc_col}_y", f"{acc_col}_z"]
    ].values
    r_mat = R.from_quat(ori, scalar_first=True)
    g = np.array([0, 0, GRAVITY])
    free_acc = r_mat.apply(acc) - g
    # print(free_acc)
    norm = np.linalg.norm(free_acc, axis=1, ord=2)
    # print("IMU NORM: ", norm)
    return norm


# ---------------------------
# 4. SINGLE TRIAL PROCESSING
# ---------------------------
def plot_correlation(
    marker_signal,
    imu_signal,
    corr,
    lags,
    best_corr,
    best_lag,
    raw_coords=None,
    coords=None,
    save_path="correlation_plot.png",
):
    fs = 60.0
    n = len(marker_signal)
    time = np.arange(n) / fs

    has_coords = (coords is not None) and (raw_coords is not None)
    try:
        # ---- CREATE SUBPLOTS ----
        if has_coords:
            fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
            ax_signal, ax_corr, ax_coords = axs
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
            ax_signal, ax_corr = axs
            ax_coords = None
        # ---- SIGNALS ----
        ax_signal.plot(time, marker_signal, label="Marker")
        ax_signal.plot(time, imu_signal, label="IMU")

        ax_signal.set_title(f"Signals (Best Lag={best_lag}, Best Corr={best_corr:.3f})")
        ax_signal.set_xlabel("Time (s)")
        ax_signal.legend()
        ax_signal.grid()

        # ---- CROSS CORRELATION ----
        ax_corr.plot(lags / fs, corr)
        ax_corr.axvline(best_lag / fs, color="r", linestyle="--", label="Best lag")

        ax_corr.set_title("Cross-correlation")
        ax_corr.set_xlabel("Lag (s)")
        ax_corr.legend()
        ax_corr.grid()

        # ---- RAW + FILTERED COORDINATES (SAME SUBPLOT) ----
        if ax_coords and raw_coords is not None and coords is not None:
            ax_coords.plot(time, raw_coords[:, 0], label="Raw X", alpha=0.6)
            ax_coords.plot(time, raw_coords[:, 1], label="Raw Y", alpha=0.6)
            ax_coords.plot(time, raw_coords[:, 2], label="Raw Z", alpha=0.6)

            ax_coords.plot(time, coords[:, 0], linestyle="--", label="Filtered X")
            ax_coords.plot(time, coords[:, 1], linestyle="--", label="Filtered Y")
            ax_coords.plot(time, coords[:, 2], linestyle="--", label="Filtered Z")

            ax_coords.set_title("Raw vs Filtered Marker Coordinates")
            ax_coords.set_xlabel("Time (s)")
            ax_coords.set_ylabel("Position")
            ax_coords.legend()
            ax_coords.grid()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    finally:
        plt.close("all")

    print(f"Saved plot to: {save_path}")


def _calculate_single_trial(args):
    info, output_dir, marker_name, imu_name = args
    participant = info["participant"]
    trial_name = info["trial_name"]
    best_lag = 0
    best_corr = 0
    marker_len = 0
    marker_nan_count = 0
    marker_nan_percent = 0
    marker_constant_percent = 0
    try:
        trc = info["raw_trc"]
        # assume sampling rates known

        raw_trc = trc[[f"{marker_name}_x", f"{marker_name}_y", f"{marker_name}_z"]]
        marker_len = trc.size
        marker_nan_count = np.isnan(raw_trc.values).sum()
        marker_nan_percent = (marker_nan_count / marker_len) * 100 if marker_len else 0

        raw_coords = raw_trc.ffill().values

        # calculate how much the marker data is constant during the trial
        diff = np.linalg.norm(np.diff(raw_coords, axis=0), axis=1)
        eps = 0.1  # mm
        is_constant = diff < eps
        marker_constant_percent = (np.sum(is_constant) / is_constant.size) * 100
        print("Constant percent:", marker_constant_percent)

        raw_coords_downsample = downsample_np(
            raw_coords,
            target_fs=IMU_FS,
            current_fs=TRC_FS,
        )

        trc_filtered = info["filtered_trc"]
        sto_accel = info["filtered_sto_accel"]
        sto_ori = info["filtered_sto_ori"]

        coords = trc_filtered[
            [f"{marker_name}_x", f"{marker_name}_y", f"{marker_name}_z"]
        ].values
        marker_signal_og = marker_acc_norm(coords, TRC_FS)
        marker_signal = downsample_np(
            marker_signal_og,
            target_fs=IMU_FS,
            current_fs=TRC_FS,
        )
        imu_signal = imu_acc_norm(sto_accel, sto_ori, imu_name)
        n = min(len(marker_signal), len(imu_signal))
        marker_signal = marker_signal[:n]
        imu_signal = imu_signal[:n]

        # align lengths
        print("Lengths: ", len(marker_signal), len(imu_signal))
        # Normalized cross correlation
        print("marker:", marker_signal.shape)
        print("imu:", imu_signal.shape)
        x = marker_signal
        y = imu_signal
        # print("x:", x.shape)
        # print("y:", y.shape)
        corr = np.correlate(x, y, mode="full")

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

        if PLOT_RESULTS:
            coords_downsample = downsample_np(
                coords,
                target_fs=IMU_FS,
                current_fs=TRC_FS,
            )
            plot_correlation(
                x,
                y,
                corr,
                lags,
                best_corr,
                best_lag,
                coords=coords_downsample[:n],
                raw_coords=raw_coords_downsample[:n],
                save_path=output_dir
                / f"{participant}-{trial_name}-{marker_name}-{imu_name}-corr.png",
            )

    except Exception as e:
        print("ERROR: ", info, e)

    result = {
        "participant": participant,
        "trial": trial_name,
        "marker_name": marker_name,
        "imu_name": imu_name,
        "marker_len": marker_len,
        "marker_nan_count": marker_nan_count,
        "marker_nan_percent": marker_nan_percent,
        "marker_constant_percent": marker_constant_percent,
        "best_lag": float(best_lag),
        "best_corr": float(best_corr),
    }

    return result


def _process_single_trial(args):
    participant, trial_name, info = args
    try:
        trc = read_opensim_marker_file(Path(info["trc"]), skip=3, index_col=1)
        sto_accel = _read_imu_file_without_header(Path(info["sto_acceleration"]))
        sto_ori = _read_imu_file_without_header(Path(info["sto_orientation"]))

        trc_filtered = butter_lowpass_filter(
            trc,
            cutoff=CUTOFF,
            sampling_rate=TRC_FS,
            order=4,
        )
        sto_accel = butter_lowpass_filter(
            sto_accel,
            cutoff=CUTOFF,
            sampling_rate=IMU_FS,
            order=4,
        )
        sto_ori = butter_lowpass_filter(
            sto_ori,
            cutoff=CUTOFF,
            sampling_rate=IMU_FS,
            order=4,
        )

    except Exception as e:
        print("ERROR: ", info, e)

    return {
        "participant": participant,
        "trial_name": trial_name,
        "raw_trc": trc,
        "raw_sto_accel": sto_accel,
        "raw_sto_ori": sto_ori,
        "filtered_trc": trc_filtered,
        "filtered_sto_accel": sto_accel,
        "filtered_sto_ori": sto_ori,
    }


# ---------------------------
# 5. PIPELINE
# ---------------------------
def process_motion_files(
    motions: dict,
    output_dir: Path,
):
    tasks_stage1 = [
        (participant, trial, info) for (participant, trial), info in motions.items()
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            results_stage1 = list(executor.map(_process_single_trial, tasks_stage1))
            tasks_stage2 = [
                (info, output_dir, marker_name, imu_name)
                for info in results_stage1
                for (marker_name, imu_name) in MARKER_PAIRS
            ]
            results = list(executor.map(_calculate_single_trial, tasks_stage2))
        except KeyboardInterrupt:
            print("Interrupted")
            executor.shutdown(cancel_futures=True)
            raise

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
        "--dry-run",
        action="store_true",
        help="If set, do not write any output files.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    Path.mkdir(output_dir, exist_ok=True)
    motions_raw = collect_motion_files(args.source_dir)
    print(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    summary_df = process_motion_files(motions, output_dir)
    # summary_df = summary_df.drop("file", axis=1)
    # summary_df = summary_df.drop("df", axis=1)
    summary_df = summary_df.sort_values(["participant", "trial"])
    print(summary_df)
    output_file = output_dir / "imu-marker-sync.csv"
    summary_df.to_csv(output_file, index=False)

    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
