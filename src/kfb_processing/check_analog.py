from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# max_workers: The maximum number of processes that can be used to
#     execute the given calls. If None or not given then as many
#     worker processes will be created as the machine has processors.
MAX_WORKERS = None

PLOT_RESULTS = False

# Sampling rate known for system
ANALOG_FS = 2400

WINDOW_SIZE = ANALOG_FS

EMG_SENSORS = {
    # "trigger",
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
    "TD_Right",
}
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
    "chair_push_right",
    "chair_push_left",
    "arm_hang",
    "heavy_lift",
    "back_fly",
    "side_fly",
    "walking",
    "jogging",
    "crab_walking",
}


def _read_file_without_header(file_path: Path, sep: str = "\t") -> pd.DataFrame:
    logger.info("Starting on: %s", file_path)
    with file_path.open("r", encoding="utf-8") as file:
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
            [np.fromstring(str(x), sep=",") for x in df[col]],
        )
        if arr.shape[1] == 3:  # ruff: ignore[magic-value-comparison]
            # Create new columns
            new_cols[f"{col}_x"] = arr[:, 0]
            new_cols[f"{col}_y"] = arr[:, 1]
            new_cols[f"{col}_z"] = arr[:, 2]
        elif arr.shape[1] == 4:  # ruff: ignore[magic-value-comparison]
            new_cols[f"{col}_w"] = arr[:, 0]
            new_cols[f"{col}_x"] = arr[:, 1]
            new_cols[f"{col}_y"] = arr[:, 2]
            new_cols[f"{col}_z"] = arr[:, 3]
        else:
            msg = f"Column {col}: unknown column shape {arr.shape}"
            raise ValueError(msg)
        cols_to_drop.append(col)

    # Replace columns
    df = df.drop(columns=cols_to_drop)
    for k, v in new_cols.items():
        df[k] = v

    return df.apply(pd.to_numeric, errors="coerce")


# ---------------------------
# 1. PAIR FILES BY TRIAL
# ---------------------------
def filter_motion_trials(
    trials: dict[tuple[str, str], dict[str, Path]],
    known_trials: set[str],
) -> dict[tuple[str, str], dict[str, Path]]:
    return {key: data for key, data in trials.items() if key[1] in known_trials}


def collect_motion_files(root_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    trials = {}

    for participant_dir in root_dir.iterdir():
        participant = participant_dir.name
        imu_dir = root_dir / participant / "imu"
        mocap_dir = root_dir / participant / "mocap"
        logger.info("IMU dir: %s Mocap dir: %s", imu_dir, mocap_dir)
        if not imu_dir.is_dir() or not mocap_dir.is_dir():
            logger.info("ERROR in dir!")
            continue

        # index mocap
        analog_files = {
            f.name.replace("_analog.sto", ""): mocap_dir / f.name
            for f in mocap_dir.iterdir()
            if f.name.endswith("_analog.sto")
        }

        # match
        for trial_name, path in analog_files.items():
            # if trial_name in sto_acceleration_files and trial_name in sto_orientation_files:
            trials[participant, trial_name] = {
                "analog": path,
            }
    # logger.info(trials)
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
    if len(data) < 2:  # ruff: ignore[magic-value-comparison]
        msg = "Not enough samples to compute sampling rate"
        raise ValueError(msg)

    sampling_rate = ANALOG_FS
    # logger.info(f"Sampling rate: {sampling_rate:.2f} Hz")

    nyquist = sampling_rate / 2

    if highcut >= nyquist:
        msg_0 = f"highcut ({highcut}) must be < Nyquist ({nyquist})"
        raise ValueError(msg_0)

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    exclude_cols = ["Frame#", "time", "Time"]
    cols_to_filter = [c for c in data.columns if c not in exclude_cols]

    data_interp = data.copy()
    data_interp[cols_to_filter] = data_interp[cols_to_filter].interpolate(
        method="linear",
        limit_direction="both",
    )

    filtered_values = filtfilt(b, a, data_interp[cols_to_filter].values, axis=0)

    filtered_df = data.copy()
    filtered_df[cols_to_filter] = filtered_values

    return filtered_df


# ---------------------------
# 4. SINGLE TRIAL PROCESSING
# ---------------------------


def compute_snr(signal: np.ndarray, baseline_len: int, window_size: int, step: int = 200) -> tuple[float, int]:
    signal = np.asarray(signal)

    if len(signal) < baseline_len:
        msg = "Signal shorter than baseline length"
        raise ValueError(msg)

    # ---- BASELINE (NOISE) ----
    baseline = signal[:baseline_len]
    noise_power = np.mean(baseline**2)

    best_start = 0

    if noise_power == 0:
        return 0, best_start

    # ---- FIND WINDOW WITH MAX POWER ----
    max_power = -np.inf

    for start in range(baseline_len, len(signal) - window_size + 1, step):
        window = signal[start : start + window_size]
        power = np.mean(window**2)

        if power > max_power:
            max_power = power
            best_start = start

    signal_power = max_power

    # ---- SNR ----
    snr: float = 10 * np.log10(signal_power / noise_power)

    return snr, best_start


def plot_emg_signals(df, snr_dict, best_windows, baseline_size, window_size, save_path) -> None:  # ruff: ignore[missing-type-function-argument, too-many-arguments, too-many-positional-arguments]
    n_cols = len(df.columns)
    fig, axs = plt.subplots(n_cols, 1, figsize=(12, 3 * n_cols), squeeze=False)

    for i, col in enumerate(sorted(df.columns)):
        ax = axs[i, 0]
        signal = df[col].to_numpy()

        ax.plot(signal, label=col)

        start = 0
        end = baseline_size
        ax.axvspan(start, end, color="green", alpha=0.3, label="Min power window")

        # Highlight best window
        start = best_windows[col]
        end = start + window_size
        ax.axvspan(start, end, color="red", alpha=0.3, label="Max power window")

        ax.set_title(f"{col} (SNR={snr_dict[col]:.2f} dB)")
        ax.set_xlabel("Samples")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    logger.info("Saved EMG plot to: %s", save_path)


def _process_single_trial(participant: str, trial_name: str, info: Any) -> dict[str, Any]:  # ruff: ignore[any-type]

    analog = _read_file_without_header(Path(info["analog"]))

    # ---- CHECK REQUIRED COLUMNS ----
    missing = EMG_SENSORS - set(analog.columns)
    if missing:
        logger.info("Missing required EMG columns: %s", missing)

    # Keep only EMG columns
    emg_df = analog[[col for col in EMG_SENSORS if col in analog.columns]]
    # ---- APPLY BANDPASS FILTER ----
    # https://wiki.has-motion.com/doku.php?id=visual3d:tutorials:emg:typical_emg_processing
    emg_df_filtered = butter_bandpass_filter(
        emg_df,
        lowcut=50,
        highcut=500,
        order=4,
    )
    return {
        "participant": participant,
        "trial_name": trial_name,
        "raw_analog": emg_df,
        "filtered_analog": emg_df_filtered,
        "missing": missing,
    }


def _calculate_single_trial(info: Any, output_dir: Path) -> dict[str, Any]:  # ruff: ignore[any-type]
    snr_dict = {}

    participant = info["participant"]
    trial_name = info["trial_name"]
    emg_df = info["filtered_analog"]
    missing = info["missing"]
    # ---- COMPUTE SNR ----
    best_windows = {}

    baseline_size = WINDOW_SIZE
    window_size = WINDOW_SIZE
    for col in emg_df.columns:
        snr, best_start = compute_snr(
            emg_df[col].values,
            baseline_size,
            window_size,
        )
        snr_dict[col] = snr
        best_windows[col] = best_start

    if PLOT_RESULTS:
        save_path = output_dir / f"{participant}_{trial_name}_emg.png"
        plot_emg_signals(
            emg_df,
            snr_dict,
            best_windows,
            baseline_size,
            window_size,
            save_path,
        )

    return {
        "participant": participant,
        "trial": trial_name,
        **snr_dict,
        "missing": missing,
    }


# ---------------------------
# 5. PIPELINE
# ---------------------------
def process_motion_files(
    motions: dict[tuple[str, str], dict[str, Path]],
    output_dir: Path,
) -> pd.DataFrame:
    tasks_stage1 = [(participant, trial, info) for (participant, trial), info in motions.items()]

    with Pool(processes=MAX_WORKERS) as executor:
        try:
            results_stage1 = executor.starmap(_process_single_trial, tasks_stage1)
            tasks_stage2 = [(info, output_dir) for info in results_stage1]
            results = executor.starmap(_calculate_single_trial, tasks_stage2)
        except KeyboardInterrupt:
            logger.info("Interrupted")
            executor.terminate()
            executor.join()
            raise

    return pd.DataFrame(results)


def main() -> None:
    # This makes a non-interactive backend to prevent memory leak
    # see https://github.com/matplotlib/matplotlib/issues/20300
    plt.switch_backend("agg")
    parser = argparse.ArgumentParser(description="Check files")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        type=Path,
        help="Directory to save output CSV (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write any output files.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    motions_raw = collect_motion_files(args.source_dir)
    logger.info(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    summary_df = process_motion_files(motions, output_dir)
    summary_df = summary_df.sort_values(["participant", "trial"])
    # Filter out empty sets so it doesn't print set() in the csv
    summary_df = summary_df.map(
        lambda x: "" if isinstance(x, set) and len(x) == 0 else x,
    )
    # fill NaN only in numeric columns
    numeric_cols = summary_df.select_dtypes(include="number").columns
    summary_df[numeric_cols] = summary_df[numeric_cols].fillna(0)
    logger.info(summary_df)
    output_file = output_dir / "emg-snr.csv"
    col = summary_df.columns[0]
    summary_df.assign(**{col: summary_df[col].map(lambda x: f"{int(x):02d}")}).to_csv(  # ty: ignore[invalid-argument-type]
        output_file,
        index=False,
    )

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
