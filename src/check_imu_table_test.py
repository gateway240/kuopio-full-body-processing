from __future__ import annotations
import pathlib

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

TEST_FILENAME = "table_test_orientations.sto"

PLOT_RESULTS = False


def read_sto_file(filepath: Path) -> pd.DataFrame:
    print("Starting on: ", filepath)
    with open(filepath, "r") as file:
        # Skip header
        for line in file:
            if line.strip() == "endheader":
                break
        df = pd.read_csv(file, sep="\t", header=0, index_col=0)

    return df


def collect_motion_files(
    root_dir: str,
) -> DefaultDict[Tuple[str, str], List[str]]:
    """
    Key = (participant, motion)
    """
    motions: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    for participant in os.listdir(root_dir):
        dir: str = os.path.join(root_dir, participant, "imu")
        if not os.path.isdir(dir):
            continue

        for fname in os.listdir(dir):
            if not fname == TEST_FILENAME:
                continue

            motion = fname.rsplit("-", 1)[0]
            path = os.path.join(dir, fname)
            motions[(participant, motion)].append(path)

    return motions


def parse_quaternion(q_str):
    """Convert string 'w,x,y,z' → list of floats"""
    return np.array([float(x) for x in q_str.split(",")])


def quaternion_series_to_euler(series):
    """Convert a pandas Series of quaternion strings to Euler angles"""
    quats = np.vstack(series.dropna().apply(parse_quaternion).values)

    # scipy expects [x, y, z, w], so reorder if needed
    # assuming your format is [w, x, y, z]
    quats_xyzw = np.column_stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]])

    rotations = R.from_quat(quats_xyzw)
    euler = rotations.as_euler("xyz", degrees=True)  # roll, pitch, yaw

    return euler


def plot_euler(df, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax_roll, ax_pitch, ax_yaw = axes

    for col in df.columns:
        try:
            euler = quaternion_series_to_euler(df[col])

            ax_roll.plot(euler[:, 0], label=col)
            ax_pitch.plot(euler[:, 1], label=col)
            ax_yaw.plot(euler[:, 2], label=col)

        except Exception as e:
            print(f"Skipping column {col}: {e}")

    # Formatting
    ax_roll.set_title("Roll (X)")
    ax_pitch.set_title("Pitch (Y)")
    ax_yaw.set_title("Yaw (Z)")

    for ax in axes:
        ax.set_ylabel("Degrees")
        ax.grid(True)
        ax.legend(fontsize=8)

    ax_yaw.set_xlabel("Frame")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _process_single_file(args):
    participant, motion, f, output_dir = args

    path = Path(f)
    df = read_sto_file(path)
    # print(df)

    df_euler = pd.concat(
        [
            pd.DataFrame(
                quaternion_series_to_euler(df[col]),
                columns=[
                    f"{col}_roll",
                    f"{col}_pitch",
                    f"{col}_yaw",
                ],
                index=df[col].dropna().index,
            )
            for col in df.columns
        ],
        axis=1,
    )
    print(df_euler)
    means, stds = df_euler.mean().to_numpy(), df_euler.std().to_numpy()

    # interleave
    summary = pd.DataFrame(
        np.column_stack([means, stds]).reshape(1, -1),
        columns=[f"{c}_{stat}" for c in df_euler.columns for stat in ["mean", "std"]],
    )
    print(summary)
    summary_dict = summary.iloc[0].to_dict()

    if PLOT_RESULTS:
        output_file = output_dir / f"{participant}.png"
        plot_euler(df, output_file)

    return {
        "participant": participant,
        "trial": motion,
        "file": f,
        "df": df,
        **summary_dict,
    }


def aggregate_and_plot(summary_df: pd.DataFrame, output_dir: Path):
    """
    One figure per sensor.
    Each figure shows roll/pitch/yaw.
    Each curve = one participant.
    """

    # group by motion first (optional but usually useful)
    for motion, motion_df in summary_df.groupby("motion"):
        # collect per sensor: sensor -> participant -> euler
        sensor_data = {}

        for _, row in motion_df.iterrows():
            participant = row["participant"]
            df = row["df"]

            for sensor in df.columns:
                try:
                    euler = quaternion_series_to_euler(df[sensor])

                    if sensor not in sensor_data:
                        sensor_data[sensor] = {}

                    sensor_data[sensor][participant] = euler

                except Exception:
                    continue

        # plot per sensor
        for sensor, participants in sensor_data.items():
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            ax_roll, ax_pitch, ax_yaw = axes

            sorted_participants = sorted(
                participants.items(),
                key=lambda x: int("".join(filter(str.isdigit, str(x[0]))))
                if any(c.isdigit() for c in str(x[0]))
                else str(x[0]),
            )

            for participant, euler in sorted_participants:
                ax_roll.plot(euler[:, 0], label=participant)
                ax_pitch.plot(euler[:, 1], label=participant)
                ax_yaw.plot(euler[:, 2], label=participant)

            ax_roll.set_title(f"{motion} - {sensor} Roll")
            ax_pitch.set_title(f"{motion} - {sensor} Pitch")
            ax_yaw.set_title(f"{motion} - {sensor} Yaw")

            for ax in axes:
                ax.set_ylabel("Degrees")
                ax.grid(True)
                ax.legend(fontsize=7, loc="upper right")

            ax_yaw.set_xlabel("Frame")

            fig.tight_layout()

            out_file = output_dir / f"{motion}_{sensor}.png"
            fig.savefig(out_file)
            plt.close(fig)


def process_motion_files(
    motions: Dict[Tuple[str, str], List[str]], output_dir: Path
) -> pd.DataFrame:
    tasks_stage1 = [
        (participant, trial, f, output_dir)
        for (participant, trial), files in motions.items()
        for f in files
    ]

    with ProcessPoolExecutor() as executor:
        try:
            results = list(executor.map(_process_single_file, tasks_stage1))
        except KeyboardInterrupt:
            print("Interrupted")
            executor.shutdown(cancel_futures=True)
            raise

    summary_df = pd.DataFrame(results)
    if PLOT_RESULTS:
        aggregate_and_plot(summary_df, output_dir)
    return summary_df


def summary_table(df_subset: pd.DataFrame, index_list: list[str]) -> pd.DataFrame:
    mean_row = df_subset.mean().round(2)
    sd_row = df_subset.std().round(2)
    range_row = df_subset.apply(lambda x: f"{x.min():.1f}–{x.max():.1f}")
    return pd.DataFrame([mean_row, sd_row, range_row], index=index_list)


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

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    motions = collect_motion_files(args.source_dir)
    # print(motions)
    summary_df = process_motion_files(motions, output_dir)
    summary_df = summary_df.drop(["file", "trial", "df"], axis=1)
    summary_df = summary_df.sort_values(["participant"])
    print(summary_df)
    output_file = output_dir / "imu-table-test-per-sensors.csv"
    summary_df.to_csv(output_file, index=False)

    # Calculate across all sensors
    roll_cols = [c for c in summary_df.columns if "_roll_mean" in c]
    pitch_cols = [c for c in summary_df.columns if "_pitch_mean" in c]
    yaw_cols = [c for c in summary_df.columns if "_yaw_mean" in c]
    summary_stats = pd.DataFrame(
        {
            "participant": summary_df["participant"],
            "roll_mean": summary_df[roll_cols].mean(axis=1),
            "roll_std": summary_df[roll_cols].std(axis=1),
            "roll_delta": summary_df[roll_cols].max(axis=1)
            - summary_df[roll_cols].min(axis=1),
            "pitch_mean": summary_df[pitch_cols].mean(axis=1),
            "pitch_std": summary_df[pitch_cols].std(axis=1),
            "pitch_delta": summary_df[pitch_cols].max(axis=1)
            - summary_df[pitch_cols].min(axis=1),
            "yaw_mean": summary_df[yaw_cols].mean(axis=1),
            "yaw_std": summary_df[yaw_cols].std(axis=1),
            "yaw_delta": summary_df[yaw_cols].max(axis=1)
            - summary_df[yaw_cols].min(axis=1),
        }
    )

    print(summary_stats)

    output_file = output_dir / "imu-table-test-per-participant.csv"
    summary_stats.to_csv(output_file, index=False)

    rename_map = {
        "participant": "\#",
        "roll_mean": r"X $\mu$",
        "roll_std": r"X $\sigma$",
        "roll_delta": r"X $\Delta$",
        "pitch_mean": r"Y $\mu$",
        "pitch_std": r"Y $\sigma$",
        "pitch_delta": r"Y $\Delta$",
        "yaw_mean": r"Z $\mu$",
        "yaw_std": r"Z $\sigma$",
        "yaw_delta": r"Z $\Delta$",
    }

    summary_stats = summary_stats.rename(columns=rename_map)

    latex = summary_stats.to_latex(
        index=False,
        caption=(
            "IMU table test (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) for each participant (\#). "
            "All values are presented in degrees (°). "
            "X,Y, and Z represent roll, pitch, and yaw respectively. "
            "Sensors were placed on a flat, non-metallic table with the same orientation "
            "after being removed from the participant at the end of each data collection session."
        ),
        label="tab:imu_table_test_per_participant",
        escape=False,
        float_format="%.2f",
    )

    print(latex)
    output_file_latex = pathlib.Path("out") / "imu-table-test-per-participant.txt"
    with open(output_file_latex, "w", newline="") as file:
        file.write(latex)

    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
