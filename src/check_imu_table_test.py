from __future__ import annotations

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


def read_sto_file(filepath: Path)-> pd.DataFrame:
    with open(filepath, "r") as f:
        lines = f.readlines()


    # Find the last non-empty line (this contains column names)
    header_idx = None

    for i, line in enumerate(lines):
        if line.strip() == "":  # blank line
            # next non-empty line is header
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    header_idx = j
                    break
            break

    if header_idx is None:
        raise ValueError("Could not find header line with columns.")

    df = pd.read_csv(
        filepath,
        sep="\t",
        skiprows=header_idx,
        header=0,
        index_col=0
    )

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
            if not fname.endswith("orientations.sto"):
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
    print(df)

    output_file = output_dir / f"{participant}.png"

    plot_euler(df, output_file)

    return {
        "participant": participant,
        "motion": motion,
        "file": f,
        "df": df,
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
                if any(c.isdigit() for c in str(x[0])) else str(x[0])
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
    motions: Dict[Tuple[str, str], List[str]], output_dir: Path, dry_run: bool = True
) -> pd.DataFrame:
    summary_rows = []

    tasks = []
    for (participant, motion), files in motions.items():
        print("Starting: ", participant, motion)
        for f in files:
            tasks.append((participant, motion, f, output_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_single_file, t) for t in tasks]

        for future in as_completed(futures):
            summary_rows.append(future.result())

    summary_df = pd.DataFrame(summary_rows)
    aggregate_and_plot(summary_df, output_dir)
    return summary_df


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

    motions = collect_motion_files(args.source_dir)
    # print(motions)
    summary_df = process_motion_files(motions, output_dir, args.dry_run)
    summary_df = summary_df.drop("file", axis=1)
    summary_df = summary_df.sort_values(["participant"])
    print(summary_df)
    output_file = output_dir / "imu-table-test.csv"
    summary_df.to_csv(output_file, index=False)


    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
