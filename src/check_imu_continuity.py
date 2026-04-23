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


def read_file(filepath: Path)-> pd.DataFrame:
    with open(filepath, "r") as f:
        lines = f.readlines()


    # Find the last non-empty line (this contains column names)
    header_idx = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if line_stripped.startswith("//") or line_stripped == "":
            continue

        # first line that looks like column headers (contains tabs or spaces + text)
        header_idx = i
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
            if not fname.endswith(".txt"):
                continue

            motion = fname.rsplit("-", 1)[0]
            path = os.path.join(dir, fname)
            motions[(participant, motion)].append(path)

    return motions

def validate_motion_df(df: pd.DataFrame, file_path: str):
    errors = []

    # --- 1. Check PacketCounter continuity ---

    packet = df.index.to_series()

    # ensure numeric
    packet = pd.to_numeric(packet, errors="coerce")

    if packet.isna().any():
        errors.append("PacketCounter contains NaNs or non-numeric values")

    MAX_PACKET = 65535

    # check monotonic + no gaps (with overflow handling)
    diff = packet.diff().dropna()

    # valid transitions:
    #  1           -> normal step
    # -65535       -> 65535 => 0 (overflow forward)
    valid_step = (diff == 1) | (diff == -MAX_PACKET)

    bad_steps = diff[~valid_step]

    if not bad_steps.empty:
        missing_frames = [
            (packet.iloc[i - 1], packet.iloc[i])
            for i in bad_steps.index
        ]

        errors.append(f"Missing frames detected: {missing_frames}")

    overflow_events = diff[(diff == -MAX_PACKET)]

    if not overflow_events.empty:
        overflow_list = [
            (packet.iloc[i - 1], packet.iloc[i])
            for i in overflow_events.index
        ]
        errors.append(f"PacketCounter overflow detected: {overflow_list}")

    # --- 2. Check all numeric columns ---
    numeric_df = df.drop(columns=["PacketCounter"], errors="ignore")

    # force numeric conversion
    coerced = numeric_df.apply(pd.to_numeric, errors="coerce")

    if coerced.isna().any().any():
        nan_locs = np.where(coerced.isna())
        errors.append(
            f"Non-numeric/NaN values found at rows={nan_locs[0][:10]}, cols={nan_locs[1][:10]}"
        )

    return errors

def _process_single_file(args):
    participant, motion, f, output_dir = args

    path = Path(f)
    df = read_file(path)

    errors = validate_motion_df(df, str(path))

    if errors:
        print(f"[WARNING] {path}")
        for e in errors:
            print("   -", e)

    return {
        "participant": participant,
        "motion": motion,
        "file": f,
        "df": df,
        "errors": errors
    }
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
    summary_df = summary_df.drop("df", axis=1)
    summary_df = summary_df.sort_values(["participant", "motion"])
    print(summary_df)
    output_file = output_dir / "imu-continuity-test.csv"
    summary_df.to_csv(output_file, index=False)


    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
