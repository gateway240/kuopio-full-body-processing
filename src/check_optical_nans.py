from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import pandas as pd

VALID_MARKERS = [
    "BBR",
    "BBS",
    "BLB",
    "BLF",
    "BMS",
    "BRB",
    "BRF",
    "C7",
    "CLAV",
    "IMU_LTIB",
    "IMU_PELVIS",
    "IMU_RTIB",
    "LANK",
    "LASI",
    "LBHD",
    "LELB",
    "LFHD",
    "LFIN",
    "LFMH",
    "LFRM",
    "LHEE",
    "LKNE",
    "LKNM",
    "LMED",
    "LPSI",
    "LSHO",
    "LSMH",
    "LTHAD",
    "LTHAP",
    "LTHI",
    "LTIAD",
    "LTIAP",
    "LTIB",
    "LTOE",
    "LUPA",
    "LVMH",
    "LWRA",
    "LWRB",
    "RANK",
    "RASI",
    "RBAK",
    "RBHD",
    "RELB",
    "RFHD",
    "RFIN",
    "RFMH",
    "RFRM",
    "RHEE",
    "RKNE",
    "RKNM",
    "RMED",
    "RPSI",
    "RSHO",
    "RSMH",
    "RTHAD",
    "RTHAP",
    "RTHI",
    "RTIAD",
    "RTIAP",
    "RTIB",
    "RTOE",
    "RUPA",
    "RVMH",
    "RWRA",
    "RWRB",
    "STRN",
    "T10",
    "TLFB",
    "TLFT",
    "TRFT",
    "TRSB",
]


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


def get_last_packet_counter(data_lines: List[str]) -> int:
    last_line: str = data_lines[-1]
    return int(last_line.split("\t")[0])


def collect_motion_files(
    root_dir: str,
) -> DefaultDict[Tuple[str, str], List[str]]:
    """
    Key = (participant, motion)
    """
    motions: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    for participant in os.listdir(root_dir):
        imu_dir: str = os.path.join(root_dir, participant, "mocap")
        if not os.path.isdir(imu_dir):
            continue

        for fname in os.listdir(imu_dir):
            if not fname.endswith(".trc"):
                continue

            motion = fname.rsplit("-", 1)[0]
            path = os.path.join(imu_dir, fname)
            motions[(participant, motion)].append(path)

    return motions


def _process_single_file(args):
    participant, motion, f = args

    df = read_opensim_marker_file(Path(f), skip=3)
    df = df.loc[:, df.columns.str.startswith(tuple(VALID_MARKERS))]

    total_elements = df.size
    print("Total number of elements:", total_elements)

    total_nan_count = df.isnull().sum().sum()
    print("Total NaN count:", total_nan_count)

    nan_ratio = total_nan_count / total_elements if total_elements > 0 else 0
    print(f"Percent NaN: {nan_ratio * 100}%")

    return {
        "participant": participant,
        "motion": motion,
        "file": f,
        "total_elements": total_elements,
        "nan_count": total_nan_count,
        "nan_percent": nan_ratio * 100,
    }


def process_motion_files(
    motions: Dict[Tuple[str, str], List[str]], dry_run: bool = True
) -> pd.DataFrame:
    summary_rows = []

    tasks = []
    for (participant, motion), files in motions.items():
        print("Starting: ", participant, motion)
        for f in files:
            tasks.append((participant, motion, f))

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

    motions = collect_motion_files(args.source_dir)
    print(motions)
    summary_df = process_motion_files(motions, args.dry_run)
    summary_df = summary_df.drop("file", axis=1)
    summary_df = summary_df.sort_values(["participant", "motion"])
    print(summary_df)
    output_file = output_dir / "nan-check.csv"
    summary_df.to_csv(output_file, index=False)

    participant_summary = summary_df.groupby("participant", as_index=False).agg(
        {
            "total_elements": "sum",
            "nan_count": "sum",
        }
    )

    participant_summary["nan_percent"] = (
        participant_summary["nan_count"] / participant_summary["total_elements"] * 100
    )
    print(participant_summary.to_string(index=False))

    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
