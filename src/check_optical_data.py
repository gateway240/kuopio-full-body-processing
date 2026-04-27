from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import pandas as pd

PARTICIPANT_MARKERS = {
    "C7",
    "CLAV",
    "LFRM",
    "LTIAP",
    "LTOE",
    "RTHAP",
    "STRN",
    "LTIAD",
    "RFIN",
    "LTHAP",
    "LSMH",
    "LKNE",
    "LUPA",
    "RELB",
    "IMU_RTIB",
    "RKNE",
    "RTHAD",
    "RMED",
    "LELB",
    "RBAK",
    "LTHI",
    "LTHAD",
    "LFHD",
    "RUPA",
    "LWRB",
    "RFHD",
    "LKNM",
    "RTIAP",
    "LASI",
    "RSHO",
    "LBHD",
    "RFRM",
    "RANK",
    "RVMH",
    "RKNM",
    "RHEE",
    "IMU_PELVIS",
    "LVMH",
    "RTIB",
    "RTIAD",
    "RTHI",
    "RWRB",
    "RWRA",
    "RSMH",
    "IMU_LTIB",
    "LFIN",
    "LPSI",
    "LSHO",
    "RPSI",
    "LFMH",
    "LTIB",
    "RFMH",
    "LHEE",
    "LWRA",
    "RASI",
    "LANK",
    "RTOE",
    "RBHD",
    "LMED",
    "T10",
}
BAG_MARKERS =  {'BRF', 'BLF', 'BRB', 'BBS', 'BMS', 'BLB', 'BBR'}
TOTE_MARKERS =  {'TRSB', 'TLFB', 'TLFT', 'TRFT'}


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
    star_columns_count = df.columns.str.startswith("*").sum()
    print("Columns starting with '*':", star_columns_count)

    df = df.loc[:, df.columns.str.startswith(tuple(PARTICIPANT_MARKERS))]
    present_markers = {
        marker
        for marker in PARTICIPANT_MARKERS
        if any(col.startswith(marker) for col in df.columns)
    }

    missing_markers = PARTICIPANT_MARKERS - present_markers

    missing_count = len(missing_markers)

    print("Missing markers:", missing_markers)
    print("Number of missing markers:", missing_count)

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
        "star_columns": star_columns_count,
        "nan_count": total_nan_count,
        "nan_percent": nan_ratio * 100,
        "missing_count": missing_count,
        "missing_markers": missing_markers,
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
    output_file = output_dir / "optical-data-check.csv"
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
