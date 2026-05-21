from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# max_workers: The maximum number of processes that can be used to
#     execute the given calls. If None or not given then as many
#     worker processes will be created as the machine has processors.
# MAX_WORKERS = 12
MAX_WORKERS = None


PARTICIPANT_MARKERS_REQUIRED = {
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
    "LKNE",
    "LUPA",
    "RELB",
    "IMU_RTIB",
    "RKNE",
    "RTHAD",
    "LELB",
    "LTHI",
    "LTHAD",
    "LFHD",
    "RUPA",
    "LWRB",
    "RFHD",
    "RTIAP",
    "LASI",
    "RSHO",
    "LBHD",
    "RFRM",
    "RANK",
    "RVMH",
    "RHEE",
    "IMU_PELVIS",
    "LVMH",
    "RTIB",
    "RTIAD",
    "RTHI",
    "RWRB",
    "RWRA",
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
    "T10",
}

PARTICIPANT_MARKERS_CALIBRATION = {
    "RBAK",
    "RKNM",
    "RMED",
    "RSMH",
    "LKNM",
    "LMED",
    "LSMH",
}

BAG_MARKERS = {"BRF", "BLF", "BRB", "BBS", "BMS", "BLB", "BBR"}
TOTE_MARKERS = {"TRSB", "TLFB", "TLFT", "TRFT"}

KNOWN_TRIALS = {
    "static_cal": PARTICIPANT_MARKERS_CALIBRATION | PARTICIPANT_MARKERS_REQUIRED,
    "dyn_sara": PARTICIPANT_MARKERS_CALIBRATION | PARTICIPANT_MARKERS_REQUIRED,
    "dyn_score_hip": PARTICIPANT_MARKERS_CALIBRATION | PARTICIPANT_MARKERS_REQUIRED,
    "dyn_score_ankle": PARTICIPANT_MARKERS_CALIBRATION | PARTICIPANT_MARKERS_REQUIRED,
    "crouch_lift": PARTICIPANT_MARKERS_REQUIRED | TOTE_MARKERS,
    "crouch_rotate": PARTICIPANT_MARKERS_REQUIRED | TOTE_MARKERS,
    "curls": PARTICIPANT_MARKERS_REQUIRED,
    "kettlebell": PARTICIPANT_MARKERS_REQUIRED,
    "squats_deep": PARTICIPANT_MARKERS_REQUIRED,
    "half_jacks": PARTICIPANT_MARKERS_REQUIRED,
    "squat_jumps": PARTICIPANT_MARKERS_REQUIRED,
    "box_jabs": PARTICIPANT_MARKERS_REQUIRED | BAG_MARKERS,
    "box_combos": PARTICIPANT_MARKERS_REQUIRED | BAG_MARKERS,
    # "chair_push_right",
    # "chair_push_left",
    # "arm_hang",
    # "heavy_lift",
    # "back_fly",
    # "side_fly",
    "walking": PARTICIPANT_MARKERS_REQUIRED,
    "jogging": PARTICIPANT_MARKERS_REQUIRED,
    "crab_walking": PARTICIPANT_MARKERS_REQUIRED,
}


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


def filter_motion_trials(
    trials: dict,
    known_trials: dict[str, set[str]],
):
    filtered = {key: data for key, data in trials.items() if key[1] in known_trials}

    return filtered


def collect_motion_files(root_dir: str):
    trials = {}

    for participant in os.listdir(root_dir):
        mocap_dir = os.path.join(root_dir, participant, "mocap")

        # index mocap
        trc_files = {
            f.replace("_markers.trc", ""): os.path.join(mocap_dir, f)
            for f in os.listdir(mocap_dir)
            if f.endswith("_markers.trc")
        }
        # match
        for trial_name, trc_path in trc_files.items():
            trials[(participant, trial_name)] = {
                "participant": participant,
                "trc": trc_path,
            }
    # print(trials)
    return trials


def _process_single_trial(args):
    info, participant, trial, participant_markers = args
    trc = info["trc"]

    df = read_opensim_marker_file(trc, skip=3)
    star_columns_count = df.columns.str.startswith("*").sum()
    print("Columns starting with '*':", star_columns_count)

    df = df.loc[:, df.columns.str.startswith(tuple(participant_markers))]

    present_markers = {
        marker
        for marker in participant_markers
        if any(col.startswith(marker) for col in df.columns)
    }
    missing_markers = participant_markers - present_markers

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
        "trial": trial,
        "file": trc,
        "total_elements": total_elements,
        "star_columns": star_columns_count,
        "nan_count": total_nan_count,
        "nan_percent": nan_ratio * 100,
        "missing_count": missing_count,
        "missing": missing_markers,
    }


def process_motion_files(
    motions: Dict[Tuple[str, str], List[str]], dry_run: bool = True
) -> pd.DataFrame:
    tasks_stage1 = [
        (
            info,
            participant,
            trial,
            KNOWN_TRIALS[trial],
        )
        for (participant, trial), info in motions.items()
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            results = list(executor.map(_process_single_trial, tasks_stage1))
        except KeyboardInterrupt:
            print("Interrupted")
            executor.shutdown(cancel_futures=True)
            raise

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

    motions_raw = collect_motion_files(args.source_dir)
    print(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    print(motions)
    summary_df = process_motion_files(motions, args.dry_run)
    summary_df = summary_df.drop(["file", "star_columns"], axis=1)
    summary_df = summary_df.sort_values(["participant", "trial"])
    print(summary_df)
    # Filter out empty sets so it doesn't print set() in the csv
    summary_df = summary_df.map(
        lambda x: "" if isinstance(x, set) and len(x) == 0 else x
    )
    output_file = output_dir / "optical-continuity.csv"
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
