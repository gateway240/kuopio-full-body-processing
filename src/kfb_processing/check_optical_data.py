from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pandas.io.formats.style_render import ExtFormatter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # logger.info(header1)
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


def get_last_packet_counter(data_lines: list[str]) -> int:
    last_line: str = data_lines[-1]
    return int(last_line.split("\t", maxsplit=1)[0])


def filter_motion_trials(
    trials: dict[tuple[Path, str], dict[str, Path]],
    known_trials: dict[str, set[str]],
) -> dict[tuple[Path, str], dict[str, Path]]:
    return {key: data for key, data in trials.items() if key[1] in known_trials.get(key[0].name, set())}


def collect_motion_files(root_dir: Path) -> dict[tuple[Path, str], dict[str, Path]]:
    trials = {}

    for participant in root_dir.iterdir():
        mocap_dir = root_dir / participant / "mocap"

        # index mocap
        trc_files = {
            f.name.replace("_markers.trc", ""): mocap_dir / f
            for f in Path.iterdir(mocap_dir)
            if f.name.endswith("_markers.trc")
        }
        # match
        for trial_name, trc_path in trc_files.items():
            trials[participant, trial_name] = {
                "participant": participant,
                "trc": trc_path,
            }
    # logger.info(trials)
    return trials


def _process_single_trial(
    info: dict[str, Path],
    participant: Path,
    trial: str,
    participant_markers: set[str],
) -> dict[str, Any]:
    trc = info["trc"]

    df = read_opensim_marker_file(trc, skip=3)
    star_columns_count = df.columns.str.startswith("*").sum()
    logger.info("Columns starting with '*':%d", star_columns_count)

    df = df.loc[:, df.columns.str.startswith(tuple(participant_markers))]

    present_markers = {marker for marker in participant_markers if any(col.startswith(marker) for col in df.columns)}
    missing_markers = participant_markers - present_markers

    missing_count = len(missing_markers)

    logger.info("Missing markers: %d", missing_markers)
    logger.info("Number of missing markers: %d", missing_count)

    total_elements = df.size
    logger.info("Total number of elements: %d", total_elements)

    total_nan_count = df.isna().sum().sum()
    logger.info("Total NaN count: %d", total_nan_count)

    nan_ratio = total_nan_count / total_elements if total_elements > 0 else 0
    logger.info("Percent NaN: %d %%", nan_ratio * 100)

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
    motions: dict[tuple[Path, str], dict[str, Path]],
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
            logger.info("Interrupted")
            executor.shutdown(cancel_futures=True)
            raise

    return pd.DataFrame(results)


def main() -> None:
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

    args = parser.parse_args()
    output_dir = args.output_dir

    motions_raw = collect_motion_files(args.source_dir)
    logger.info(motions_raw)
    motions = filter_motion_trials(motions_raw, KNOWN_TRIALS)
    logger.info(motions)
    summary_df = process_motion_files(motions)
    summary_df = summary_df.drop(["file", "star_columns"], axis=1)
    summary_df = summary_df.sort_values(["participant", "trial"])
    logger.info(summary_df)
    # Filter out empty sets so it doesn't print set() in the csv
    summary_df = summary_df.map(
        lambda x: "" if isinstance(x, set) and len(x) == 0 else x,
    )
    output_file = output_dir / "optical-continuity.csv"
    summary_df.to_csv(output_file, index=False)

    summary_stats = summary_df.groupby("participant", as_index=False).agg(
        mean_nan_percent=("nan_percent", "mean"),
        std_nan_percent=("nan_percent", "std"),
        range_nan_percent=("nan_percent", lambda x: f"{x.min():.1f} – {x.max():.1f}"),  # ruff: ignore[ambiguous-unicode-character-string]
        # range_nan_percent=(
        #     "nan_percent",
        #     lambda x: x.max() - x.min(),
        # ),
    )

    output_dir_latex = Path("out")
    output_file = output_dir_latex / "optical-continuity-per-participant.csv"
    col = summary_stats.columns[0]
    summary_stats.assign(
        **{
            col: summary_stats[col].astype(int).map("{:02d}".format),
        },
    ).to_csv(output_file, index=False)

    rename_map = {
        "participant": r"\#",
        "mean_nan_percent": r"$\mu$",
        "std_nan_percent": r"$\sigma$",
        "range_nan_percent": r"$\Delta$",
    }

    summary_stats = summary_stats[list(rename_map.keys())].rename(columns=rename_map)

    fmt: ExtFormatter = {
        col: "{:.2f}" for col in summary_stats.columns[1:] if pd.api.types.is_numeric_dtype(summary_stats[col])
    }

    latex = (
        summary_stats.style
        .format(fmt)
        .hide(axis="index")
        .to_latex(
            caption=(
                r"Optical continuity  (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) "
                r"for each participant (\#). The values represent the percentage (\%) of optical data points "
                "in all trials for a given participant which contained NaN values."
            ),
            label="tab:optical_continuity_per_participant",
            position_float="centering",
            hrules=True,  # adds \toprule, \midrule, \bottomrule
        )
    )

    logger.info(latex)
    output_file_latex = output_dir_latex / "optical-continuity-per-participant.txt"
    Path(output_file_latex).write_text(latex, encoding="utf-8", newline="")

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
