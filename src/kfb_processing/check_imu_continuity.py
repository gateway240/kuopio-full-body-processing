from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_file(filepath: Path) -> pd.DataFrame:
    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the last non-empty line (this contains column names)
    header_idx = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if line_stripped.startswith("//") or not line_stripped:
            continue

        # first line that looks like column headers (contains tabs or spaces + text)
        header_idx = i
        break

    if header_idx is None:
        msg = "Could not find header line with columns."
        raise ValueError(msg)

    return pd.read_csv(filepath, sep="\t", skiprows=header_idx, header=0, index_col=0)


def collect_motion_files(
    root_dir: Path,
) -> dict[tuple[str, str], list[Path]]:
    """
    Key = (participant, motion)
    """
    motions: defaultdict[tuple[str, str], list[Path]] = defaultdict(list)

    for participant_dir in root_dir.iterdir():
        participant = participant_dir.name
        directory = root_dir / participant / "imu"
        if not directory.is_dir():
            continue

        for fname in Path.iterdir(directory):
            if not fname.name.endswith(".txt"):
                continue

            motion = fname.name.rsplit("-", 1)[0]
            path = directory / fname.name
            motions[participant, motion].append(path)

    return motions


MAX_PACKET = 65535


def validate_motion_df(df: pd.DataFrame) -> list[str]:
    errors = []

    # --- 1. Check PacketCounter continuity ---

    packet = df.index.to_series()

    # ensure numeric
    packet = pd.to_numeric(packet, errors="coerce")

    if packet.isna().any():
        errors.append("PacketCounter contains NaNs or non-numeric values")

    # check monotonic + no gaps (with overflow handling)
    diff = packet.diff().dropna()

    # valid transitions:
    #  1           -> normal step
    # -65535       -> 65535 => 0 (overflow forward)
    valid_step = (diff == 1) | (diff == -MAX_PACKET)

    bad_steps = diff[~valid_step]

    if not bad_steps.empty:
        missing_frames = [(packet.iloc[i - 1], packet.iloc[i]) for i in bad_steps.index]

        errors.append(f"Missing frames detected: {missing_frames}")

    overflow_events = diff[(diff == -MAX_PACKET)]

    if not overflow_events.empty:
        # overflow_list = [
        #     (packet.iloc[i - 1], packet.iloc[i]) for i in overflow_events.index
        # ]
        # errors.append(f"PacketCounter overflow detected: {overflow_list}")
        errors.append("PacketCounter Overflow")

    # --- 2. Check all numeric columns ---
    numeric_df = df.drop(columns=["PacketCounter"], errors="ignore")

    # force numeric conversion
    coerced = numeric_df.apply(pd.to_numeric, errors="coerce")

    if coerced.isna().any().any():
        nan_locs = np.where(coerced.isna())
        errors.append(
            f"Non-numeric/NaN values found at rows={nan_locs[0][:10]}, cols={nan_locs[1][:10]}",
        )

    return errors


def _process_single_file(participant: Path, motion: str, f: Path) -> dict[str, Any]:
    path = f
    df = read_file(path)

    errors = validate_motion_df(df)

    if errors:
        logger.info("[WARNING] %s", path)
        for e in errors:
            logger.info("  - %s", e)

    return {
        "participant": participant,
        "trial": motion,
        "file": f,
        "df": df,
        "notes": errors,
    }


def process_motion_files(
    motions: dict[tuple[str, str], list[Path]],
) -> pd.DataFrame:
    tasks = [(participant, motion, f) for (participant, motion), files in motions.items() for f in files]
    with Pool() as executor:
        results = executor.starmap(_process_single_file, tasks)

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
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    motions = collect_motion_files(args.source_dir)
    # logger.info(motions)
    summary_df = process_motion_files(motions)
    summary_df = summary_df.drop("file", axis=1)
    summary_df = summary_df.drop("df", axis=1)
    summary_df = summary_df.map(
        lambda x: "" if isinstance(x, list) and len(x) == 0 else x,
    )
    summary_df = (
        summary_df.sort_values(["participant", "trial"]).groupby(["participant", "trial"], as_index=False).first()
    )
    logger.info(summary_df)
    output_file = output_dir / "imu-continuity.csv"
    summary_df.to_csv(output_file, index=False)

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
