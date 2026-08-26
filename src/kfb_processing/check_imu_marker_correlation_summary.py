from __future__ import annotations

import argparse
import logging
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from pandas.io.formats.style_render import ExtFormatter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAN_THRESHOLD_PERCENT = 50.0

KNOWN_TRIALS = {
    # "static_cal",
    # "dyn_sara",
    "dyn_score_hip",
    "dyn_score_ankle",
    # "crouch_lift",
    # "crouch_rotate",
    # "curls",
    "kettlebell",
    "squats_deep",
    "half_jacks",
    "squat_jumps",
    # "box_jabs",
    # "box_combos",
    # "chair_push_right",
    # "chair_push_left",
    # "arm_hang",
    # "heavy_lift",
    # "back_fly",
    # "side_fly",
    "walking",
    "jogging",
    # "crab_walking",
}


def main() -> None:  # ruff: ignore[too-many-locals]
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

    input_file = output_dir / "imu-marker-sync.csv"
    summary_df = pd.read_csv(input_file)
    top_corr = (
        summary_df.loc[
            (summary_df["marker_len"] > 0.0)
            & ~(summary_df["marker_nan_percent"] > NAN_THRESHOLD_PERCENT)
            & ~(summary_df["marker_constant_percent"] > NAN_THRESHOLD_PERCENT)
            & summary_df["trial"].isin(KNOWN_TRIALS)
        ].sort_values("best_corr", ascending=False)
        # .head(PAIRS_TO_SELECT)
    )

    # Aggregate statistics per participant across all trials
    summary_stats = (
        top_corr
        .groupby("participant")
        .agg(
            corr_mean=("best_corr", "mean"),
            corr_std=("best_corr", "std"),
            corr_range=("best_corr", lambda x: f"{x.min():.2f} – {x.max():.2f}"),  # ruff: ignore[ambiguous-unicode-character-string]
            lag_mean=("best_lag", "mean"),
            lag_std=("best_lag", "std"),
            lag_range=("best_lag", lambda x: f"{x.min():.0f} – {x.max():.0f}"),  # ruff: ignore[ambiguous-unicode-character-string]
        )
        .reset_index()
    )

    output_dir_latex = pathlib.Path("out")
    output_file = output_dir_latex / "imu-marker-correlation-per-participant.csv"
    col = summary_stats.columns[0]
    summary_stats.assign(**{col: summary_stats[col].map(lambda x: f"{x:02d}")}).to_csv(
        output_file,
        index=False,
    )

    rename_map = {
        "participant": r"\#",
        "corr_mean": r"Corr $\mu$",
        "corr_std": r"Corr $\sigma$",
        "corr_range": r"Corr $\Delta$",
        "lag_mean": r"Lag $\mu$",
        "lag_std": r"Lag $\sigma$",
        "lag_range": r"Lag $\Delta$",
    }

    summary_stats = summary_stats.rename(columns=rename_map)

    fmt: ExtFormatter = {summary_stats.columns[0]: "{:02d}"}  # first column as integer
    fmt.update(
        {col: "{:.2f}" for col in summary_stats.columns[1:] if pd.api.types.is_numeric_dtype(summary_stats[col])},
    )

    trials = list(KNOWN_TRIALS)
    progfunc = r"\progfunc"
    latex = (
        summary_stats.style
        .format(fmt)
        .hide(axis="index")
        .to_latex(
            caption=(
                "Optical Marker IMU sensor temporal alignment "
                r" (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) for each participant (\#). "
                "The correlation values are unitless and bounded from 0 to 1. "
                "The lag values are represented as frames from perfect alignment. "
                r"All marker-sensor pairs are included unless the marker was absent or occluded for over 50\% of the trial. "  # ruff: ignore[line-too-long]
                "These summary values are aggregated from the trials "
                f"{', '.join(f'{progfunc}{{{x}}}' for x in trials[:-1])}, and {progfunc}{{{trials[-1]}}}. "
                "The results for each individual trial are contained in the `imu-marker-sync.csv'' file for further analysis. "  # ruff: ignore[line-too-long]
            ),
            label="tab:imu_marker_correlation_per_participant",
            position_float="centering",
            hrules=True,  # adds \toprule, \midrule, \bottomrule
        )
    )

    logger.info("\n%s", latex)
    output_file_latex = output_dir_latex / "imu-marker-correlation-per-participant.txt"
    output_file_latex.write_text(latex, encoding="utf-8", newline="")

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
