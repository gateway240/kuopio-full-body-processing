from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOT_RESULTS = False

SNR_THRESHOLD = 3.0

PAIRS_TO_SELECT = 10

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


def main() -> None:
    # This makes a non-interactive backend to prevent memory leak
    # see https://github.com/matplotlib/matplotlib/issues/20300
    plt.switch_backend("agg")
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
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    input_file = output_dir / "imu-marker-sync.csv"
    summary_df = pd.read_csv(input_file)
    top_corr = (
        summary_df
        .sort_values("best_corr", ascending=False)
        .groupby(["participant", "trial"], group_keys=False)
        .head(PAIRS_TO_SELECT)
    )

    # Aggregate statistics per participant across all trials
    summary_stats = (
        top_corr
        .groupby("participant")
        .agg(
            corr_mean=("best_corr", "mean"),
            corr_std=("best_corr", "std"),
            corr_range=("best_corr", lambda x: f"{x.min():.2f} – {x.max():.2f}"),
            lag_mean=("best_lag", "mean"),
            lag_std=("best_lag", "std"),
            lag_range=("best_lag", lambda x: f"{x.min():.0f} – {x.max():.0f}"),
        )
        .reset_index()
    )

    output_dir_latex = pathlib.Path("out")
    output_file = output_dir_latex / "imu-marker-correlation-per-participant.csv"
    col = summary_stats.columns[0]
    summary_stats.assign(
        **{col: summary_stats[col].map(lambda x: f"{x:02d}")}
    ).to_csv(output_file, index=False)

    rename_map = {
        "participant": "\#",
        "corr_mean": r"Corr $\mu$",
        "corr_std": r"Corr $\sigma$",
        "corr_range": r"Corr $\Delta$",
        "lag_mean": r"Lag $\mu$",
        "lag_std": r"Lag $\sigma$",
        "lag_range": r"Lag $\Delta$",
    }

    summary_stats = summary_stats.rename(columns=rename_map)

    fmt = {summary_stats.columns[0]: "{:02d}"}  # first column as integer
    fmt.update({
        col: "{:.2f}"
        for col in summary_stats.columns[1:]
        if pd.api.types.is_numeric_dtype(summary_stats[col])
    })

    latex = (
        summary_stats.style.format(fmt)
        .hide(axis="index")
        .to_latex(
            caption=(
                "Optical Marker IMU sensor temporal alignment (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) for each participant (\#). "
                "The correlation values are unitless and bounded from 0 to 1. "
                "The lag values are represented as frames from perfect alignment. "
                "The 8 marker sensor pairs with the highest correlation are selected for each trial to create the summary statistics. "
                "Further details for each individual trail can be found in ``imu-marker-correlation.csv'' "
            ),
            label="tab:imu_marker_correlation_per_participant",
            position_float="centering",
            hrules=True,  # adds \toprule, \midrule, \bottomrule
        )
    )

    print(latex)
    output_file_latex = output_dir_latex / "imu-marker-correlation-per-participant.txt"
    with open(output_file_latex, "w", newline="") as file:
        file.write(latex)

    print(f"\nDone. Processed: {len(summary_df)} trials!")


if __name__ == "__main__":
    main()
