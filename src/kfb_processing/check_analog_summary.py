from __future__ import annotations

import argparse
import logging
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler

if TYPE_CHECKING:
    from pandas.io.formats.style_render import ExtFormatter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PLOT_RESULTS = False

SNR_THRESHOLD = 3.0

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

    input_file = output_dir / "emg-snr.csv"
    summary_df = pd.read_csv(input_file)
    exclude_cols = ["participant", "trial", "missing"]
    snr_cols = [c for c in summary_df.columns if c not in exclude_cols]

    # Define a function to calculate stats per participant
    def calculate_participant_stats(group: pd.DataFrame) -> pd.Series:
        # Flatten all SNR columns into a 1D array and remove NaNs
        values = group[snr_cols].to_numpy().flatten()
        values = values[~pd.isna(values)]

        # Compute mean, std, and range
        snr_mean = values.mean() if len(values) > 0 else pd.NA
        snr_std = values.std(ddof=1) if len(values) > 1 else pd.NA
        snr_range = f"{values.min():.1f} – {values.max():.1f}"  # ruff: ignore[ambiguous-unicode-character-string]

        return pd.Series(
            {"snr_mean": snr_mean, "snr_std": snr_std, "snr_range": snr_range},
        )

    # Apply the function per participant
    summary_stats = summary_df.groupby("participant").apply(calculate_participant_stats).reset_index()

    logger.info(summary_stats)

    output_dir_latex = pathlib.Path("out")
    output_file = output_dir_latex / "emg-snr-per-participant.csv"
    col = summary_stats.columns[0]
    summary_stats.assign(**{col: summary_stats[col].map(lambda x: f"{x:02d}")}).to_csv(
        output_file,
        index=False,
    )

    rename_map = {
        "participant": r"\#",
        "snr_mean": r"SNR $\mu$",
        "snr_std": r"SNR $\sigma$",
        "snr_range": r"SNR $\Delta$",
    }

    summary_stats = summary_stats.rename(columns=rename_map)

    fmt: ExtFormatter = {summary_stats.columns[0]: "{:02d}"}  # first column as integer
    fmt.update(
        {col: "{:.2f}" for col in summary_stats.columns[1:] if pd.api.types.is_numeric_dtype(summary_stats[col])},
    )
    styler = Styler(summary_stats)
    styler.format(fmt)
    styler.hide(axis="index")
    latex = styler.to_latex(
        caption=(
            "EMG signal-to-noise  ratio (SNR) summaries across all trials "
            r"(mean $\mu$, standard deviation $\sigma$, and range $\Delta$) for each participant (\#). "
            "All values are presented in decibels (dB). "
            "For more granular, per-trial metrics, see the provided ``emg-snr.csv'' file. "
        ),
        label="tab:emg_snr_per_participant",
        position_float="centering",
        hrules=True,  # adds \toprule, \midrule, \bottomrule
    )

    logger.info("\n%s", latex)
    output_file_latex = output_dir_latex / "emg-snr-per-participant.txt"
    output_file_latex.write_text(latex, encoding="utf-8", newline="")

    logger.info("Done. Processed: %d trials!", len(summary_df))


if __name__ == "__main__":
    main()
