import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Process data.")
parser.add_argument(
    "--input_csv",
    default="participant-record.csv",
    help="Input CSV filename (default: participant-record.csv)",
)
parser.add_argument(
    "--input_dir",
    default="out",
    type=Path,
    help="Directory to save output CSV (default: current directory)",
)
parser.add_argument(
    "--output_dir",
    default="out",
    type=Path,
    help="Directory to save output CSV (default: current directory)",
)
args = parser.parse_args()

input_csv = args.input_csv
output_demographic_file = "latex-demographics.txt"
output_dimensions_file = "latex-dimensions.txt"
output_dir = args.output_dir
input_file = args.input_dir / input_csv
output_demographic = output_dir / output_demographic_file
output_dimensions = output_dir / output_dimensions_file


df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["id"], errors="ignore")


def summary_table(df_subset: pd.DataFrame, index_list: list[str]) -> pd.DataFrame:
    mean_row = df_subset.mean().round(2)
    sd_row = df_subset.std().round(2)
    range_row = df_subset.apply(lambda x: f"{x.min():.1f}–{x.max():.1f}")  # ruff: ignore[ambiguous-unicode-character-string]
    return pd.DataFrame([mean_row, sd_row, range_row], index=index_list)


# --- Table 1: Basic anthropometrics ---
cols_basic = ["body_mass", "height", "age"]
rename_map = {
    "body_mass": "Mass [kg]",
    "height": "Height [cm]",
    "age": "Age [years]",
}
num_participants = len(numeric_df)
index_list = [r"$\mu$", r"$\sigma$", r"$\Delta$"]

summary_basic = summary_table(
    numeric_df[cols_basic].rename(columns=rename_map),
    index_list,
)

latex_demographics = summary_basic.to_latex(
    index=True,
    caption=r"Demographic summary (mean $\mu$, standard deviation $\sigma$, and range $\Delta$) of the participants "
    f"(N = {num_participants}) in the dataset",
    label="tab:anthro_basic",
    escape=False,
    float_format="%.1f",
)
logger.info(latex_demographics)
output_demographic.write_text(latex_demographics, newline="")


cols_custom = [
    "body_mass",
    "height",
    "age",
    "femur_length_left",
    "femur_length_right",
    "tibia_length_left",
    "tibia_length_right",
    "knee_width_left",
    "knee_width_right",
    "ankle_width_left",
    "ankle_width_right",
    "foot_length_left",
    # "foot_length_right",
    "foot_width_left",
    # "foot_width_right",
    "hip_width",
    "torso_length",
    "torso_width",
    "humerus_length_left",
    "humerus_length_right",
    "forearm_length_left",
    "forearm_length_right",
    "elbow_width_left",
    "elbow_width_right",
    "wrist_width_left",
    "wrist_width_right",
    "hand_thickness_left",
    "hand_thickness_right",
    # "walking_speed",
    # "jogging_speed",
    # "crab_walking_speed"
]

rename_map = {
    "body_mass": "Mass [kg]",
    "height": "Height [cm]",
    "age": "Age [years]",
    "femur_length_left": "Femur Length L [cm]",
    "femur_length_right": "Femur Length R [cm]",
    "tibia_length_left": "Tibia Length L [cm]",
    "tibia_length_right": "Tibia Length R [cm]",
    "knee_width_left": "Knee Width L [cm]",
    "knee_width_right": "Knee Width R [cm]",
    "ankle_width_left": "Ankle Width L [cm]",
    "ankle_width_right": "Ankle Width R [cm]",
    "foot_length_left": "Foot Length L/R [cm]",
    "foot_length_right": "Foot Length R [cm]",
    "foot_width_left": "Foot Width L/R [cm]",
    "foot_width_right": "Foot Width R [cm]",
    "hip_width": "Hip Width [cm]",
    "torso_length": "Torso Length [cm]",
    "torso_width": "Torso Width [cm]",
    "humerus_length_left": "Humerus Length L [cm]",
    "humerus_length_right": "Humerus Length R [cm]",
    "forearm_length_left": "Forearm Length L [cm]",
    "forearm_length_right": "Forearm Length R [cm]",
    "elbow_width_left": "Elbow Width L [cm]",
    "elbow_width_right": "Elbow Width R [cm]",
    "wrist_width_left": "Wrist Width L [cm]",
    "wrist_width_right": "Wrist Width R [cm]",
    "hand_thickness_left": "Hand Thickness L [cm]",
    "hand_thickness_right": "Hand Thickness R [cm]",
}

subset_df = numeric_df[cols_custom].rename(columns=rename_map)
summary_custom = summary_table(subset_df, index_list)

summary_transposed = summary_custom.T
summary_transposed = summary_transposed.reset_index()
summary_transposed = summary_transposed.rename(columns={"index": "Measurement"})

latex_dimensions = summary_transposed.to_latex(
    index=False,
    caption=(
        "Demographic and anthropometric measurements summary "
        r"(mean $\mu$, standard deviation $\sigma$, and range $\Delta$) statistics. "
        "The abbreviations L and R indicate left and right, respectively."
    ),
    label="tab:demographic_anthropometric_stats",
    escape=False,
    float_format="%.1f",
)

logger.info(latex_dimensions)

output_dimensions.write_text(latex_dimensions, newline="")
