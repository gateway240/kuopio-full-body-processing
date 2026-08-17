import argparse
import logging
from pathlib import Path

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
output_file = "latex-trial-info.txt"
output_dir = args.output_dir
input_file = args.input_dir / input_csv
output_path = output_dir / output_file


df = pd.read_csv(input_file)

# --- Table 1: Basic info ---
rename_map = {
    "label": "Trial",
    "description": "Description",
    "group": "Group",
    "reps": "Reps",
    "mocap": "MC",
    "imu": "IMU",
    "emg": "EMG",
    "fp_l": "LF",
    "fp_r": "RF",
}
num_participants = len(df)

df = df.drop("description", axis=1)
df["label"] = df["label"].map(lambda x: f"\\progfunc{{{x}}}")
df["fp_l"] = df["fp_l"].map(lambda x: "-" if x == 0 else x)
df["fp_r"] = df["fp_r"].map(lambda x: "-" if x == 0 else x)

df = df.rename(columns=rename_map)


latex_output = df.to_latex(
    index=False,
    caption="22 Motion trials and available modalities for each contained in the dataset",
    label="tab:motion_trials",
    escape=False,
    float_format="%d",
)
logger.info(latex_output)
output_path.write_text(latex_output, newline="")
