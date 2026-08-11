import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Process data.")
parser.add_argument(
    "--input_csv",
    default="participant-record.csv",
    help="Input CSV filename (default: participant-record.csv)",
)
parser.add_argument(
    "--input_dir",
    default="out",
    help="Directory to save output CSV (default: current directory)",
)
parser.add_argument(
    "--output_dir",
    default="out",
    help="Directory to save output CSV (default: current directory)",
)
args = parser.parse_args()

input_csv = args.input_csv
output_file = "latex-trial-info.txt"
output_dir = args.output_dir
input_file = os.path.join(args.input_dir, input_csv)
output_path = os.path.join(output_dir, output_file)


df = pd.read_csv(input_file)

# --- Table 1: Basic info ---
rename_map = {
    "label": "Trial",
    "description": "Description",
    "reps": "Reps",
    "mocap": "Mocap",
    "imu": "IMU",
    "emg": "EMG",

}
num_participants = len(df)

df.drop("description",axis=1, inplace=True)
df["label"] = df["label"].map(lambda x: f"\\progfunc{{{x}}}")

df = df.rename(columns=rename_map)


latex_output = df.to_latex(
    index=False,
    caption="22 Motion trials and available modalities for each contained in the dataset",
    label="tab:motion_trials",
    escape=False,
    float_format="%.1f",
)
print(latex_output)
with open(output_path, "w", newline="") as csvfile:
    csvfile.write(latex_output)

