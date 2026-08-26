import argparse
import logging
from pathlib import Path

import pandas as pd
from pandas.io.formats.style import Styler

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
    "#": r"\#",
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

styler = Styler(df)
styler.format()
styler.hide(axis="index")
latex_output = styler.to_latex(
    caption=(
        "The 22 motion trials for each participant, which can be grouped into the categories: "
        "(i)~calibration [CAL], (ii)~ergonomics and fitness [FIT], (iii)~boxing [BOX], "
        "(iv)~reference activation [RA], and (v)~treadmill exercises [TR]. "
        "The optical motion capture [MC], IMU, and EMG columns, indicate (yes [y] or no [n]) whether the "
        "modality was present in the trial. "
        "The repetitions [Reps] columns indicates the number of repetitions of the motion "
        "or time duration of the trial in the case of the TR category. "
        "The left foot [LF] and right foot [RF] columns "
        "indicate which force platforms were active under each foot during the trial. "
        "A blank value [-] indicates that a force platform was not active during the specific trial."
    ),
    label="tab:motion_trials",
    hrules=True,
    position_float="centering",
)

logger.info("\n%s", latex_output)
output_path.write_text(latex_output, newline="")
