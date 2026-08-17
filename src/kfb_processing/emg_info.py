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
output_file = "latex-emg-info.txt"
output_dir = args.output_dir
input_file = args.input_dir / input_csv
output_path = output_dir / output_file


df = pd.read_csv(input_file)

# --- Table 1: Basic info ---
rename_map = {
    "label": "Sensor",
    "side": "Side",
    "muscle": "Muscle",
    "sensor": "Sensor",
    "trial": "Trial",
    "type": "Type",
}
num_participants = len(df)

df = df.drop(["#", "id", "description"], axis=1)
df["trial"] = df["trial"].map(lambda x: f"\\progfunc{{{x}}}")
df = df.rename(columns=rename_map)


latex_output = df.to_latex(
    index=False,
    caption="EMG modalities",
    label="tab:emg_info",
    escape=False,
    float_format="%d",
)
logger.info(latex_output)
output_path.write_text(latex_output, newline="")
