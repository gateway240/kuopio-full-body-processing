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
output_file = "latex-emg-info.txt"
output_dir = args.output_dir
input_file = args.input_dir / input_csv
output_path = output_dir / output_file


df = pd.read_csv(input_file)

# --- Table 1: Basic info ---
rename_map = {
    "#": r"\#",
    "side": "Side",
    "muscle": "Muscle",
    "sensor": "Sensor",
    "trial": "Reference Trial",
    "type": "Type",
}
num_participants = len(df)

df["trial"] = df["trial"].map(lambda x: f"\\progfunc{{{x}}}")
df["muscle"] = df.apply(lambda r: f"{r['description']} [{r['muscle']}]", axis=1)
df = df.drop(["label", "id", "description", "sensor"], axis=1)
df = df.rename(columns=rename_map)

styler = Styler(df)
styler.format()
styler.hide(axis="index")
latex_output = styler.to_latex(
    caption=(
        "The 22 measured EMG channels, their corresponding muscles, and suggested reference trials. "
        "Since muscles were measured symmetrically, "
        "the side column indicates the left (L) or right (R) side of the body. "
        "The type column indicates if the reference trial is dynamic (D) or isometric(I)."
    ),
    label="tab:emg_info",
    hrules=True,
    position_float="centering",
)
logger.info("\n%s", latex_output)
output_path.write_text(latex_output, newline="")
