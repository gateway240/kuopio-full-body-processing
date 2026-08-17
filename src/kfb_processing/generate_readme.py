import argparse
import csv
import logging
from pathlib import Path
from typing import Self

import pandas as pd
from tabulate import tabulate

config_dir = "measurement-config"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReadmeBuilder:
    """Composable README generator with multiple output formats."""

    def __init__(self, fmt: str = "text") -> None:
        """
        fmt: "text" or "html"
        """
        self.format = fmt.lower()
        if self.format not in {"text", "html"}:
            self.format = "text"
            logger.info("Format must be 'text' or 'html'")
        self.sections: list[str] = []

    # --------------------------
    # Rendering Helpers
    # --------------------------
    def _render_heading(self, text: str, level: int) -> str:
        if self.format == "html":
            return f"<h{level}>{text}</h{level}>"

        # plain text
        if level == 1:
            underline = "=" * len(text)
        elif level == 2:  # ruff: ignore[magic-value-comparison]
            underline = "-" * len(text)
        else:
            underline = ""
        return f"{text}\n{underline}\n"

    def _render_paragraph(self, text: str) -> str:
        return f"<p>{text.strip()}</p>" if self.format == "html" else text.strip() + "\n"

    def _render_list(
        self,
        items: list[str],
    ) -> str:
        """
        Render a list.

        HTML: simple <ul> or <ol> without prefixes
        Plain text:
            - default: "-" for unordered, "1." for ordered
            - optional prefix: str or callable
        """
        # HTML rendering
        if self.format == "html":
            tag = "ul"
            li = "\n".join(f"<li>{item}</li>" for item in items)
            return f"<{tag}>\n{li}\n</{tag}>"

        # Plain text rendering
        lines = [f"- {item}" for i, item in enumerate(items)]
        return "\n".join(lines) + "\n"

    def _render_code(self, code: str) -> str:
        if self.format == "html":
            return f"<pre><code>{code.strip()}</code></pre>"
        lines = ["---- CODE ----", code.strip(), "--------------"]
        return "\n".join(lines) + "\n"

    def _render_table(
        self,
        rows: list[list[str]],
        headers: list[str],
        tablefmt: str,
    ) -> str:
        if self.format == "html":
            # tabulate supports HTML format directly
            return tabulate(rows, headers=headers, tablefmt="html")
        return tabulate(rows, headers=headers, tablefmt=tablefmt) + "\n"

    # --------------------------
    # Public API
    # --------------------------

    def add_heading(self, text: str, level: int = 1) -> Self:
        self.sections.append(self._render_heading(text, level))
        return self

    def add_paragraph(self, text: str) -> Self:
        self.sections.append(self._render_paragraph(text))
        return self

    def add_list(self, items: list[str]) -> Self:
        self.sections.append(self._render_list(items))
        return self

    def add_csv_table(self, csv_path: Path, tablefmt: str) -> Self:
        if not csv_path.exists():
            msg = f"CSV file not found: {csv_path}"
            raise FileNotFoundError(msg)

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)

        if not data:
            msg_0 = f"CSV file {csv_path} is empty."
            raise ValueError(msg_0)

        headers = data[0]
        rows = data[1:]
        self.sections.append(self._render_table(rows, headers, tablefmt))
        return self

    def add_code_block(self, code: str) -> Self:
        self.sections.append(self._render_code(code))
        return self

    def build(self) -> str:
        sep = "\n\n" if self.format == "html" else "\n"
        return sep.join(self.sections).strip() + "\n"

    def write(self, filepath: Path) -> None:
        filepath.write_text(self.build(), encoding="utf-8")
        logger.info("README generated at %s", filepath)


intro = """
The Kuopio Full-Body Dataset includes 13 willing participants who visited
the HUMEA laboratory at the University of Eastern Finland, Kuopio, Finland.
The measurements were conducted between March and September 2025.
The dataset contains 18 full-body, multi-modality motions measured with:
"""
intro_list = [
    "optical motion capture (Vicon Nexus)",
    "inertial measurement units (Xsens IMU sensors)",
    "electromyography sensors (Delsys EMG sensors)",
]

anthropometric_config = """
All length measurements are listed in centimeters (cm).
Participant and object masses are listed in kilograms (kg).
The distance between anatomical landmarks was measured
with either a small bone caliper (<30cm) or a soft measurement tape (>30cm).
"""

optical_config = """
Vicon Nexus (version 2.16) software to capture data from:
"""
optical_list = [
    "10 Vicon Vero cameras (Vicon Motion Systems Ltd, UK) at 1000 Hz",
    """2 OR6-7MA all-aluminum floor-embedded AMTI force platforms at 2400 Hz;
    Dimensions: 464 mm x 508 mm;
    MA= "mini amp" AMTI's signal amplifier which is inside the force plates
    (Advanced Mechanical Technology, Inc., Watertown, Massachusetts, USA)""",
    """1 BMS464508HF-2K floor-embedded AMTI force platforms at 2400 Hz;
    Dimensions: 464 mm x 508 mm;
    HF= High Frequency with composite top
    (Advanced Mechanical Technology, Inc., Watertown, Massachusetts, USA)
    """,
    """2 embedded force plates in a Motek M-gait Research split-belt treadmill
    (Motek Medical B.V., Amsterdam, Netherlands) at 2400 Hz""",
    "22 Delsys Wireless EMG sensors (describe below) at 2400 Hz",
]

tote_config = """
The workplace ergonomics tasks involve lifting a plastic storage tote.
The tote has:
"""
tote_list = [
    "a mass of 12 kg",
    "a height of 28 cm, length of 55 cm and a width of 39 cm",
    "3-4 optical markers attached to the sides",
]

bag_config = """
The boxing tasks involved adding a punching bag to the capture volume.
The punching bag has:
"""
bag_list = [
    "a mass of 40 kg",
    "a circumference of 100 cm and a height of 150 cm",
    "7 optical markers attached",
]


emg_config = """
A total of 22 channels of EMG data were collected from 14 physical sensor units.
EMGs are placed according to the SENIAM (http://seniam.org) guidelines.
The Delsys (Delsys, Natick, Massachusetts, USA) Trigno wireless EMG system has:
"""
emg_list = [
    "12 Avanti dry sensors",
    "2 Snap Lead wet sensors",
    "2 Quattro (4 channel) sensors",
]

imu_config = """
The Xsens MT Manager 2022.2 (Movella Inc, Henderson, NV, USA) wireless IMU system has:
"""
imu_list = [
    "13 MTw Awinda wireless IMU sensors at 60 Hz on the participant",
    """*Note: 1 sensor is placed on the bottom of the punching bag during the boxing trials.
         It should be disabled in analysis for trials other than boxing.""",
]

ethics = """
The University of Eastern Finland Committee on Research Ethics (statement no. 2/2025)
reviewed and approved this study and its data management practices.
All participants were volunteers who gave their informed consent to participate.
The measurements were performed in accordance with the Declaration of Helsinki.
"""

acknowledgments = """
This work was supported by:
"""
acknowledgments_list = [
    """the Finnish Ministry of Education and Culture's Pilot for Doctoral Programs
    (Pilot project Mathematics of Sensing, Imaging, and Modeling)""",
    "the Research Council of Finland under funding decision number 349469",
    "Biocenter Kuopio is acknowledged for financial and infrastructural support",
]

contact = """
Alexander Beattie, alexander.beattie@uef.fi
"""


def generate_valid_markers(markers: list[Path]) -> set[str]:
    valid_markers: set[str] = set()

    for path in markers:
        if not path.exists():
            msg = f"CSV file not found: {path}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(path)

        if df.empty:
            msg_0 = f"CSV file {path} is empty."
            raise ValueError(msg_0)

        if "id" not in df.columns:
            msg_1 = f"CSV file {path} missing required 'id' column"
            raise ValueError(msg_1)

        ids = df["id"].dropna().astype(str).str.strip()

        valid_markers.update(ids)

    return valid_markers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument(
        "--output_dir",
        default="out",
        type=Path,
        help="Directory to save output CSV (default: current directory)",
    )
    parser.add_argument(
        "--doc_fmt",
        default="text",
        help="Document and table format (default: fancy_grid)",
    )
    parser.add_argument(
        "--table_fmt",
        default="pretty",
        help="Document and table format (default: fancy_grid)",
    )

    args = parser.parse_args()
    doc_fmt = args.doc_fmt
    config_dir = Path("measurement-config")
    optical_participant_file = config_dir / "optical-marker-participant.csv"
    optical_bag_file = config_dir / "optical-marker-bag.csv"
    optical_tote_file = config_dir / "optical-marker-tote.csv"
    emg_file = config_dir / "emg-sensor-mappings.csv"
    imu_file = config_dir / "imu-sensor-mappings.csv"

    movement_file = config_dir / "movements.csv"
    anthropometric_file = config_dir / "anthropometric-descriptions.csv"
    calibration_file = config_dir / "calibration-descriptions.csv"

    tablefmt = args.table_fmt

    readme = (
        ReadmeBuilder(fmt=doc_fmt)
        .add_heading("Kuopio Full-Body Dataset", level=1)
        .add_paragraph(intro)
        .add_list(intro_list)
        .add_heading("Movement List", level=2)
        .add_csv_table(movement_file, tablefmt=tablefmt)
        .add_heading("Anthropometric Measurements", level=2)
        .add_paragraph(anthropometric_config)
        .add_csv_table(anthropometric_file, tablefmt=tablefmt)
        .add_heading(
            "Manual Joint Angle Measurements (During Static Calibration)",
            level=2,
        )
        .add_csv_table(calibration_file, tablefmt=tablefmt)
        .add_heading("Optical Capture Setup - Participant", level=2)
        .add_paragraph(optical_config)
        .add_list(optical_list)
        .add_csv_table(optical_participant_file, tablefmt=tablefmt)
        .add_heading("Optical Capture Setup - Storage Tote", level=2)
        .add_paragraph(tote_config)
        .add_list(tote_list)
        .add_csv_table(optical_tote_file, tablefmt=tablefmt)
        .add_heading("Optical Capture Setup - Punching Bag", level=2)
        .add_paragraph(bag_config)
        .add_list(bag_list)
        .add_csv_table(optical_bag_file, tablefmt=tablefmt)
        .add_heading("EMG Sensor Setup. (Q = Quattro Sensor)", level=2)
        .add_paragraph(emg_config)
        .add_list(emg_list)
        .add_csv_table(emg_file, tablefmt=tablefmt)
        .add_heading("IMU Sensor Setup", level=2)
        .add_paragraph(imu_config)
        .add_list(imu_list)
        .add_csv_table(imu_file, tablefmt=tablefmt)
        .add_heading("Ethics Statement", level=2)
        .add_paragraph(ethics)
        .add_heading("Acknowledgments", level=2)
        .add_paragraph(acknowledgments)
        .add_list(acknowledgments_list)
        .add_heading("Contact", level=2)
        .add_paragraph(contact)
        # .add_heading("Usage", level=2)
        # .add_paragraph(
        #     "To update the configuration, edit the CSV file and re-run this script:"
        # )
        # .add_code_block("python generate_readme.py")
        # .add_heading("Notes", level=2)
        # .add_list(
        #     [
        #         "Ensure the CSV headers match expected fields.",
        #         "Use UTF-8 encoding when editing CSV files.",
        #     ]
        # )
    )

    logger.info(readme.build())
    logger.info(
        "Valid Participant Markers: %s",
        generate_valid_markers([optical_participant_file]),
    )
    logger.info("Valid Bag Markers: %s", generate_valid_markers([optical_bag_file]))
    logger.info("Valid Tote Markers: %s", generate_valid_markers([optical_tote_file]))

    out_path = args.output_dir / "readme.txt"
    readme.write(out_path)
