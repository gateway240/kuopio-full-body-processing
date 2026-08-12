import argparse
import logging
import os
from pathlib import Path

# Add this jankness for windows timestamps
TIMEZONE_OFFSET = 2 * 3600

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match files and overwrite timestamps.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing P### folders with files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory containing files to update timestamps for",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.exists():
        msg = f"Input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)

    if not output_dir.exists():
        msg_0 = f"Output directory does not exist: {output_dir}"
        raise FileNotFoundError(msg_0)

    # Walk through input directory structure
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            input_file = Path(root) / file
            input_name = input_file.name.lower()

            for out_file in output_dir.iterdir():
                if not out_file.is_file():
                    continue

                output_name = out_file.name.lower()

                # This is the rule you want:
                # ANYTHING_trial_01.something → trial_01.something
                if not input_name.endswith(output_name):
                    continue

                # Copy timestamps
                stat = input_file.stat()
                new_atime = stat.st_atime + TIMEZONE_OFFSET
                new_mtime = stat.st_mtime + TIMEZONE_OFFSET
                os.utime(out_file, (new_atime, new_mtime))

                logger.info("[OK] Matched: %s  →  %s", input_file.name, out_file.name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
