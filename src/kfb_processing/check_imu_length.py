from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data_lines(filepath: Path) -> tuple[list[str], list[str]]:
    header_lines = []
    data_lines = []
    # logger.info(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            # logger.info(line)
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("//") or "PacketCounter" in stripped:
                header_lines.append(line)
            else:
                data_lines.append(line)

    return header_lines, data_lines


def get_last_packet_counter(data_lines: list[str]) -> int:
    last_line: str = data_lines[-1]
    return int(last_line.split("\t", maxsplit=1)[0])


def collect_motion_files(
    root_dir: Path,
) -> dict[tuple[Path, str], list[Path]]:
    """
    Key = (participant, motion)
    """
    motions: defaultdict[tuple[Path, str], list[Path]] = defaultdict()

    for participant in Path.iterdir(root_dir):
        imu_dir = root_dir / participant / "imu"
        if not imu_dir.is_dir():
            continue

        for fname in Path.iterdir(imu_dir):
            if not fname.name.endswith(".txt"):
                continue

            motion = fname.name.rsplit("-", 1)[0]
            path = imu_dir / fname
            motions[participant, motion].append(path)

    return motions


def trim_data_by_packet(
    header_lines: list[str],
    data_lines: list[str],
    max_packet: int,
) -> list[str]:
    """
    Keeps all header lines and only data lines with PacketCounter <= max_packet
    """
    trimmed = [line for line in data_lines if int(line.split("\t", 1)[0]) <= max_packet]
    return header_lines + trimmed


def process_motion_files(
    motions: dict[tuple[Path, str], list[Path]],
    dry_run: bool = True,  # ruff: ignore[boolean-default-value-positional-argument, boolean-type-hint-positional-argument]
) -> int:
    processed_files = 0
    for (participant, motion), files in motions.items():
        file_data: dict[Path, tuple[list[str], list[str]]] = {}

        for f in files:
            header, data = read_data_lines(f)
            file_data[f] = (header, data)
            # logger.info(f)
            # logger.info(header)
            # logger.info()

        # Get last packet counter per file
        last_packets: dict[Path, int] = {f: get_last_packet_counter(data) for f, (_, data) in file_data.items()}
        # Skip already aligned
        if len(set(last_packets.values())) == 1:
            continue

        logger.info("Participant %s, motion '%s' has mismatched lengths:", participant, motion)
        # for f, l in last_packets.items():
        #     logger.info(f"  {os.path.basename(f)}: {l} lines")

        shortest_file = min(last_packets, key=last_packets.__getitem__)
        shortest_data = file_data[shortest_file][1]
        last_packet: int = get_last_packet_counter(shortest_data)

        logger.info("  → Shortest file: %s ", shortest_file.name)
        logger.info("  → Last PacketCounter: %s", last_packet)

        for f, (header, data) in file_data.items():
            trimmed_lines = trim_data_by_packet(header, data, last_packet)

            if not dry_run:
                with f.open("w") as out:
                    out.writelines(trimmed_lines)

            logger.info(
                "  %s: %d → %d lines",
                f.name,
                len(data) + len(header),
                len(trimmed_lines),
            )
        processed_files += 1
    return processed_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IMU file lengths and fix")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write any output files.",
    )

    args = parser.parse_args()

    motions = collect_motion_files(args.source_dir)
    processed_files = process_motion_files(motions, args.dry_run)
    logger.info("Done. Processed: %d files!", processed_files)


if __name__ == "__main__":
    main()
