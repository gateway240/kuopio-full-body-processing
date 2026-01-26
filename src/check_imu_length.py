from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple


def read_data_lines(filepath: str) -> Tuple[List[str], List[str]]:
    header_lines = []
    data_lines = []
    # print(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # print(line)
            stripped = line.strip()
            if not stripped:
                continue
            elif stripped.startswith("//"):
                header_lines.append(line)
            elif "PacketCounter" in stripped:
                header_lines.append(line)
            else:
                data_lines.append(line)

    return header_lines, data_lines


def get_last_packet_counter(data_lines: List[str]) -> int:
    last_line: str = data_lines[-1]
    return int(last_line.split("\t")[0])


def collect_motion_files(
    root_dir: str,
) -> DefaultDict[Tuple[str, str], List[str]]:
    """
    Key = (participant, motion)
    """
    motions: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    for participant in os.listdir(root_dir):
        imu_dir: str = os.path.join(root_dir, participant, "imu")
        if not os.path.isdir(imu_dir):
            continue

        for fname in os.listdir(imu_dir):
            if not fname.endswith(".txt"):
                continue

            motion = fname.rsplit("-", 1)[0]
            path = os.path.join(imu_dir, fname)
            motions[(participant, motion)].append(path)

    return motions


def trim_data_by_packet(
    header_lines: List[str],
    data_lines: List[str],
    max_packet: int,
) -> List[str]:
    """
    Keeps all header lines and only data lines with PacketCounter <= max_packet
    """
    trimmed = [line for line in data_lines if int(line.split("\t", 1)[0]) <= max_packet]
    return header_lines + trimmed


def process_motion_files(
    motions: Dict[Tuple[str, str], List[str]], dry_run: bool = True
) -> int:
    processed_files = 0
    for (participant, motion), files in motions.items():
        file_data: Dict[str, Tuple[List[str], List[str]]] = {}

        for f in files:
            header, data = read_data_lines(f)
            file_data[f] = (header, data)
            # print(f)
            # print(header)
            # print()

        # Get last packet counter per file
        last_packets: Dict[str, int] = {
            f: get_last_packet_counter(data) for f, (_, data) in file_data.items()
        }

        # Skip already aligned
        if len(set(last_packets.values())) == 1:
            continue

        print(f"\nParticipant {participant}, motion '{motion}' has mismatched lengths:")
        # for f, l in last_packets.items():
        #     print(f"  {os.path.basename(f)}: {l} lines")

        shortest_file: str = min(last_packets, key=lambda f: last_packets[f])
        shortest_data: List[str] = file_data[shortest_file][1]
        last_packet: int = get_last_packet_counter(shortest_data)

        print(f"  → Shortest file: {os.path.basename(shortest_file)}")
        print(f"  → Last PacketCounter: {last_packet}")

        for f, (header, data) in file_data.items():
            trimmed_lines = trim_data_by_packet(header, data, last_packet)

            if not dry_run:
                with open(f, "w") as out:
                    out.writelines(trimmed_lines)

            print(
                f"  {os.path.basename(f)}: "
                f"{len(data) + len(header)} → {len(trimmed_lines)} lines"
            )
        processed_files += 1
    return processed_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IMU file lengths and fix")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Root directory containing subject folders",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="If set, do not write any output files."
    )

    args = parser.parse_args()

    motions = collect_motion_files(args.source_dir)
    processed_files = process_motion_files(motions, args.dry_run)
    print(f"\nDone. Processed: {processed_files} files!")


if __name__ == "__main__":
    main()
