import argparse
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# This script should generate a diagram with no assumptions about structure
# other than the following:
#   - Underneath a root folder there are subfolders
#     and inside them are numeric dirs (01...13)
#   - Within the numeric dirs there can be folders
#   - At the final level of the folders, all file extensions should be aggregated.
#     Additionally, files will contain a suffix like _orientations.sto which
#     should be included in the grouping such that the label is TRIAL_orientations.sto
#   - At the end, all common structure is extracted and collapsed

IGNORED = {
    ".DS_Store",
}

FOLDER_ICON = "📁"
FILE_ICON = "📄"

VALID_SUFFIXES = [
    "_markers.trc",
    "_analog.sto",
    "_grfs.sto",
    "_accelerations.sto",
    "_orientations.sto",
]

VALID_EXTENSIONS = {
    ".c3d",
    ".csv",
    ".mtb",
    ".sto",
    ".system",
    ".trc",
    ".txt",
    ".x1d",
    ".x2d",
    ".xcp",
}

MODALITIES = ["imu", "mocap"]


def safe_dirs(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir() and p.name not in IGNORED])


def is_allowed(f: Path) -> bool:
    return f.suffix.lower() in VALID_EXTENSIONS


def group_key(f: Path) -> str:
    name = f.name

    for suf in VALID_SUFFIXES:
        if name.endswith(suf):
            return "TRIAL" + suf

    return f"TRIAL{f.suffix}" if f.suffix else "TRIAL"


def emit(lines: list[str], a: str, b: str | None = None) -> None:
    if b is None:
        lines.append(a)
    else:
        lines.append(f"{a} -> {b}")


def emit_node(lines: list[str], node_id: str, label: str, font_size: float) -> None:  # ruff: ignore[unused-function-argument]
    emit(lines, f'{node_id}: "{label}"')
    # emit(lines, f"{node_id}.style.font-size: {font_size}")


def node_id(*parts: Any) -> str:  # ruff: ignore[any-type]
    def clean(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))

    return "__".join(clean(p) for p in parts)


def stable_id(*parts: list[str]) -> str:
    key = "/".join(str(p) for p in parts).encode()
    return hashlib.md5(key).hexdigest()[:10]


def emit_raw_files(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    lines: list[str],
    parent_id: str,
    files: list[Path],
    root_name: str,
    session_name: str | None,
    font_size: float,
) -> None:
    for f in sorted(files):
        fid = node_id(root_name, session_name or "root", "file", f.name)
        file_node = f"file_{fid}"
        emit_node(lines, file_node, f"{FILE_ICON} {f.name}", font_size)
        emit(lines, parent_id, file_node)


def emit_grouped_files(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    lines: list[str],
    parent_id: str,
    root_name: str,
    session_name: str,
    participants: list[Path],
    modality: str,
    font_size: float,
) -> None:
    groups = defaultdict(list)

    for p in participants:
        mod_path = p / modality
        if not mod_path.exists():
            continue

        for f in mod_path.iterdir():
            if f.is_file() and f.name not in IGNORED and is_allowed(f):
                groups[group_key(f)].append(f)

    for g in sorted(groups):
        gid = node_id(root_name, session_name, "participants", modality, g)
        file_node = f"file_{gid}"
        emit_node(lines, file_node, f"{FILE_ICON} {g}", font_size)
        emit(lines, parent_id, file_node)


def get_participant_range(session: Path) -> tuple[str, list[Path]]:
    participants = sorted(
        [p for p in session.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda x: int(x.name),
    )
    if not participants:
        msg = "No participants found!"
        raise ValueError(msg)

    names = [p.name for p in participants]
    label = f"[{names[0]}–{names[-1]}]" if len(participants) > 1 else f"[{names[0]}]"  # ruff: ignore[ambiguous-unicode-character-string]
    return label, participants


def build(root: Path, lines: list[str], font_size: float) -> None:
    root_id = node_id(root.name)
    emit_node(lines, root_id, f"{FOLDER_ICON} {root.name}", font_size)

    root_files = [f for f in root.iterdir() if f.is_file() and f.name not in IGNORED and is_allowed(f)]
    emit_raw_files(lines, root_id, root_files, root.name, None, font_size)

    for session in safe_dirs(root):
        sid = node_id(root.name, session.name)
        emit_node(lines, sid, f"{FOLDER_ICON} {session.name}", font_size)
        emit(lines, root_id, sid)

        session_files = [f for f in session.iterdir() if f.is_file() and f.name not in IGNORED and is_allowed(f)]
        emit_raw_files(lines, sid, session_files, root.name, session.name, font_size)

        label, participants = get_participant_range(session)
        if not participants:
            continue

        pid = node_id(root.name, session.name, label)
        emit_node(lines, pid, f"{FOLDER_ICON} {label}", font_size)
        emit(lines, sid, pid)

        for modality in MODALITIES:
            mid = node_id(root.name, session.name, label, modality)
            emit_node(lines, mid, f"{FOLDER_ICON} {modality}", font_size)
            emit(lines, pid, mid)

            emit_grouped_files(
                lines,
                mid,
                root.name,
                session.name,
                participants,
                modality,
                font_size,
            )


def generate(root_dir: Path, lines: list[str], out: Path, font_size: float) -> None:
    root = root_dir.resolve()

    build(root, lines, font_size)

    out.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Generated: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", default="dataset_graph.d2")
    parser.add_argument("--font-size", type=int, default=18)
    args = parser.parse_args()
    # See: https://d2lang.com/tour/globs/#changing-defaults
    lines = [
        "direction: right",
        "",
        "**: {",
        f"style.font-size: {args.font_size}",
        "}",
        "(*** -> ***)[*]: {",
        "style.stroke: black",
        "}",
    ]

    generate(Path(args.directory), lines=lines, out=Path(args.output), font_size=args.font_size)


if __name__ == "__main__":
    main()
