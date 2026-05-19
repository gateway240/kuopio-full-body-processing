#!/usr/bin/env python3
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

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

MODALITIES = ["imu", "mocap"]


def safe_dirs(path: Path):
    return sorted([p for p in path.iterdir() if p.is_dir() and p.name not in IGNORED])


def group_key(f: Path):
    # same logic you already had (simplified)
    name = f.name

    for suf in VALID_SUFFIXES:
        if name.endswith(suf):
            return "TRIAL" + suf

    return f"TRIAL{f.suffix}" if f.suffix else "TRIAL"


def emit(lines, a, b=None):
    lines.append(a if b is None else f"{a} -> {b}")


def node_id(*parts):
    def clean(s):
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))

    return "__".join(clean(p) for p in parts)


def stable_id(*parts):
    key = "/".join(str(p) for p in parts).encode()
    return hashlib.md5(key).hexdigest()[:10]


def emit_raw_files(lines, parent_id, files, tag, root_name, session_name):
    for f in sorted(files):
        fid = node_id(root_name, session_name or "root", tag, f.name)
        emit(lines, f'file_{fid}: "{FILE_ICON} {f.name}"')
        emit(lines, parent_id, f"file_{fid}")


def emit_grouped_files(
    lines, parent_id, root_name, session_name, participants, modality
):
    groups = defaultdict(list)

    for p in participants:
        mod_path = p / modality
        if not mod_path.exists():
            continue

        for f in mod_path.iterdir():
            if f.is_file() and f.name not in IGNORED:
                groups[group_key(f)].append(f)

    for g in sorted(groups):
        gid = node_id(root_name, session_name, "participants", modality, g)
        emit(lines, f'file_{gid}: "{FILE_ICON} {g}"')
        emit(lines, parent_id, f"file_{gid}")


def get_participant_range(session: Path):
    participants = sorted(
        [p for p in session.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda x: int(x.name),
    )
    if not participants:
        return None, []

    names = [p.name for p in participants]
    label = f"[{names[0]}–{names[-1]}]" if len(names) > 1 else f"[{names[0]}]"

    return label, participants


def build(root: Path, lines):
    root_id = node_id(root.name)
    emit(lines, f'{root_id}: "{FOLDER_ICON} {root.name}"')

    # --- ROOT FILES (NEW) ---
    root_files = [f for f in root.iterdir() if f.is_file() and f.name not in IGNORED]
    emit_raw_files(lines, root_id, root_files, "root_files", root.name, None)

    # --- SESSIONS ---
    for session in safe_dirs(root):
        sid = node_id(root.name, session.name)
        emit(lines, f'{sid}: "{FOLDER_ICON} {session.name}"')
        emit(lines, root_id, sid)

        # --- SESSION FILES (NEW) ---
        session_files = [f for f in session.iterdir() if f.is_file() and f.name not in IGNORED]
        emit_raw_files(lines, sid, session_files, "session_files", root.name, session.name)

        # --- PARTICIPANT RANGE ---
        label, participants = get_participant_range(session)
        if not participants:
            continue

        pid = node_id(root.name, session.name, label)
        emit(lines, f'{pid}: "{FOLDER_ICON} {label}"')
        emit(lines, sid, pid)

        # --- MODALITIES ---
        for modality in MODALITIES:

            mid = node_id(root.name, session.name, label, modality)
            emit(lines, f'{mid}: "{FOLDER_ICON} {modality}"')
            emit(lines, pid, mid)

            emit_grouped_files(
                lines,
                mid,
                root.name,
                session.name,
                participants,
                modality
            )


def generate(root_dir, out):
    root = Path(root_dir).resolve()
    lines = ["direction: right", ""]

    build(root, lines)

    with open(out, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", default="dataset_graph.d2")
    args = parser.parse_args()

    generate(args.directory, args.output)


if __name__ == "__main__":
    main()
