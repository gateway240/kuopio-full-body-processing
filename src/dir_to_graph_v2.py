#!/usr/bin/env python3
import hashlib
import argparse
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
    "_accelerations.sto",
    "_orientations.sto",
]

# -----------------------------
# UTILITIES
# -----------------------------
def sanitize(s: str):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))


def node_id(*parts):
    return "__".join(sanitize(p) for p in parts)


def emit(lines, a, b=None):
    if b is None:
        lines.append(a)
    else:
        lines.append(f"{a} -> {b}")


def is_numeric(name: str):
    return name.isdigit()


def range_label(names):
    nums = sorted(int(n) for n in names)
    return f"{nums[0]:02d}–{nums[-1]:02d}"


def safe_iterdir(path: Path):
    return [p for p in path.iterdir() if p.name not in IGNORED]


def match_suffix(filename: str):
    for suf in VALID_SUFFIXES:
        if filename.endswith(suf):
            return suf
    return None

def stable_id(root_name, *parts):
    path = Path(*parts).resolve()

    key = str(path).encode("utf-8")
    h = hashlib.md5(key).hexdigest()[:10]

    return f"{root_name}_{h}"

def group_key(f: Path):
    name = f.name

    # 1. known scientific suffix
    suf = match_suffix(name)
    if suf:
        return f"TRIAL{suf}"

    # 2. unknown BUT txt -> collapse all txt together
    if name.endswith(".txt"):
        return "TRIAL_<SENSOR>.txt"

    # 3. fallback: use extension
    ext = f.suffix.lower() if f.suffix else ".noext"
    return f"TRIAL.{ext.lstrip('.')}"


def collect_numeric_tree(numeric_dirs, base_path: Path):
    def tree():
        return defaultdict(tree)
    root = tree()

    for d in numeric_dirs:
        print("D: ",d)
        for f in d.rglob("*"):
            if not f.is_file() or f.name in IGNORED:
                continue

            rel = f.relative_to(d).parts

            node = root
            for part in rel[:-1]:
                node = node[part]

            node.setdefault("__files__", []).append(f)

    return root


def emit_tree(tree, parent_id, prefix, lines, root_name):
    for k, v in tree.items():

        if k == "__files__":
            groups = defaultdict(list)

            for f in v:
                groups[group_key(f)].append(f)

            for key in sorted(groups):
                fid = stable_id(root_name, *prefix, key)
                emit(lines, f'{fid}: "{FILE_ICON} {key}"')
                emit(lines, parent_id, fid)

            continue

        nid = stable_id(root_name, *prefix, k)
        emit(lines, f'{nid}: "{FOLDER_ICON} {k}"')
        emit(lines, parent_id, nid)

        emit_tree(v, nid, prefix + (k,), lines, root_name)

# -----------------------------
# BUILD ENTRY
# -----------------------------
def build(path: Path, parent_id: str, root_name: str, lines):
    node = node_id(root_name, *path.parts)
    emit(lines, f'{node}: "{FOLDER_ICON} {path.name}"')
    emit(lines, parent_id, node)

    entries = safe_iterdir(path)

    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    # numeric grouping check only for THIS directory
    if dirs and all(is_numeric(d.name) for d in dirs):
        label = range_label([d.name for d in dirs])
        group_id = node_id(root_name, *path.parts, "group")

        emit(lines, f'{group_id}: "{FOLDER_ICON} [{label}]"')
        emit(lines, node, group_id)

        tree = collect_numeric_tree(dirs, path)
        emit_tree(tree, group_id, (), lines, root_name)

        return

    # files first (correct scope)
    for f in files:
        fid = node_id(root_name, *f.parts)
        emit(lines, f'{fid}: "{FILE_ICON} {f.name}"')
        emit(lines, node, fid)

    # then recurse dirs
    for d in dirs:
        build(d, node, root_name, lines)


def generate(root_dir, out):

    root = Path(root_dir).resolve()
    lines = ["direction: right", ""]

    root_id = node_id(root.name)
    emit(lines, f'{root_id}: "{FOLDER_ICON} {root.name}"')

    for item in safe_iterdir(root):

        if item.is_dir():
            build(item, root_id, root.name, lines)
        else:
            fid = node_id(root.name, item.name)
            emit(lines, f'{fid}: "{FILE_ICON} {item.name}"')
            emit(lines, root_id, fid)

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