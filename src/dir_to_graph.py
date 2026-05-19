#!/usr/bin/env python3

import argparse
from pathlib import Path


IGNORED = {
    ".git",
    "__pycache__",
    ".DS_Store",
    "node_modules",
}


def sanitize(name: str) -> str:
    return (
        name.replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def node_id(*parts):
    return "__".join(sanitize(p) for p in parts)


def collect_extensions(folder: Path):
    exts = set()

    for f in folder.rglob("*"):
        if f.is_file():
            exts.add(f.suffix.lower() or ".noext")

    return sorted(exts)


def emit_file_node(lines, parent_id, file_path, *id_parts):

    file_id = node_id(*id_parts, file_path.name)

    lines.append(f'{file_id}: "{file_path.name}"')
    lines.append(f"{parent_id} -> {file_id}")


def generate(root_dir: str, output: str):

    root = Path(root_dir).resolve()

    lines = [
        "direction: right",
        "",
        "vars: {",
        "  d2-config: {",
        "    layout-engine: elk",
        "  }",
        "}",
        "",
    ]

    root_id = sanitize(root.name)

    lines.append(f'{root_id}: "{root.name}"')
    lines.append("")

    #
    # Root-level files
    #
    root_files = sorted(
        [
            p for p in root.iterdir()
            if p.is_file() and p.name not in IGNORED
        ]
    )

    for f in root_files:
        emit_file_node(
            lines,
            root_id,
            f,
            root.name
        )

    #
    # Main stage folders
    #
    for stage in sorted(root.iterdir()):

        if not stage.is_dir():
            continue

        if stage.name in IGNORED:
            continue

        stage_id = node_id(root.name, stage.name)

        lines.append(f'{stage_id}: "{stage.name}"')
        lines.append(f"{root_id} -> {stage_id}")

        #
        # Files directly under stage
        #
        stage_files = sorted(
            [
                p for p in stage.iterdir()
                if p.is_file()
            ]
        )

        for f in stage_files:

            emit_file_node(
                lines,
                stage_id,
                f,
                root.name,
                stage.name
            )

        #
        # Subject folders
        #
        subjects = sorted(
            [
                p for p in stage.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.name
        )

        for subj in subjects:

            subj_id = node_id(
                root.name,
                stage.name,
                subj.name
            )

            lines.append(f'{subj_id}: "{subj.name}"')
            lines.append(f"{stage_id} -> {subj_id}")

            #
            # Collapse deep files by extension
            #
            exts = collect_extensions(subj)

            for ext in exts:

                ext_label = f"*{ext}"

                ext_id = node_id(
                    root.name,
                    stage.name,
                    subj.name,
                    ext
                )

                lines.append(f'{ext_id}: "{ext_label}"')
                lines.append(f"{subj_id} -> {ext_id}")

        lines.append("")

    with open(output, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {output}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        help="Dataset root"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="dataset_graph.d2"
    )

    args = parser.parse_args()

    generate(args.directory, args.output)


if __name__ == "__main__":
    main()