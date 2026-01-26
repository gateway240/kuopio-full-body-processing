import argparse
import os
from typing import List, Set, Tuple

# -------------------------
# Configuration
# -------------------------

SUBJECT_IDS: List[str] = [f"{i:02d}" for i in range(1, 14)]

LABELS: List[str] = [
    "static_cal",
    "dyn_sara",
    "dyn_score_hip",
    "dyn_score_ankle",
    "crouch_lift",
    "crouch_rotate",
    "curls",
    "kettlebell",
    "squats_deep",
    "half_jacks",
    "squat_jumps",
    "box_jabs",
    "box_combos",
    "chair_push_right",
    "chair_push_left",
    "arm_hang",
    "heavy_lift",
    "back_fly",
    "side_fly",
    "walking",
    "jogging",
    "crab_walking",
]

IMU_SUFFIXES: List[str] = [
    "_orientations.sto",
    "_accelerations.sto",
]

MOCAP_SUFFIXES: List[str] = [
    "_analog.sto",
    "_grfs.sto",
    "_markers.trc",
]

# -------------------------
# Helper functions
# -------------------------


def expected_files(labels: List[str], suffixes: List[str]) -> Set[str]:
    return {f"{label}{suffix}" for label in labels for suffix in suffixes}


def check_folder(
    folder_path: str, expected: Set[str]
) -> Tuple[Set[str], Set[str], bool]:
    """
    Returns:
        missing files,
        extra files,
        whether the folder itself is missing
    """
    if not os.path.isdir(folder_path):
        return expected, set(), True

    actual: Set[str] = {
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    }

    missing: Set[str] = expected - actual
    extra: Set[str] = actual - expected

    return missing, extra, False


def check_subject(subject_dir: str, subject_id: str) -> None:
    imu_dir: str = os.path.join(subject_dir, "imu")
    mocap_dir: str = os.path.join(subject_dir, "mocap")

    imu_expected: Set[str] = expected_files(LABELS, IMU_SUFFIXES)
    mocap_expected: Set[str] = expected_files(LABELS, MOCAP_SUFFIXES)

    imu_missing, imu_extra, imu_missing_dir = check_folder(imu_dir, imu_expected)
    mocap_missing, mocap_extra, mocap_missing_dir = check_folder(
        mocap_dir, mocap_expected
    )

    print(f"\nSubject {subject_id}")

    print("  [IMU]")
    if imu_missing_dir:
        print("    WARN: imu folder is MISSING!")
    else:
        if imu_missing:
            print("    Missing files:")
            for f in sorted(imu_missing):
                print(f"      - {f}")
        else:
            print("    No missing files")

        # if imu_extra:
        #     print("    Extra files found:")
        #     for f in sorted(imu_extra):
        #         print(f"      - {f}")
        # else:
        #     print("    No extra files")

    print("  [MOCAP]")
    if mocap_missing_dir:
        print("    WARN: mocap folder is MISSING!")
    else:
        if mocap_missing:
            print("    Missing files:")
            for f in sorted(mocap_missing):
                print(f"      - {f}")
        else:
            print("    No missing files")

        # if mocap_extra:
        #     print("    Extra files found:")
        #     for f in sorted(mocap_extra):
        #         print(f"      - {f}")
        # else:
        #     print("    No extra files")


# -------------------------
# Main traversal
# -------------------------


def check_dataset(root_dir: str) -> None:
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    actual_subjects: Set[str] = {
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    }

    expected_subjects: Set[str] = set(SUBJECT_IDS)

    missing_subjects: Set[str] = expected_subjects - actual_subjects
    extra_subjects: Set[str] = actual_subjects - expected_subjects

    print("=== DATASET CHECK REPORT ===")

    if missing_subjects:
        print("\nMissing subject directories:")
        for s in sorted(missing_subjects):
            print(f"  - {s}")

    if extra_subjects:
        print("\nUnexpected subject directories:")
        for s in sorted(extra_subjects):
            print(f"  - {s}")

    for subject_id in SUBJECT_IDS:
        subject_dir: str = os.path.join(root_dir, subject_id)
        if os.path.isdir(subject_dir):
            check_subject(subject_dir, subject_id)

    print("\n=== CHECK COMPLETE ===")


# -------------------------
# Entry point
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate imu/ and mocap/ folders.")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Root directory containing subject folders",
    )

    args = parser.parse_args()
    check_dataset(args.source_dir)


if __name__ == "__main__":
    main()
