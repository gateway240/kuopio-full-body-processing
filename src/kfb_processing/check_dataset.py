import argparse
import logging
import pathlib
from pathlib import Path

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------

SUBJECT_IDS: list[str] = [f"{i:02d}" for i in range(1, 14)]

LABELS: list[str] = [
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

IMU_SUFFIXES: list[str] = [
    "_orientations.sto",
    "_accelerations.sto",
]

MOCAP_SUFFIXES: list[str] = [
    "_analog.sto",
    "_grfs.sto",
    "_markers.trc",
]

# -------------------------
# Helper functions
# -------------------------


def expected_files(labels: list[str], suffixes: list[str]) -> set[str]:
    return {f"{label}{suffix}" for label in labels for suffix in suffixes}


def check_folder(
    folder_path: Path,
    expected: set[str],
) -> tuple[set[str], set[Path], bool]:
    """
    Returns:
        missing files,
        extra files,
        whether the folder itself is missing
    """
    if not pathlib.Path(folder_path).is_dir():
        return expected, set(), True

    actual = {f for f in Path.iterdir(folder_path) if (folder_path / f).is_file()}

    missing = expected - actual
    extra = actual - expected

    return missing, extra, False


def check_subject(subject_dir: Path, subject_id: str) -> None:
    imu_dir = subject_dir / "imu"
    mocap_dir = subject_dir / "mocap"

    imu_expected: set[str] = expected_files(LABELS, IMU_SUFFIXES)
    mocap_expected: set[str] = expected_files(LABELS, MOCAP_SUFFIXES)

    imu_missing, _imu_extra, imu_missing_dir = check_folder(imu_dir, imu_expected)
    mocap_missing, _mocap_extra, mocap_missing_dir = check_folder(
        mocap_dir,
        mocap_expected,
    )

    logger.info("Subject %s", subject_id)

    logger.info("  [IMU]")
    if imu_missing_dir:
        logger.info("    WARN: imu folder is MISSING!")
    elif imu_missing:
        logger.info("    Missing files:")
        for f in sorted(imu_missing):
            logger.info("      - %s", f)
    else:
        logger.info("    No missing files")

        # if imu_extra:
        #     logger.info("    Extra files found:")
        #     for f in sorted(imu_extra):
        #         logger.info(f"      - {f}")
        # else:
        #     logger.info("    No extra files")

    logger.info("  [MOCAP]")
    if mocap_missing_dir:
        logger.info("    WARN: mocap folder is MISSING!")
    elif mocap_missing:
        logger.info("    Missing files:")
        for f in sorted(mocap_missing):
            logger.info("      - %s", f)
    else:
        logger.info("    No missing files")

        # if mocap_extra:
        #     logger.info("    Extra files found:")
        #     for f in sorted(mocap_extra):
        #         logger.info(f"      - {f}")
        # else:
        #     logger.info("    No extra files")


# -------------------------
# Main traversal
# -------------------------


def check_dataset(root_dir: Path) -> None:
    if not root_dir.is_dir():
        msg = f"Not a directory: {root_dir}"
        raise ValueError(msg)

    actual_subjects = {d for d in Path.iterdir(root_dir) if (root_dir / d).is_dir()}

    expected_subjects = set(SUBJECT_IDS)

    missing_subjects = expected_subjects - actual_subjects
    extra_subjects = actual_subjects - expected_subjects

    logger.info("=== DATASET CHECK REPORT ===")

    if missing_subjects:
        logger.info("\nMissing subject directories:")
        for s in sorted(missing_subjects):
            logger.info("  - %s", s)

    if extra_subjects:
        logger.info("\nUnexpected subject directories:")
        for s in sorted(extra_subjects):
            logger.info("  - %s", s)

    for subject_id in SUBJECT_IDS:
        subject_dir = root_dir / subject_id
        if subject_dir.is_dir():
            check_subject(subject_dir, subject_id)

    logger.info("\n=== CHECK COMPLETE ===")


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
