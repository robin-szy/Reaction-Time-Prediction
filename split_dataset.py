"""
Split the reaction-time scanpath dataset into train/validation data and a held-out test set.

Expected original structure:
    .
    ├── scanpaths/
    │   ├── file1.csv
    │   ├── file2.csv
    │   └── ...
    └── scanpaths_metadata.csv

Expected metadata format, whitespace-separated, no header:
    reaction_time filename.csv

Creates:
    reaction_split/
        train_val/
            *.csv
        test/
            *.csv
        test_flat/
            *.csv
        train_val_labels.csv
        test_labels.csv

Notes:
- train_val_labels.csv is for local training/validation only.
- test_labels.csv is for local checking only. Do not use it while fitting/tuning.
- test_flat/ simulates the professor's eval folder: CSV files only, no labels.
"""

from pathlib import Path
import argparse
import random
import shutil
import csv


# ===== Defaults =====
SCANPATHS_DIR = Path("./scanpaths")
METADATA_FILE = Path("./scanpaths_metadata.csv")
OUTPUT_DIR = Path("./scanpaths_split")
TEST_RATIO = 0.10
SEED = 98
COPY_FILES = True  # True = copy, False = move


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanpaths-dir", type=Path, default=SCANPATHS_DIR)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    return parser.parse_args()


def reset_output_dir(path: Path):
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Delete it manually first if you want to recreate the split."
        )
    path.mkdir(parents=True)


def copy_or_move(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def read_metadata(metadata_file: Path):
    """
    Reads whitespace-separated metadata with no header:
        reaction_time filename
    Returns a list of (filename, reaction_time).
    """
    items = []
    with metadata_file.open("r", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Bad metadata line {line_no}: expected '<reaction_time> <filename>', got: {line!r}"
                )

            reaction_time_str, filename = parts
            try:
                reaction_time = float(reaction_time_str)
            except ValueError as exc:
                raise ValueError(
                    f"Bad reaction time on line {line_no}: {reaction_time_str!r}"
                ) from exc

            if reaction_time < 0:
                raise ValueError(f"Negative reaction time on line {line_no}: {reaction_time}")

            items.append((filename, reaction_time))

    if not items:
        raise ValueError(f"No metadata rows found in {metadata_file}")

    return items


def write_labels(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for filename, reaction_time in rows:
            f.write(f"{reaction_time} {filename}\n")


def main():
    args = parse_args()
    copy_files = not args.move

    if not args.scanpaths_dir.exists():
        raise FileNotFoundError(f"Missing scanpaths folder: {args.scanpaths_dir}")
    if not args.metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {args.metadata_file}")
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be between 0 and 1")

    reset_output_dir(args.output_dir)

    items = read_metadata(args.metadata_file)

    # Validate all referenced files exist before copying anything meaningful.
    missing = [name for name, _ in items if not (args.scanpaths_dir / name).exists()]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Metadata references {len(missing)} missing scanpath files: {preview}{suffix}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(items)

    n_test = max(1, round(len(items) * args.test_ratio))
    test_items = items[:n_test]
    train_val_items = items[n_test:]

    for filename, reaction_time in train_val_items:
        src = args.scanpaths_dir / filename
        dst = args.output_dir / "train_val" / filename
        copy_or_move(src, dst, copy_files)

    for filename, reaction_time in test_items:
        src = args.scanpaths_dir / filename
        dst = args.output_dir / "test" / filename
        copy_or_move(src, dst, copy_files)

        flat_dst = args.output_dir / "test_flat" / filename
        flat_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, flat_dst)

    write_labels(args.output_dir / "train_val_labels.csv", train_val_items)
    write_labels(args.output_dir / "test_labels.csv", test_items)

    print("Done.")
    print(f"Total scanpaths:       {len(items)}")
    print(f"Train/validation set:  {len(train_val_items)}")
    print(f"Held-out test set:     {len(test_items)}")
    print(f"Train/val data:        {args.output_dir / 'train_val'}")
    print(f"Train/val labels:      {args.output_dir / 'train_val_labels.csv'}")
    print(f"Untouched test data:   {args.output_dir / 'test'}")
    print(f"Flat eval folder:      {args.output_dir / 'test_flat'}")
    print(f"Local test labels:     {args.output_dir / 'test_labels.csv'}")


if __name__ == "__main__":
    main()
