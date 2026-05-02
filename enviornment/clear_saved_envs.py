#!/usr/bin/env python3

import argparse
from pathlib import Path


VALID_FAMILIES = [
    "narrow_passage",
    "sparse_clutter",
    "dense_clutter",
    "semantic_trap",
    "perturbed",
]
BASE_DIR = Path(__file__).resolve().parent
SAVED_DIR = BASE_DIR / "saved_enviornments"


def clear_directory(directory):
    if not directory.exists():
        return 0

    removed = 0
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            removed += clear_directory(path)
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clear saved environment scene/image files."
    )
    parser.add_argument(
        "--family",
        choices=VALID_FAMILIES,
        help="Only clear saved files for one family. Defaults to all saved environments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.family:
        image_dir = SAVED_DIR / "images" / args.family
        scene_dir = SAVED_DIR / "scenes" / args.family
        removed = clear_directory(image_dir) + clear_directory(scene_dir)
        print(f"Cleared {removed} saved file(s) for family '{args.family}'.")
        return

    removed = clear_directory(SAVED_DIR / "images") + clear_directory(SAVED_DIR / "scenes")
    print(f"Cleared {removed} saved file(s) from saved environments.")


if __name__ == "__main__":
    main()
