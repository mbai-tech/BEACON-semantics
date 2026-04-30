#!/usr/bin/env python3

import argparse
from pathlib import Path

from draw_complex import draw_scene
from scene_complex import generate_scene, save_scene_json


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "complex"
VALID_FAMILIES = [
    "sparse",
    "cluttered",
    "collision_required",
    "collision_shortcut",
]


def ensure_output_dirs(family):
    images_dir = DATA_DIR / "images" / family
    scenes_dir = DATA_DIR / "scenes" / family
    images_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, scenes_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex mixed-shape environment scenes."
    )
    parser.add_argument(
        "--family",
        choices=VALID_FAMILIES,
        help="Generate scenes for one family only. Defaults to all families.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of scenes to generate per family.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    families = [args.family] if args.family else VALID_FAMILIES

    for family in families:
        images_dir, scenes_dir = ensure_output_dirs(family)
        for i in range(args.count):
            scene = generate_scene(family=family)
            image_path = images_dir / f"{i:03d}.png"
            scene_path = scenes_dir / f"{i:03d}.json"

            draw_scene(scene, image_path)
            save_scene_json(scene, scene_path)

    print(f"Generated {args.count} scene(s) per family in {DATA_DIR}")


if __name__ == "__main__":
    main()
