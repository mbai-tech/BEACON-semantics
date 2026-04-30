#!/usr/bin/env python3

import sys
from pathlib import Path

from scene_generator import generate_scene, save_scene_json


VALID_FAMILIES = [
    "narrow_passage",
    "sparse_clutter",
    "dense_clutter",
    "semantic_trap",
    "perturbed",
]
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def clear_output_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def next_available_path(directory, base_name, suffix):
    candidate = directory / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = directory / f"{base_name}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in VALID_FAMILIES:
        print("Usage: python3 enviornment/run_family.py <family> [seed]")
        print("Valid families:", ", ".join(VALID_FAMILIES))
        sys.exit(1)

    family = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else None
    scene = generate_scene(family, seed=seed)

    if seed is None:
        output_dir = DATA_DIR
        images_dir = output_dir / "images" / family
        scenes_dir = output_dir / "scenes" / family
        clear_output_dir(images_dir)
        clear_output_dir(scenes_dir)
        json_path = scenes_dir / "scene.json"
        image_path = images_dir / "scene.png"
    else:
        output_dir = Path(__file__).resolve().parent / "saved_enviornments"
        images_dir = output_dir / "images" / family
        scenes_dir = output_dir / "scenes" / family
        images_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"seed{scene['seed']}"
        json_path = next_available_path(scenes_dir, base_name, ".json")
        image_path = next_available_path(images_dir, base_name, ".png")

    save_scene_json(scene, json_path)

    try:
        from draw_scene import draw_scene
    except Exception:
        draw_scene = None

    if callable(draw_scene):
        draw_scene(scene, image_path)
        print(f"Saved image to {image_path}")

    print(f"Seed: {scene['seed']}")
    print(f"Saved scene JSON to {json_path}")


if __name__ == "__main__":
    main()
