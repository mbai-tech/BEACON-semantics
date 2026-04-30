from pathlib import Path

from draw_scene import draw_scene
from scene_generator import generate_scene, save_scene_json


FAMILIES = [
    "sparse_clutter",
    "dense_clutter",
    "narrow_passage",
    "semantic_trap",
    "perturbed",
]
SCENES_PER_FAMILY = 100
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def clear_output_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            clear_output_dir(path)
            path.rmdir()


def main():
    for family in FAMILIES:
        images_dir = DATA_DIR / "images" / family
        scenes_dir = DATA_DIR / "scenes" / family
        clear_output_dir(images_dir)
        clear_output_dir(scenes_dir)
        for i in range(SCENES_PER_FAMILY):
            scene = generate_scene(family)
            seed = scene["seed"]
            base_name = f"{i:03d}_seed{seed}"
            image_path = images_dir / f"{base_name}.png"
            json_path = scenes_dir / f"{base_name}.json"

            draw_scene(scene, image_path)
            save_scene_json(scene, json_path)

    print("Done generating scenes.")


if __name__ == "__main__":
    main()
