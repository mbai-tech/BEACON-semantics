"""Watch BEACON physically push movable obstacles in PyBullet.

The robot is a real rigid body following BEACON's planned path.
Movable obstacles (blue) have mass and will be pushed by physics.
Safe / fragile obstacles are static and won't move.

"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "enviornment"))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("pybullet not installed. Run: python3 -m pip install pybullet")
    sys.exit(1)

from beacon.environment.scene_generator_shapely import generate_scene
import beacon.core.planner as _planner_mod
from beacon.core.planner import robot_clearance_to_obstacle

def _reveal_sphere(scene: dict, position, sensing_range: float) -> list:
    """3D sphere sensing: accounts for obstacle height when computing distance."""
    newly = []
    for obs in scene["obstacles"]:
        if obs["observed"]:
            continue
        clearance_2d = robot_clearance_to_obstacle(position, obs)
        cls          = obs.get("class_true", obs.get("true_class", "movable"))
        obs_height   = HEIGHT_BY_CLASS.get(cls, 0.25)
        robot_z      = ROBOT_RADIUS
        z_gap        = max(0.0, robot_z - obs_height)
        clearance_3d = math.sqrt(clearance_2d ** 2 + z_gap ** 2)
        if clearance_3d <= sensing_range:
            obs["observed"] = True
            newly.append(obs)
    return newly

_planner_mod.reveal_nearby_obstacles = _reveal_sphere

from beacon.planning.beacon_planner import run_beacon

ROBOT_RADIUS  = 0.18
ROBOT_MASS    = 8.0    # heavy enough to push obstacles
ROBOT_SPEED   = 1.8    # m/s — slow enough to avoid tunneling
REACH_DIST    = 0.12   # consider waypoint reached within this distance
MAX_STEPS_WP  = 120    # max physics steps per waypoint before skipping
SENSING_RANGE = 0.55   # metres — matches beacon/core/constants.py
HEIGHT_BY_CLASS = {"safe": 0.18, "movable": 0.28, "fragile": 0.38}


def make_demo_scene() -> dict:
    """Hardcoded scene: movable block directly between start and goal."""
    return {
        "family": "demo",
        "seed": 0,
        "workspace": [0, 8, 0, 8],
        "start": [1.0, 4.0, 0.0],
        "goal":  [7.0, 4.0, 0.0],
        "obstacles": [
            {
                "id": 0,
                "class_true": "movable",
                "true_class": "movable",
                "shape_type": "rectangle",
                "vertices": [
                    [3.6, 3.4], [4.4, 3.4], [4.4, 4.6], [3.6, 4.6]
                ],
            },
            {
                "id": 1,
                "class_true": "fragile",
                "true_class": "fragile",
                "shape_type": "rectangle",
                "vertices": [
                    [3.6, 1.0], [4.4, 1.0], [4.4, 2.2], [3.6, 2.2]
                ],
            },
            {
                "id": 2,
                "class_true": "fragile",
                "true_class": "fragile",
                "shape_type": "rectangle",
                "vertices": [
                    [3.6, 5.8], [4.4, 5.8], [4.4, 7.0], [3.6, 7.0]
                ],
            },
        ],
    }


def setup_scene(scene: dict) -> None:
    from render_scene_pybullet import setup_pybullet_scene
    for obs in scene["obstacles"]:
        if "class_true" not in obs and "true_class" in obs:
            obs["class_true"] = obs["true_class"]
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    setup_pybullet_scene(scene)


def create_robot(start: list) -> int:
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=ROBOT_RADIUS)
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=ROBOT_RADIUS, rgbaColor=[1.0, 0.3, 1.0, 1.0]
    )
    robot_id = p.createMultiBody(
        baseMass=ROBOT_MASS,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[start[0], start[1], ROBOT_RADIUS],
    )
    p.changeDynamics(robot_id, -1,
                     lateralFriction=0.5,
                     linearDamping=0.9,
                     ccdSweptSphereRadius=ROBOT_RADIUS)  # prevents tunneling
    return robot_id


def follow_path(robot_id: int, path: list) -> None:
    STUCK_DIST = 0.02   # if robot moves less than this per check, it's stuck
    STUCK_CHECK_EVERY = 20  # steps between stuck checks

    for waypoint in path:
        tx, ty = waypoint[0], waypoint[1]
        last_check_pos = None

        for step in range(MAX_STEPS_WP):
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            dx, dy = tx - pos[0], ty - pos[1]
            dist   = math.sqrt(dx * dx + dy * dy)

            if dist < REACH_DIST:
                break

            # Stuck detection: skip waypoint if barely moved
            if step % STUCK_CHECK_EVERY == 0:
                if last_check_pos is not None:
                    moved = math.sqrt((pos[0] - last_check_pos[0])**2 +
                                      (pos[1] - last_check_pos[1])**2)
                    if moved < STUCK_DIST:
                        break   # stuck — skip to next waypoint
                last_check_pos = pos

            vx = dx / dist * ROBOT_SPEED
            vy = dy / dist * ROBOT_SPEED
            p.resetBaseVelocity(robot_id, [vx, vy, 0.0])
            p.stepSimulation()
            time.sleep(1.0 / 960.0)

    p.resetBaseVelocity(robot_id, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="semantic_trap",
                        choices=["sparse_clutter", "dense_clutter",
                                 "narrow_passage", "semantic_trap"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo", action="store_true",
                        help="Use a simple hardcoded scene with one blue block in the path")
    args = parser.parse_args()

    if args.demo:
        print("\nUsing demo scene: movable block directly in the path")
        scene = make_demo_scene()
    else:
        print(f"\nGenerating scene: {args.family}  seed={args.seed}")
        scene = generate_scene(family=args.family, seed=args.seed)

    print("Running BEACON planner ...")
    result = run_beacon(scene)
    status = "SUCCESS" if result.success else "FAILED"
    damage = result.scene_summary.total_semantic_damage if result.scene_summary else 0.0
    print(f"  {status}  path steps={len(result.path)}  damage={damage:.2f}")

    movable = sum(
        1 for o in scene["obstacles"]
        if o.get("class_true", o.get("true_class", "")) == "movable"
    )
    print(f"  Movable obstacles (blue, will be pushed): {movable}")

    print("\nOpening PyBullet — watch the purple robot push the blue obstacles ...")
    print("Controls: rotate/zoom with mouse.  Q to quit.\n")

    p.connect(p.GUI)
    setup_scene(scene)

    p.addUserDebugText(
        "Purple = robot   Blue = movable (pushable)   Red = fragile (static)",
        [scene["workspace"][0] + 0.1, scene["workspace"][3] - 0.4, 0.5],
        textColorRGB=[1, 1, 1],
        textSize=1.0,
    )

    robot_id = create_robot(scene["start"])

    # Brief pause so you can see the initial scene before movement
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    follow_path(robot_id, result.path)

    print("Path complete. Press Q to quit.")
    while True:
        keys = p.getKeyboardEvents()
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
