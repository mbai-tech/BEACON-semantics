"""View algorithm paths on a 3D PyBullet scene.

Runs Bug1, Bug2, D* Lite, and BEACON on one scene and draws each
path as a colored line in the PyBullet 3D viewer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "enviornment"))

from beacon.environment.scene_generator_shapely import generate_scene
from beacon.core.bug_algorithm        import run_bug
from beacon.core.bug2_algorithm       import run_bug2
from beacon.core.dstar_lite_algorithm import run_dstar_lite
from beacon.planning.beacon_planner   import run_beacon

ALGORITHMS = {
    "Bug1":    (run_bug,       [1.0, 0.2, 0.2]),   # red
    "Bug2":    (run_bug2,      [0.2, 0.6, 1.0]),   # blue
    "D* Lite": (run_dstar_lite,[0.2, 0.9, 0.3]),   # green
    "BEACON":  (run_beacon,    [1.0, 0.8, 0.0]),   # yellow
}

PATH_Z        = 0.08   # draw paths just above ground
SENSING_RANGE = 0.55   # metres — matches beacon/core/constants.py


def draw_sphere(center: list, radius: float, color: list, n: int = 16) -> None:
    """Draw a wireframe sphere in PyBullet using debug lines (2 great circles)."""
    import pybullet as p
    import math

    cx, cy, cz = center

    for axis in range(2):   # XY, XZ planes only
        for i in range(n):
            a0 = 2 * math.pi * i       / n
            a1 = 2 * math.pi * (i + 1) / n
            if axis == 0:   # XY plane
                p0 = [cx + radius * math.cos(a0), cy + radius * math.sin(a0), cz]
                p1 = [cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz]
            elif axis == 1: # XZ plane
                p0 = [cx + radius * math.cos(a0), cy, cz + radius * math.sin(a0)]
                p1 = [cx + radius * math.cos(a1), cy, cz + radius * math.sin(a1)]
            else:           # YZ plane
                p0 = [cx, cy + radius * math.cos(a0), cz + radius * math.sin(a0)]
                p1 = [cx, cy + radius * math.cos(a1), cz + radius * math.sin(a1)]
            p.addUserDebugLine(p0, p1, lineColorRGB=color, lineWidth=2)


def draw_sensing_spheres(path: list, color: list, every: int = 30) -> None:
    """Draw sensing-range spheres at evenly spaced positions along a path."""
    faded = [c * 0.5 for c in color]   # dimmer than the path
    for i in range(0, len(path), every):
        x, y = path[i][0], path[i][1]
        draw_sphere([x, y, SENSING_RANGE], SENSING_RANGE, faded)


def draw_path(path: list, color: list, width: float = 0.06) -> None:
    """Draw path as flat box segments so thickness is not GPU-capped."""
    import pybullet as p
    import math

    vis_color = color + [1.0]   # RGBA

    for i in range(1, len(path)):
        x0, y0 = path[i - 1][0], path[i - 1][1]
        x1, y1 = path[i][0],     path[i][1]

        dx, dy  = x1 - x0, y1 - y0
        length  = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue

        mx, my  = (x0 + x1) / 2, (y0 + y1) / 2
        angle   = math.atan2(dy, dx)
        half    = angle / 2
        quat    = [0.0, 0.0, math.sin(half), math.cos(half)]

        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, width / 2, 0.015],
            rgbaColor=vis_color,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[mx, my, PATH_Z],
            baseOrientation=quat,
        )


def add_label(text: str, position: list, color: list) -> None:
    import pybullet as p
    p.addUserDebugText(
        text,
        [position[0], position[1], PATH_Z + 0.15],
        textColorRGB=color,
        textSize=1.2,
    )


def setup_scene(scene: dict) -> None:
    import pybullet as p
    import pybullet_data
    from render_scene_pybullet import setup_pybullet_scene

    for obs in scene["obstacles"]:
        if "class_true" not in obs and "true_class" in obs:
            obs["class_true"] = obs["true_class"]

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    setup_pybullet_scene(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="narrow_passage",
                        choices=["sparse_clutter", "dense_clutter",
                                 "narrow_passage", "semantic_trap"])
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--alg",    default=None,
                        choices=list(ALGORITHMS.keys()),
                        help="Run one algorithm only")
    args = parser.parse_args()

    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("pybullet not installed. Run: pip install pybullet")
        sys.exit(1)

    scene = generate_scene(family=args.family, seed=args.seed)
    print(f"\nScene: {args.family}  seed={args.seed}")
    print(f"Obstacles: {len(scene['obstacles'])}")
    print(f"Start: {scene['start'][:2]}  Goal: {scene['goal'][:2]}\n")

    algs_to_run = {k: v for k, v in ALGORITHMS.items()
                   if args.alg is None or k == args.alg}

    results = {}
    for name, (fn, color) in algs_to_run.items():
        print(f"  Running {name} ...", end=" ", flush=True)
        result = fn(scene)
        path_len = sum(
            ((result.path[i][0] - result.path[i-1][0])**2 +
             (result.path[i][1] - result.path[i-1][1])**2) ** 0.5
            for i in range(1, len(result.path))
        )
        damage = 0.0
        if result.scene_summary:
            damage = result.scene_summary.total_semantic_damage
        results[name] = (result, color, path_len, damage)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"{status}  path={path_len:.2f}m  damage={damage:.2f}")

    print("\nOpening PyBullet 3D viewer ...")
    p.connect(p.GUI)
    setup_scene(scene)

    for name, (result, color, path_len, damage) in results.items():
        draw_path(result.path, color)
        draw_sensing_spheres(result.path, color)
        if result.path:
            mid = result.path[len(result.path) // 2]
            label = f"{name} {'OK' if result.success else 'FAIL'}"
            add_label(label, [mid[0], mid[1]], color)

    # Legend in top corner
    legend_lines = [
        f"  Red    = Bug1",
        f"  Blue   = Bug2",
        f"  Green  = D* Lite",
        f"  Yellow = BEACON",
        f"  Family : {args.family}",
        f"  Seed   : {args.seed}",
    ]
    for i, line in enumerate(legend_lines):
        xmin = scene["workspace"][0]
        ymax = scene["workspace"][3]
        p.addUserDebugText(
            line,
            [xmin + 0.2, ymax - 0.4 * (i + 1), 0.5],
            textColorRGB=[1, 1, 1],
            textSize=1.0,
        )

    print("Controls: rotate/zoom with mouse. Press Q to quit.\n")
    while True:
        keys = p.getKeyboardEvents()
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
