#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BEACON_ROOT = REPO_ROOT / "BEACON"
if str(BEACON_ROOT) not in sys.path:
    sys.path.insert(0, str(BEACON_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "data" / ".matplotlib"))

from BEACON.render import render_scene_with_path  # type: ignore
from experiments.run_semantics_planner_experiments import (  # type: ignore
    BASE_PARAMETERS,
    BeaconDStarLitePlanner,
    HumanLikePlanner,
    InstrumentedExecutor,
    MoveAction,
    PLANNERS,
    PlannerConfig,
    PushAction,
    PushPolicy,
    PyBulletEnv,
    WeightedBeaconInteractionCost,
    WeightedInteractionEvaluator,
    World,
    compute_path_length,
    configure_environment,
    planner_factory,
    semantics_env_to_beacon_scene,
)


DEFAULT_ENVIRONMENTS = [
    "open_room",
    "corridor",
    "sparse",
    "collision_shortcut",
    "cluttered",
    "collision_required",
    "mixed_materials",
]

PLANNER_LABELS = {
    "dstar_lite": "D* Lite",
    "bug1": "Bug1",
    "bug2": "Bug2",
    "beacon_human_like": "BEACON",
}


@dataclass
class PlannerRun:
    planner: str
    success: bool
    path_points: list[list[float]]
    path_length: float
    steps: int
    scene: dict


@dataclass
class ExampleScene:
    environment: str
    seed: int
    score: float
    planner_runs: dict[str, PlannerRun]


def centered_path(path: list[tuple[int, int]]) -> list[list[float]]:
    return [[float(x) + 0.5, float(y) + 0.5] for x, y in path]


def euclidean_path_length_xy(path_points: list[list[float]]) -> float:
    if len(path_points) < 2:
        return 0.0
    return sum(math.dist(a, b) for a, b in zip(path_points, path_points[1:]))


def run_standard_planner(environment_name: str, seed: int, planner_name: str, params: dict) -> PlannerRun:
    environment = configure_environment(environment_name, seed, params)
    planner = planner_factory(planner_name, environment)
    executor = InstrumentedExecutor(
        environment,
        planner_name,
        planner,
        WeightedInteractionEvaluator(environment, energy_weight=float(params["energy_weight"])),
    )
    executor.execute()

    scene = semantics_env_to_beacon_scene(environment)
    path_points = centered_path(executor.robot_path)
    return PlannerRun(
        planner=planner_name,
        success=bool(executor.reached_goal),
        path_points=path_points,
        path_length=euclidean_path_length_xy(path_points),
        steps=max(0, len(path_points) - 1),
        scene=scene,
    )


def run_beacon_planner(environment_name: str, seed: int, params: dict) -> PlannerRun:
    environment = configure_environment(environment_name, seed, params)
    scene = semantics_env_to_beacon_scene(environment)
    config = PlannerConfig(step=0.6, robot_radius=float(params["robot_radius"]))
    world = World.from_scene(scene)
    dstar_planner = BeaconDStarLitePlanner(world, config=config)
    push_policy = PushPolicy(
        WeightedBeaconInteractionCost(
            energy_weight=float(params["energy_weight"]),
            push_duration=config.step_duration,
        )
    )
    planner = HumanLikePlanner(dstar_planner, push_policy, world, alpha=environment.robot.sensing_radius)
    env = PyBulletEnv(world)

    executed_trajectory = [[float(world.start[0]), float(world.start[1]), 0.0]]
    goal_tolerance = max(config.step, config.robot_radius)
    max_steps = max(50, environment.width * environment.height * 4)
    success = False

    for _ in range(max_steps):
        robot_position = env.get_robot_position()
        if math.dist(robot_position, world.goal) <= goal_tolerance:
            success = True
            break

        action = planner.step(robot_position)
        if action is None:
            break

        env.execute(action)
        if isinstance(action, MoveAction):
            path_segment = action.path_segment or [
                [robot_position[0], robot_position[1], 0.0],
                [action.target[0], action.target[1], 0.0],
            ]
            if executed_trajectory[-1][:2] != path_segment[0][:2]:
                executed_trajectory.append(
                    [float(path_segment[0][0]), float(path_segment[0][1]), float(path_segment[0][2])]
                )
            for point in path_segment[1:]:
                executed_trajectory.append([float(point[0]), float(point[1]), float(point[2])])
        elif isinstance(action, PushAction):
            pose = env.get_robot_position()
            executed_trajectory.append([float(pose[0]), float(pose[1]), 0.0])

    final_pose = env.get_robot_position()
    if math.dist(final_pose, world.goal) <= goal_tolerance:
        success = True

    final_scene = world.to_scene()
    path_points = [[float(point[0]), float(point[1])] for point in executed_trajectory]
    return PlannerRun(
        planner="beacon_human_like",
        success=success,
        path_points=path_points,
        path_length=compute_path_length(executed_trajectory),
        steps=max(0, len(path_points) - 1),
        scene=final_scene,
    )


def scene_score(planner_runs: dict[str, PlannerRun]) -> float:
    lengths = [run.path_length for run in planner_runs.values()]
    steps = [run.steps for run in planner_runs.values()]
    spread = max(lengths) - min(lengths)
    average_steps = sum(steps) / max(1, len(steps))
    success_bonus = 10.0 if all(run.success for run in planner_runs.values()) else 0.0
    return success_bonus + spread + 0.05 * average_steps


def scene_matches_mode(planner_runs: dict[str, PlannerRun], success_mode: str) -> bool:
    if success_mode == "all":
        return all(run.success for run in planner_runs.values())
    if success_mode == "beacon":
        return planner_runs["beacon_human_like"].success
    raise ValueError(f"Unsupported success mode: {success_mode}")


def collect_examples(
    target_scene_count: int,
    max_seed: int,
    environments: list[str],
    params: dict,
    success_mode: str,
) -> list[ExampleScene]:
    candidates: list[ExampleScene] = []

    for environment_name in environments:
        for seed in range(max_seed + 1):
            beacon_run = run_beacon_planner(environment_name, seed, params)
            if success_mode == "beacon" and not beacon_run.success:
                continue

            planner_runs = {
                "beacon_human_like": beacon_run,
                "dstar_lite": run_standard_planner(environment_name, seed, "dstar_lite", params),
                "bug1": run_standard_planner(environment_name, seed, "bug1", params),
                "bug2": run_standard_planner(environment_name, seed, "bug2", params),
            }
            if not scene_matches_mode(planner_runs, success_mode):
                continue

            candidates.append(
                ExampleScene(
                    environment=environment_name,
                    seed=seed,
                    score=scene_score(planner_runs),
                    planner_runs=planner_runs,
                )
            )
            if len(candidates) >= max(target_scene_count * 2, target_scene_count):
                break
        if len(candidates) >= max(target_scene_count * 2, target_scene_count):
            break

    candidates.sort(key=lambda example: (example.score, example.environment, -example.seed), reverse=True)
    return candidates[:target_scene_count]


def render_examples(examples: list[ExampleScene], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for scene_index, example in enumerate(examples, start=1):
        for planner_name in PLANNERS:
            run = example.planner_runs[planner_name]
            file_name = f"{scene_index:02d}_{example.environment}_seed{example.seed:02d}_{planner_name}.png"
            output_path = output_dir / file_name
            render_scene_with_path(
                run.scene,
                output_path,
                path_points=run.path_points,
                title_suffix=f"({PLANNER_LABELS[planner_name]})",
            )
            manifest.append(
                {
                    "environment": example.environment,
                    "seed": example.seed,
                    "planner": planner_name,
                    "label": PLANNER_LABELS[planner_name],
                    "path_length": round(run.path_length, 3),
                    "steps": run.steps,
                    "output": str(output_path),
                }
            )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a gallery of example run PNGs for D* Lite, Bug1, Bug2, and BEACON."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Approximate number of PNGs to generate. Rounded up to a multiple of four.",
    )
    parser.add_argument(
        "--scene-count",
        type=int,
        help="Exact number of scenes to generate. Overrides --count when provided.",
    )
    parser.add_argument(
        "--max-seed",
        type=int,
        default=12,
        help="Largest seed to consider while searching for successful scenes.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "examples" / "example_runs"),
        help="Directory where the PNGs and manifest JSON should be written.",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=DEFAULT_ENVIRONMENTS,
        help="Environment names to search in priority order.",
    )
    parser.add_argument(
        "--success-mode",
        choices=["all", "beacon"],
        default="all",
        help="Choose whether all planners must succeed or only BEACON must succeed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_scene_count = args.scene_count if args.scene_count is not None else max(1, math.ceil(args.count / len(PLANNERS)))
    output_dir = Path(args.output_dir).resolve()
    params = dict(BASE_PARAMETERS)

    examples = collect_examples(target_scene_count, args.max_seed, args.environments, params, args.success_mode)
    if not examples:
        raise SystemExit("No matching scenes were found. Try increasing --max-seed or changing --success-mode.")

    manifest = render_examples(examples, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Rendered {len(manifest)} PNGs across {len(examples)} scenes.")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
