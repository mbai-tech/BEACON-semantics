#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ishita/Documents/GitHub/BEACON-semantics")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEMANTICS_ROOT = Path("/Users/ishita/Documents/GitHub/Semantics")
if str(SEMANTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTICS_ROOT))

from robot_push_planner.core.environment import Environment  # type: ignore
from robot_push_planner.core.robot import Robot  # type: ignore
from robot_push_planner.environments.base_env import _make_obstacle  # type: ignore

import experiments.run_semantics_planner_experiments as exp


OUTPUT_DIR = ROOT / "data/experiments/push_heavy"
RAW_CSV = OUTPUT_DIR / "push_heavy_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "push_heavy_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "push_heavy_summary.json"

PUSH_HEAVY_ENVIRONMENTS = [
    "push_wall",
    "push_chain",
    "push_corridor",
    "push_shortcut",
]
SEEDS = [21, 22, 23]
PLANNERS = ["dstar_lite", "bug1", "bug2", "beacon_human_like"]

PUSH_HEAVY_PARAMS = {
    "robot_radius": 0.25,
    "obstacle_density_scale": 1.0,
    "obstacle_size_scale": 1.0,
    "friction_scale": 1.0,
    "energy_weight": 0.05,
    "obstacle_count_scale": 1.0,
}
PUSH_HEAVY_BEACON_STEP = 0.6
PUSH_HEAVY_BEACON_ALPHA = 2.5


def base_environment(name: str, start=(1, 10), goal=(18, 10)) -> Environment:
    robot = Robot(position=start, radius=0.25, max_force=220.0, sensing_radius=3.0)
    return Environment(
        width=20,
        height=20,
        robot=robot,
        start=start,
        goal=goal,
        name=name,
    )


def add_obstacle(env: Environment, rng: random.Random, obstacle_id: str, position: tuple[int, int], semantic_class: str, geometry: str = "rectangle", volume: float = 1.2) -> None:
    env.add_obstacle(_make_obstacle(rng, obstacle_id, position, semantic_class, geometry=geometry, volume=volume))


def add_light_movable(env: Environment, rng: random.Random, obstacle_id: str, position: tuple[int, int], geometry: str = "rectangle", volume: float = 0.18, density: float = 12.0, friction: float = 0.08) -> None:
    add_obstacle(env, rng, obstacle_id, position, "movable", geometry=geometry, volume=volume)
    obstacle = env.obstacles[-1]
    obstacle.volume = volume
    obstacle.density = density
    obstacle.friction_coefficient = friction


def add_rigid_wall(env: Environment, rng: random.Random, obstacle_id: str, position: tuple[int, int], geometry: str = "rectangle", volume: float = 1.0) -> None:
    add_obstacle(env, rng, obstacle_id, position, "safe", geometry=geometry, volume=volume)
    obstacle = env.obstacles[-1]
    obstacle.semantic_class = "wall"
    obstacle.pushable = False


def add_corridor_shell(
    env: Environment,
    rng: random.Random,
    corridor_axis: str,
    corridor_value: int,
    start_gate: int,
    end_gate: int,
    thickness: int = 1,
    cap_extent: int = 2,
    volume: float = 1.0,
) -> None:
    idx = 1
    if corridor_axis == "horizontal":
        for x in range(start_gate, end_gate + 1):
            for offset in range(1, thickness + 1):
                add_rigid_wall(env, rng, f"safe_{idx}", (x, corridor_value - offset), volume=volume)
                idx += 1
                add_rigid_wall(env, rng, f"safe_{idx}", (x, corridor_value + offset), volume=volume)
                idx += 1
        for y in range(max(1, corridor_value - thickness - cap_extent), min(env.height - 1, corridor_value + thickness + cap_extent + 1)):
            if abs(y - corridor_value) <= thickness:
                continue
            add_rigid_wall(env, rng, f"safe_{idx}", (start_gate, y), volume=volume)
            idx += 1
            add_rigid_wall(env, rng, f"safe_{idx}", (end_gate, y), volume=volume)
            idx += 1
        return

    for y in range(start_gate, end_gate + 1):
        for offset in range(1, thickness + 1):
            add_rigid_wall(env, rng, f"safe_{idx}", (corridor_value - offset, y), volume=volume)
            idx += 1
            add_rigid_wall(env, rng, f"safe_{idx}", (corridor_value + offset, y), volume=volume)
            idx += 1
    for x in range(max(1, corridor_value - thickness - cap_extent), min(env.width - 1, corridor_value + thickness + cap_extent + 1)):
        if abs(x - corridor_value) <= thickness:
            continue
        add_rigid_wall(env, rng, f"safe_{idx}", (x, start_gate), volume=volume)
        idx += 1
        add_rigid_wall(env, rng, f"safe_{idx}", (x, end_gate), volume=volume)
        idx += 1


def build_push_wall(seed: int) -> Environment:
    rng = random.Random(seed)
    env = base_environment("push_wall")
    add_corridor_shell(env, rng, corridor_axis="horizontal", corridor_value=10, start_gate=2, end_gate=17, thickness=1, volume=0.9)
    for x in range(8, 11):
        add_light_movable(env, rng, f"movable_{x}", (x, 10), geometry="rectangle")
    return env


def build_push_chain(seed: int) -> Environment:
    rng = random.Random(seed)
    env = base_environment("push_chain", start=(1, 9), goal=(18, 9))
    add_corridor_shell(env, rng, corridor_axis="horizontal", corridor_value=9, start_gate=2, end_gate=17, thickness=1, volume=0.9)
    for x in range(7, 12):
        add_light_movable(env, rng, f"chain_{x}", (x, 9), geometry="rectangle")
    return env


def build_push_corridor(seed: int) -> Environment:
    rng = random.Random(seed)
    env = base_environment("push_corridor", start=(1, 10), goal=(18, 10))
    add_corridor_shell(env, rng, corridor_axis="horizontal", corridor_value=10, start_gate=2, end_gate=10, thickness=1, volume=0.95)
    add_corridor_shell(env, rng, corridor_axis="vertical", corridor_value=10, start_gate=10, end_gate=17, thickness=1, volume=0.95)
    for pos in [(7, 10), (8, 10), (10, 11), (10, 12)]:
        add_light_movable(env, rng, f"plug_{pos[0]}_{pos[1]}", pos, geometry="rectangle")
    return env


def build_push_shortcut(seed: int) -> Environment:
    rng = random.Random(seed)
    env = base_environment("push_shortcut", start=(1, 1), goal=(18, 18))
    # A two-stage shortcut: the direct midline is blocked by movable plugs while a rigid shell
    # makes the around-the-outside route substantially longer.
    add_corridor_shell(env, rng, corridor_axis="horizontal", corridor_value=4, start_gate=2, end_gate=10, thickness=1, volume=0.85)
    add_corridor_shell(env, rng, corridor_axis="vertical", corridor_value=10, start_gate=4, end_gate=15, thickness=1, volume=0.85)
    add_corridor_shell(env, rng, corridor_axis="horizontal", corridor_value=15, start_gate=10, end_gate=17, thickness=1, volume=0.85)
    for pos in [(8, 4), (9, 4), (10, 9), (10, 10)]:
        add_light_movable(env, rng, f"shortcut_{pos[0]}_{pos[1]}", pos, geometry="rectangle")
    return env


def build_push_heavy_environment(name: str, seed: int, params: dict) -> Environment:
    builders = {
        "push_wall": build_push_wall,
        "push_chain": build_push_chain,
        "push_corridor": build_push_corridor,
        "push_shortcut": build_push_shortcut,
    }
    env = builders[name](seed)
    env = exp.configure_environment(env.name, seed, params) if False else env
    env.robot.radius = float(params["robot_radius"])
    for obstacle in env.obstacles:
        obstacle.density *= float(params["obstacle_density_scale"])
        obstacle.friction_coefficient *= float(params["friction_scale"])
        exp.scale_obstacle_geometry(obstacle, float(params["obstacle_size_scale"]))
    exp.attach_expanded_blocking(env)
    return env


def run_trial(environment_name: str, seed: int, planner_name: str, params: dict) -> exp.TrialMetrics:
    if planner_name == "beacon_human_like":
        return run_beacon_trial(environment_name, seed, params)

    environment = build_push_heavy_environment(environment_name, seed, params)
    initial_obstacle_count = len(environment.obstacles)
    evaluator = exp.WeightedInteractionEvaluator(environment, energy_weight=params["energy_weight"])
    planner = exp.planner_factory(planner_name, environment)
    executor = exp.InstrumentedExecutor(environment, planner_name, planner, evaluator)
    executor.execute()
    path_length = exp.euclidean_path_length(executor.robot_path)
    total_path_cost = path_length + params["energy_weight"] * executor.push_energy
    return exp.TrialMetrics(
        environment=environment_name,
        seed=seed,
        planner=planner_name,
        factor="push_heavy",
        factor_level="baseline",
        robot_radius=float(params["robot_radius"]),
        obstacle_density_scale=float(params["obstacle_density_scale"]),
        obstacle_size_scale=float(params["obstacle_size_scale"]),
        friction_scale=float(params["friction_scale"]),
        energy_weight=float(params["energy_weight"]),
        obstacle_count_scale=float(params["obstacle_count_scale"]),
        success=bool(executor.reached_goal),
        total_path_cost=round(total_path_cost, 4),
        path_length=round(path_length, 4),
        time_to_goal=max(0, len(executor.robot_path) - 1),
        energy_consumption=round(executor.energy_consumption, 4),
        obstacle_interactions=executor.obstacle_interactions,
        pushed_obstacles=executor.push_actions,
        unique_pushed_obstacles=len(set(executor.pushed_ids)),
        secondary_obstacle_movements=executor.secondary_obstacle_movements,
        moved_obstacles=len(environment.moved_obstacles),
        initial_obstacle_count=initial_obstacle_count,
    )


def run_beacon_trial(environment_name: str, seed: int, params: dict) -> exp.TrialMetrics:
    environment = build_push_heavy_environment(environment_name, seed, params)
    initial_obstacle_count = len(environment.obstacles)
    scene = exp.semantics_env_to_beacon_scene(environment)
    config = exp.PlannerConfig(step=PUSH_HEAVY_BEACON_STEP, robot_radius=float(params["robot_radius"]))
    world = exp.World.from_scene(scene)
    dstar_planner = exp.BeaconDStarLitePlanner(world, config=config)
    push_policy = exp.PushPolicy(
        exp.WeightedBeaconInteractionCost(
            energy_weight=float(params["energy_weight"]),
            push_duration=config.step_duration,
        )
    )
    planner = exp.HumanLikePlanner(dstar_planner, push_policy, world, alpha=min(environment.robot.sensing_radius, PUSH_HEAVY_BEACON_ALPHA))
    env = exp.PyBulletEnv(world)

    executed_trajectory = [[float(world.start[0]), float(world.start[1]), 0.0]]
    cumulative_push_work = 0.0
    cumulative_navigation_work = 0.0
    cumulative_energy = 0.0
    action_log = []
    flowchart_trace = []
    goal_tolerance = max(config.step, config.robot_radius)
    max_steps = max(60, environment.width * environment.height * 5)

    for step_index in range(max_steps):
        robot_position = env.get_robot_position()
        if math.dist(robot_position, world.goal) <= goal_tolerance:
            break
        action = planner.step(robot_position)
        decision_trace = dict(planner.last_decision_trace)
        if action is None:
            flowchart_trace.append(decision_trace)
            break
        execution = env.execute(action) or {}
        if isinstance(action, exp.MoveAction):
            path_segment = action.path_segment or [
                [robot_position[0], robot_position[1], 0.0],
                [action.target[0], action.target[1], 0.0],
            ]
            if executed_trajectory[-1][:2] != path_segment[0][:2]:
                executed_trajectory.append([float(path_segment[0][0]), float(path_segment[0][1]), float(path_segment[0][2])])
            for point in path_segment[1:]:
                executed_trajectory.append([float(point[0]), float(point[1]), float(point[2])])
            move_distance = exp.compute_path_length(path_segment)
            cumulative_navigation_work += push_policy.cost_model.compute_navigation_work(move_distance)
        else:
            cumulative_push_work += float(getattr(action, "estimated_work", 0.0))
        cumulative_energy = cumulative_navigation_work + cumulative_push_work

        updated_obstacle_positions = []
        if execution.get("updated_obstacle_ids"):
            for obstacle in world.obstacles:
                if obstacle.id in execution["updated_obstacle_ids"]:
                    updated_obstacle_positions.append({"obstacle_id": int(obstacle.id)})

        action_log.append(
            {
                "action_type": "push" if isinstance(action, exp.PushAction) else "move",
                "updated_obstacle_positions": updated_obstacle_positions,
                "obstacle_id": int(action.obstacle_id) if isinstance(action, exp.PushAction) else None,
            }
        )
        flowchart_trace.append(decision_trace)

    final_position = env.get_robot_position()
    success = math.dist(final_position, world.goal) <= goal_tolerance
    path_length = exp.compute_path_length(executed_trajectory)
    pushed_entries = [entry for entry in action_log if entry["action_type"] == "push"]
    secondary_movements = sum(max(0, len(entry.get("updated_obstacle_positions", [])) - 1) for entry in pushed_entries)
    unique_moved = {
        update["obstacle_id"]
        for entry in action_log
        for update in entry.get("updated_obstacle_positions", [])
    }
    return exp.TrialMetrics(
        environment=environment_name,
        seed=seed,
        planner="beacon_human_like",
        factor="push_heavy",
        factor_level="baseline",
        robot_radius=float(params["robot_radius"]),
        obstacle_density_scale=float(params["obstacle_density_scale"]),
        obstacle_size_scale=float(params["obstacle_size_scale"]),
        friction_scale=float(params["friction_scale"]),
        energy_weight=float(params["energy_weight"]),
        obstacle_count_scale=float(params["obstacle_count_scale"]),
        success=bool(success),
        total_path_cost=round(path_length + float(params["energy_weight"]) * cumulative_push_work, 4),
        path_length=round(path_length, 4),
        time_to_goal=len(action_log),
        energy_consumption=round(cumulative_energy, 4),
        obstacle_interactions=sum(1 for trace in flowchart_trace if trace.get("blocking_obstacle_id") is not None),
        pushed_obstacles=len(pushed_entries),
        unique_pushed_obstacles=len({entry["obstacle_id"] for entry in pushed_entries if entry["obstacle_id"] is not None}),
        secondary_obstacle_movements=secondary_movements,
        moved_obstacles=len(unique_moved),
        initial_obstacle_count=initial_obstacle_count,
    )


def write_outputs(trials: list[exp.TrialMetrics]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].__dict__.keys()))
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial.__dict__)

    summary_rows, planner_rollup = exp.summarize(trials)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "evaluation": "push_heavy",
        "assumptions": {
            "design": "structured push-heavy benchmark with long detours around rigid barriers",
            "energy_consumption": "navigation work + push energy",
            "total_path_cost": "path_length + energy_weight * push_energy",
            "energy_weight": PUSH_HEAVY_PARAMS["energy_weight"],
            "beacon_push_heavy_step": PUSH_HEAVY_BEACON_STEP,
            "beacon_push_heavy_alpha": PUSH_HEAVY_BEACON_ALPHA,
        },
        "planners": PLANNERS,
        "environments": PUSH_HEAVY_ENVIRONMENTS,
        "seeds": SEEDS,
        "baseline_parameters": PUSH_HEAVY_PARAMS,
        "summary_rows": summary_rows,
        "planner_rollup": planner_rollup,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    trials: list[exp.TrialMetrics] = []
    total = len(PUSH_HEAVY_ENVIRONMENTS) * len(SEEDS) * len(PLANNERS)
    run_index = 0
    for env_name in PUSH_HEAVY_ENVIRONMENTS:
        for seed in SEEDS:
            for planner_name in PLANNERS:
                run_index += 1
                print(f"[{run_index}/{total}] env={env_name} seed={seed} planner={planner_name}", flush=True)
                trials.append(run_trial(env_name, seed, planner_name, dict(PUSH_HEAVY_PARAMS)))
    write_outputs(trials)
    print(f"Saved raw results to {RAW_CSV}")
    print(f"Saved summary CSV to {SUMMARY_CSV}")
    print(f"Saved summary JSON to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
