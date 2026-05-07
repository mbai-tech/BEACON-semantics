#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path("/Users/ishita/Documents/GitHub/BEACON-semantics")
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_semantics_planner_experiments as legacy  # noqa: E402

SEMANTICS_ROOT = Path("/Users/ishita/Documents/GitHub/Semantics")
if str(SEMANTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTICS_ROOT))

from robot_push_planner.planner.dstar_lite import DStarLitePlanner as SemanticsDStarLitePlanner  # noqa: E402


SCENE_SWEEP_ENVIRONMENTS = [
    "sparse",
    "cluttered",
    "collision_required",
    "collision_shortcut",
]
PLANNER_SWEEP_ENVIRONMENT = "cluttered"
SCENE_SWEEP_EPISODES_PER_ENV = 100
PLANNER_SWEEP_EPISODES = 100

ALL_PLANNERS = [
    "bug1_geometric",
    "bug2_geometric",
    "dstar_lite_geometric",
    "baseline_semantic",
    "beacon_human_like",
]
PLANNER_LABELS = {
    "bug1_geometric": "Bug",
    "bug2_geometric": "Bug2",
    "dstar_lite_geometric": "D* Lite",
    "baseline_semantic": "Baseline Sem.",
    "beacon_human_like": "BEACON",
}

DEFAULT_PARAMETERS = {
    "robot_radius": 0.12,
    "obstacle_density_scale": 1.0,
    "obstacle_size_scale": 1.0,
    "friction_scale": 1.0,
    "energy_weight": 1.0,
    "obstacle_count_scale": 1.0,
}

SCENE_SWEEP_FACTORS = {
    "obstacle_density_scale": [0.7, 1.0, 1.4],
    "obstacle_size_scale": [0.75, 1.0, 1.35],
    "friction_scale": [0.7, 1.0, 1.3],
    "obstacle_count_scale": [0.7, 1.0, 1.3],
}

PLANNER_SWEEP_FACTORS = {
    "robot_radius": [0.12, 0.15, 0.45],
    "energy_weight": [0.5, 1.0, 1.5],
}


@dataclass
class TrialResult:
    sweep_group: str
    environment: str
    seed: int
    planner: str
    planner_label: str
    factor: str
    factor_level: str
    robot_radius: float
    obstacle_density_scale: float
    obstacle_size_scale: float
    friction_scale: float
    energy_weight: float
    obstacle_count_scale: float
    success: bool
    success_rate_value: float
    energy_consumption: float
    path_length: float
    time_to_goal: float


class GeometricExecutor:
    def __init__(self, environment, planner, max_steps=None):
        self.environment = environment
        self.planner = planner
        self.max_steps = max_steps or max(50, environment.width * environment.height * 4)
        self.robot_path = [environment.start]
        self.reached_goal = False
        self.navigation_energy = 0.0

    def execute(self):
        for _ in range(self.max_steps):
            current = self.environment.robot.position
            if current == self.environment.goal:
                self.reached_goal = True
                return

            path = self.planner.plan_path(current, self.environment.goal)
            if not path or len(path) < 2:
                self.reached_goal = False
                return

            nxt = path[1]
            self.environment.robot.position = nxt
            self.robot_path.append(nxt)
            self.navigation_energy += self.planner.cost(current, nxt)

        self.reached_goal = self.environment.robot.position == self.environment.goal


def make_seeds(start_seed: int, episodes: int) -> list[int]:
    return [start_seed + index for index in range(episodes)]


def make_params(factor: str, value: float) -> dict:
    params = dict(DEFAULT_PARAMETERS)
    params[factor] = value
    return params


def run_geometric_trial(environment_name: str, seed: int, planner_name: str, params: dict) -> TrialResult:
    environment = legacy.configure_environment(environment_name, seed, params)
    if planner_name == "bug1_geometric":
        planner = legacy.BugPlanner(environment, mode="bug1")
    elif planner_name == "bug2_geometric":
        planner = legacy.BugPlanner(environment, mode="bug2")
    elif planner_name == "dstar_lite_geometric":
        planner = SemanticsDStarLitePlanner(environment)
    else:
        raise ValueError(f"Unknown geometric planner: {planner_name}")

    executor = GeometricExecutor(environment, planner)
    executor.execute()
    path_length = legacy.euclidean_path_length(executor.robot_path)
    return TrialResult(
        sweep_group="",
        environment=environment_name,
        seed=seed,
        planner=planner_name,
        planner_label=PLANNER_LABELS[planner_name],
        factor="",
        factor_level="",
        robot_radius=float(params["robot_radius"]),
        obstacle_density_scale=float(params["obstacle_density_scale"]),
        obstacle_size_scale=float(params["obstacle_size_scale"]),
        friction_scale=float(params["friction_scale"]),
        energy_weight=float(params["energy_weight"]),
        obstacle_count_scale=float(params["obstacle_count_scale"]),
        success=bool(executor.reached_goal),
        success_rate_value=1.0 if executor.reached_goal else 0.0,
        energy_consumption=round(executor.navigation_energy, 4),
        path_length=round(path_length, 4),
        time_to_goal=max(0.0, float(len(executor.robot_path) - 1)),
    )


def run_semantic_baseline_trial(environment_name: str, seed: int, params: dict) -> TrialResult:
    environment = legacy.configure_environment(environment_name, seed, params)
    evaluator = legacy.WeightedInteractionEvaluator(environment, energy_weight=params["energy_weight"])
    planner = SemanticsDStarLitePlanner(environment)
    executor = legacy.InstrumentedExecutor(environment, "baseline_semantic", planner, evaluator)
    executor.execute()
    path_length = legacy.euclidean_path_length(executor.robot_path)
    return TrialResult(
        sweep_group="",
        environment=environment_name,
        seed=seed,
        planner="baseline_semantic",
        planner_label=PLANNER_LABELS["baseline_semantic"],
        factor="",
        factor_level="",
        robot_radius=float(params["robot_radius"]),
        obstacle_density_scale=float(params["obstacle_density_scale"]),
        obstacle_size_scale=float(params["obstacle_size_scale"]),
        friction_scale=float(params["friction_scale"]),
        energy_weight=float(params["energy_weight"]),
        obstacle_count_scale=float(params["obstacle_count_scale"]),
        success=bool(executor.reached_goal),
        success_rate_value=1.0 if executor.reached_goal else 0.0,
        energy_consumption=round(executor.energy_consumption, 4),
        path_length=round(path_length, 4),
        time_to_goal=max(0.0, float(len(executor.robot_path) - 1)),
    )


def run_beacon_trial(environment_name: str, seed: int, params: dict) -> TrialResult:
    trial = legacy.run_beacon_human_like_trial(environment_name, seed, "", "", params)
    return TrialResult(
        sweep_group="",
        environment=environment_name,
        seed=seed,
        planner="beacon_human_like",
        planner_label=PLANNER_LABELS["beacon_human_like"],
        factor="",
        factor_level="",
        robot_radius=float(params["robot_radius"]),
        obstacle_density_scale=float(params["obstacle_density_scale"]),
        obstacle_size_scale=float(params["obstacle_size_scale"]),
        friction_scale=float(params["friction_scale"]),
        energy_weight=float(params["energy_weight"]),
        obstacle_count_scale=float(params["obstacle_count_scale"]),
        success=bool(trial.success),
        success_rate_value=1.0 if trial.success else 0.0,
        energy_consumption=round(trial.energy_consumption, 4),
        path_length=round(trial.path_length, 4),
        time_to_goal=float(trial.time_to_goal),
    )


def run_case_dict(case: dict) -> TrialResult:
    params = make_params(case["factor"], case["value"])
    planner_name = case["planner"]
    if planner_name in {"bug1_geometric", "bug2_geometric", "dstar_lite_geometric"}:
        result = run_geometric_trial(case["environment"], case["seed"], planner_name, params)
    elif planner_name == "baseline_semantic":
        result = run_semantic_baseline_trial(case["environment"], case["seed"], params)
    elif planner_name == "beacon_human_like":
        result = run_beacon_trial(case["environment"], case["seed"], params)
    else:
        raise ValueError(f"Unknown planner: {planner_name}")

    result.sweep_group = case["sweep_group"]
    result.factor = case["factor"]
    result.factor_level = str(case["value"])
    return result


def build_scene_sweep_cases(scene_episodes_per_env: int, scene_start_seed: int) -> list[dict]:
    cases = []
    for factor, values in SCENE_SWEEP_FACTORS.items():
        for value in values:
            for environment_name in SCENE_SWEEP_ENVIRONMENTS:
                for seed in make_seeds(scene_start_seed, scene_episodes_per_env):
                    for planner_name in ALL_PLANNERS:
                        cases.append(
                            {
                                "sweep_group": "scene_generation",
                                "environment": environment_name,
                                "seed": seed,
                                "planner": planner_name,
                                "factor": factor,
                                "value": value,
                            }
                        )
    return cases


def build_planner_sweep_cases(planner_episodes: int, planner_start_seed: int) -> list[dict]:
    cases = []
    for factor, values in PLANNER_SWEEP_FACTORS.items():
        for value in values:
            for seed in make_seeds(planner_start_seed, planner_episodes):
                for planner_name in ["baseline_semantic", "beacon_human_like"]:
                    cases.append(
                        {
                            "sweep_group": "planner_hyperparameter",
                            "environment": PLANNER_SWEEP_ENVIRONMENT,
                            "seed": seed,
                            "planner": planner_name,
                            "factor": factor,
                            "value": value,
                        }
                    )
    return cases


def summarize(results: list[TrialResult]) -> list[dict]:
    grouped = {}
    for result in results:
        key = (
            result.sweep_group,
            result.factor,
            result.factor_level,
            result.planner,
            result.planner_label,
        )
        grouped.setdefault(key, []).append(result)

    rows = []
    for key in sorted(grouped):
        sweep_group, factor, factor_level, planner, planner_label = key
        items = grouped[key]
        rows.append(
            {
                "sweep_group": sweep_group,
                "factor": factor,
                "factor_level": factor_level,
                "planner": planner,
                "planner_label": planner_label,
                "episodes": len(items),
                "success_rate_percent": round(
                    100.0 * sum(item.success_rate_value for item in items) / len(items),
                    2,
                ),
                "mean_energy": round(
                    sum(item.energy_consumption for item in items) / len(items),
                    4,
                ),
                "mean_path_length": round(
                    sum(item.path_length for item in items) / len(items),
                    4,
                ),
                "mean_time_to_goal": round(
                    sum(item.time_to_goal for item in items) / len(items),
                    4,
                ),
            }
        )
    return rows


def build_table_payload(summary_rows: list[dict]) -> dict:
    payload = {
        "scene_generation": {},
        "planner_hyperparameter": {},
    }
    for row in summary_rows:
        group = row["sweep_group"]
        factor = row["factor"]
        planner = row["planner_label"]
        payload[group].setdefault(factor, {})
        payload[group][factor].setdefault(planner, {})
        payload[group][factor][planner][row["factor_level"]] = {
            "SR": row["success_rate_percent"],
            "E": row["mean_energy"],
        }
    return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, raw_results: list[TrialResult], summary_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "appendix_hyperparameter_sweep_raw.csv"
    summary_path = output_dir / "appendix_hyperparameter_sweep_summary.csv"
    json_path = output_dir / "appendix_hyperparameter_sweep_tables.json"
    metadata_path = output_dir / "appendix_hyperparameter_sweep_metadata.json"

    write_csv(raw_path, [asdict(item) for item in raw_results])
    write_csv(summary_path, summary_rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(build_table_payload(summary_rows), handle, indent=2)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene_sweep_environments": SCENE_SWEEP_ENVIRONMENTS,
                "scene_sweep_episodes_per_env": SCENE_SWEEP_EPISODES_PER_ENV,
                "planner_sweep_environment": PLANNER_SWEEP_ENVIRONMENT,
                "planner_sweep_episodes": PLANNER_SWEEP_EPISODES,
                "default_parameters": DEFAULT_PARAMETERS,
                "scene_sweep_factors": SCENE_SWEEP_FACTORS,
                "planner_sweep_factors": PLANNER_SWEEP_FACTORS,
                "planners": PLANNER_LABELS,
                "notes": [
                    "Bug and Bug2 are geometric non-contact planners.",
                    "D* Lite is geometric non-contact D* Lite.",
                    "Baseline Sem. is the Semantics push-aware D* Lite planner.",
                    "BEACON is the current human-like BEACON planner in this repo.",
                    "Scene-generation sweep explicitly uses 4 environments x 100 seeds = 400 episodes per factor level.",
                    "Planner sweep explicitly uses the cluttered environment with 100 seeds.",
                ],
            },
            handle,
            indent=2,
        )


def run_cases(cases: list[dict], workers: int) -> list[TrialResult]:
    results: list[TrialResult] = []
    if workers <= 1:
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{total}] {case['sweep_group']} env={case['environment']} "
                f"planner={case['planner']} factor={case['factor']} value={case['value']} seed={case['seed']}",
                flush=True,
            )
            results.append(run_case_dict(case))
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_case = {pool.submit(run_case_dict, case): case for case in cases}
        completed = 0
        total = len(cases)
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            completed += 1
            print(
                f"[{completed}/{total}] done env={case['environment']} planner={case['planner']} "
                f"factor={case['factor']} value={case['value']} seed={case['seed']}",
                flush=True,
            )
            results.append(future.result())
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the appendix hyperparameter benchmark with explicit planner definitions.",
    )
    parser.add_argument(
        "--mode",
        choices=["sanity", "scene", "planner", "full"],
        default="sanity",
        help="Which subset of the appendix benchmark to run.",
    )
    parser.add_argument("--scene-episodes-per-env", type=int, default=SCENE_SWEEP_EPISODES_PER_ENV)
    parser.add_argument("--planner-episodes", type=int, default=PLANNER_SWEEP_EPISODES)
    parser.add_argument("--scene-start-seed", type=int, default=11)
    parser.add_argument("--planner-start-seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for independent trial runs.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "experiments" / "appendix_benchmark"),
        help="Directory for raw and summarized appendix sweep outputs.",
    )
    return parser.parse_args()


def build_cases(args) -> list[dict]:
    if args.mode == "sanity":
        return [
            {
                "sweep_group": "scene_generation",
                "environment": "cluttered",
                "seed": 11,
                "planner": planner_name,
                "factor": "obstacle_density_scale",
                "value": 1.0,
            }
            for planner_name in ALL_PLANNERS
        ] + [
            {
                "sweep_group": "planner_hyperparameter",
                "environment": "cluttered",
                "seed": 11,
                "planner": planner_name,
                "factor": "robot_radius",
                "value": 0.15,
            }
            for planner_name in ["baseline_semantic", "beacon_human_like"]
        ]
    if args.mode == "scene":
        return build_scene_sweep_cases(args.scene_episodes_per_env, args.scene_start_seed)
    if args.mode == "planner":
        return build_planner_sweep_cases(args.planner_episodes, args.planner_start_seed)
    return build_scene_sweep_cases(args.scene_episodes_per_env, args.scene_start_seed) + build_planner_sweep_cases(
        args.planner_episodes,
        args.planner_start_seed,
    )


def main():
    args = parse_args()
    cases = build_cases(args)
    results = run_cases(cases, workers=max(1, args.workers))
    summary_rows = summarize(results)
    write_outputs(Path(args.output_dir), results, summary_rows)
    print(f"Saved appendix benchmark outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
