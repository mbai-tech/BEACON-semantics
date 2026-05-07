#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BEACON_ROOT = REPO_ROOT / "BEACON"
if str(BEACON_ROOT) not in sys.path:
    sys.path.insert(0, str(BEACON_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "data" / ".matplotlib"))

from BEACON.render import render_scene_comparison_grid  # type: ignore
from experiments.generate_example_run_pngs import (  # type: ignore
    BASE_PARAMETERS,
    PLANNER_LABELS,
    PLANNERS,
    run_beacon_planner,
    run_standard_planner,
)
from experiments.run_semantics_planner_experiments import ENVIRONMENTS  # type: ignore


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "examples" / "family_comparison_panels"


def base_seed_for_environment(environment_name: str) -> int:
    return 11 + ENVIRONMENTS.index(environment_name)


def run_all_planners(environment_name: str, seed: int, params: dict) -> dict:
    planner_runs = {
        "dstar_lite": run_standard_planner(environment_name, seed, "dstar_lite", params),
        "bug1": run_standard_planner(environment_name, seed, "bug1", params),
        "bug2": run_standard_planner(environment_name, seed, "bug2", params),
        "beacon_human_like": run_beacon_planner(environment_name, seed, params),
    }
    return planner_runs


def scene_is_eligible(planner_runs: dict, require_all_success: bool) -> bool:
    return all(run.success for run in planner_runs.values()) if require_all_success else True


def find_scenes_for_environment(
    environment_name: str,
    target_count: int,
    max_seed_offset: int,
    require_all_success: bool,
    params: dict,
) -> list[tuple[int, dict]]:
    selected = []
    start_seed = base_seed_for_environment(environment_name)

    for offset in range(max_seed_offset + 1):
        seed = start_seed + offset
        planner_runs = run_all_planners(environment_name, seed, params)
        if not scene_is_eligible(planner_runs, require_all_success):
            continue
        selected.append((seed, planner_runs))
        if len(selected) >= target_count:
            break

    return selected


def panel_payload(planner_runs: dict) -> dict:
    payload = {}
    for planner_name in PLANNERS:
        run = planner_runs[planner_name]
        payload[planner_name] = {
            "scene": run.scene,
            "path_points": run.path_points,
            "title_suffix": (
                f"({PLANNER_LABELS[planner_name]} | "
                f"{'success' if run.success else 'failed'}, "
                f"len={run.path_length:.1f}, steps={run.steps})"
            ),
        }
    return payload


def render_environment_panels(
    environments: list[str],
    scenes_per_environment: int,
    max_seed_offset: int,
    require_all_success: bool,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = dict(BASE_PARAMETERS)
    manifest = []

    for environment_name in environments:
        selected = find_scenes_for_environment(
            environment_name,
            target_count=scenes_per_environment,
            max_seed_offset=max_seed_offset,
            require_all_success=require_all_success,
            params=params,
        )
        if len(selected) < scenes_per_environment:
            raise SystemExit(
                f"Only found {len(selected)} scene(s) for {environment_name}. "
                f"Try increasing --max-seed-offset or relaxing success requirements."
            )

        for scene_index, (seed, planner_runs) in enumerate(selected, start=1):
            filename = f"{environment_name}_seed{seed:02d}_comparison.png"
            output_path = output_dir / filename
            render_scene_comparison_grid(
                panel_payload(planner_runs),
                output_path,
                title=f"{environment_name} | seed {seed:02d} | scene {scene_index}/{scenes_per_environment}",
                planner_order=PLANNERS,
                columns=2,
            )
            manifest.append(
                {
                    "environment": environment_name,
                    "seed": seed,
                    "output": str(output_path),
                    "planners": {
                        planner_name: {
                            "success": planner_runs[planner_name].success,
                            "path_length": round(planner_runs[planner_name].path_length, 4),
                            "steps": planner_runs[planner_name].steps,
                        }
                        for planner_name in PLANNERS
                    },
                }
            )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all planners on two scenes per family and save side-by-side comparison PNGs."
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=ENVIRONMENTS,
        help="Environment families to process.",
    )
    parser.add_argument(
        "--scenes-per-environment",
        type=int,
        default=2,
        help="How many scenes to render per family.",
    )
    parser.add_argument(
        "--max-seed-offset",
        type=int,
        default=8,
        help="Search this many consecutive seeds starting from the experiment seed for each family.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep scenes even if one or more planners fail.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where comparison PNGs and the manifest JSON are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest = render_environment_panels(
        environments=args.environments,
        scenes_per_environment=args.scenes_per_environment,
        max_seed_offset=args.max_seed_offset,
        require_all_success=not args.allow_failures,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} comparison panel(s) to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
