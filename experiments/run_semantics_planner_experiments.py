#!/usr/bin/env python3

from __future__ import annotations

import copy
import csv
import heapq
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType


SEMANTICS_ROOT = Path("/Users/ishita/Documents/GitHub/Semantics")
if str(SEMANTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTICS_ROOT))

BEACON_ROOT = Path("/Users/ishita/Documents/GitHub/BEACON-semantics/BEACON")
if str(BEACON_ROOT) not in sys.path:
    sys.path.insert(0, str(BEACON_ROOT))

from robot_push_planner.core.environment import Environment  # type: ignore
from robot_push_planner.core.obstacle import Obstacle  # type: ignore
from robot_push_planner.environments import build_environment  # type: ignore
from robot_push_planner.environments.base_env import _make_obstacle  # type: ignore
from robot_push_planner.interaction.evaluator import InteractionEvaluator  # type: ignore
from robot_push_planner.physics.push_physics import PushPhysics  # type: ignore
from robot_push_planner.planner.dstar_lite import DStarLitePlanner  # type: ignore
from semantic_dstar_lite import PlannerConfig, compute_path_length  # type: ignore
from environment.world import World  # type: ignore
from interaction.interaction_cost import InteractionCost  # type: ignore
from interaction.push_policy import PushPolicy  # type: ignore
from planning.dstar_lite import DStarLitePlanner as BeaconDStarLitePlanner  # type: ignore
from planning.planner import HumanLikePlanner  # type: ignore
from simulation.pybullet_env import PyBulletEnv  # type: ignore
from core.state import MoveAction, PushAction  # type: ignore


OUTPUT_DIR = Path("/Users/ishita/Documents/GitHub/BEACON-semantics/data/experiments/semantics")
RAW_CSV = OUTPUT_DIR / "planner_sweep_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "planner_sweep_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "planner_sweep_summary.json"

ENVIRONMENTS = [
    "sparse",
    "collision_required",
    "collision_shortcut",
    "open_room",
    "corridor",
    "cluttered",
    "dense_barrier",
    "chain_reaction",
    "mixed_materials",
]

PLANNERS = ["dstar_lite", "bug1", "bug2", "beacon_human_like"]
BEACON_TUNED_STEP = 0.5
BEACON_TUNED_ALPHA = 2.5

BASE_PARAMETERS = {
    "robot_radius": 0.25,
    "obstacle_density_scale": 1.0,
    "obstacle_size_scale": 1.0,
    "friction_scale": 1.0,
    "energy_weight": 1.0,
    "obstacle_count_scale": 1.0,
}

FACTOR_LEVELS = {
    "robot_radius": [0.15, 0.25, 0.45],
    "obstacle_density_scale": [0.7, 1.0, 1.4],
    "obstacle_size_scale": [0.75, 1.0, 1.35],
    "friction_scale": [0.7, 1.0, 1.3],
    "energy_weight": [0.5, 1.0, 1.5],
    "obstacle_count_scale": [0.7, 1.0, 1.3],
}


def planner_step_cost(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0


def euclidean_path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def obstacle_radius(obstacle: Obstacle) -> float:
    if obstacle.vertices:
        cx = obstacle.position[0] + 0.5
        cy = obstacle.position[1] + 0.5
        return max(math.dist((cx, cy), tuple(vertex)) for vertex in obstacle.vertices)
    return max(0.35, math.sqrt(max(obstacle.volume, 0.01) / math.pi) * 0.35)


def attach_expanded_blocking(environment: Environment) -> None:
    def is_blocked(self: Environment, position: tuple[int, int]) -> bool:
        px = position[0] + 0.5
        py = position[1] + 0.5
        for obstacle in self.obstacles:
            ox = obstacle.position[0] + 0.5
            oy = obstacle.position[1] + 0.5
            clearance = obstacle_radius(obstacle) + float(self.robot.radius)
            if math.dist((px, py), (ox, oy)) <= clearance:
                return True
        return False

    environment.is_blocked = MethodType(is_blocked, environment)


def scale_obstacle_geometry(obstacle: Obstacle, scale: float) -> None:
    obstacle.volume *= scale
    if obstacle.vertices:
        cx = obstacle.position[0] + 0.5
        cy = obstacle.position[1] + 0.5
        scaled = []
        for vx, vy in obstacle.vertices:
            dx = vx - cx
            dy = vy - cy
            scaled.append((cx + dx * scale, cy + dy * scale))
        obstacle.vertices = scaled


def free_positions(environment: Environment) -> list[tuple[int, int]]:
    occupied = {environment.start, environment.goal}
    occupied.update(obstacle.position for obstacle in environment.obstacles)
    return [
        (x, y)
        for x in range(1, environment.width - 1)
        for y in range(1, environment.height - 1)
        if (x, y) not in occupied
    ]


def resize_obstacle_count(environment: Environment, scale: float) -> None:
    current = len(environment.obstacles)
    target = max(1, int(round(current * scale)))
    rng = random.Random((environment.seed or 0) + 991)

    if target < current:
        keep = set(rng.sample(range(current), target))
        environment.obstacles = [obstacle for index, obstacle in enumerate(environment.obstacles) if index in keep]
        return

    if target == current:
        return

    available = free_positions(environment)
    rng.shuffle(available)
    next_index = current + 1
    while len(environment.obstacles) < target and available:
        position = available.pop()
        semantic_class = rng.choices(
            ["safe", "movable", "fragile"],
            weights=[0.25, 0.5, 0.25],
            k=1,
        )[0]
        obstacle = _make_obstacle(rng, f"extra_{next_index}", position, semantic_class)
        environment.add_obstacle(obstacle)
        next_index += 1


def configure_environment(name: str, seed: int, params: dict) -> Environment:
    environment = build_environment(name, seed=seed)
    environment = copy.deepcopy(environment)
    environment.robot.radius = float(params["robot_radius"])

    for obstacle in environment.obstacles:
        obstacle.density *= float(params["obstacle_density_scale"])
        obstacle.friction_coefficient *= float(params["friction_scale"])
        scale_obstacle_geometry(obstacle, float(params["obstacle_size_scale"]))

    resize_obstacle_count(environment, float(params["obstacle_count_scale"]))
    attach_expanded_blocking(environment)
    return environment


def semantics_env_to_beacon_scene(environment: Environment) -> dict:
    obstacles = []
    for index, obstacle in enumerate(environment.obstacles):
        vertices = [
            [float(vertex[0]), float(vertex[1])]
            for vertex in (obstacle.vertices or [])
        ]
        if not vertices:
            cx = obstacle.position[0] + 0.5
            cy = obstacle.position[1] + 0.5
            vertices = [
                [cx - 0.25, cy - 0.25],
                [cx + 0.25, cy - 0.25],
                [cx + 0.25, cy + 0.25],
                [cx - 0.25, cy + 0.25],
            ]

        mass = float(obstacle.density) * float(obstacle.volume)
        label = obstacle.semantic_class
        if label not in {"safe", "movable", "fragile"}:
            label = "movable" if obstacle.pushable else "safe"

        obstacles.append(
            {
                "id": int(index),
                "shape_type": obstacle.geometry,
                "class_true": label,
                "radius": 0.0,
                "center": [float(obstacle.position[0]) + 0.5, float(obstacle.position[1]) + 0.5],
                "vertices": vertices,
                "semantic_probs": {label: 1.0},
                "material": {
                    "name": label,
                    "density": float(obstacle.density),
                    "mass": mass,
                    "friction": float(obstacle.friction_coefficient),
                    "fragility": 0.9 if label == "fragile" else 0.2,
                    "pushable": bool(obstacle.pushable),
                },
                "movable": bool(obstacle.pushable),
                "previous_vertices": [],
            }
        )

    return {
        "family": environment.name,
        "seed": environment.seed,
        "workspace": [0.0, float(environment.width), 0.0, float(environment.height)],
        "start": [float(environment.start[0]) + 0.5, float(environment.start[1]) + 0.5, 0.0],
        "goal": [float(environment.goal[0]) + 0.5, float(environment.goal[1]) + 0.5, 0.0],
        "obstacles": obstacles,
    }


class WeightedInteractionEvaluator(InteractionEvaluator):
    def __init__(self, environment, energy_weight: float = 1.0):
        super().__init__(environment)
        self.energy_weight = float(energy_weight)

    def decide_push_or_detour(self, robot, obstacle, push_distance, detour_cost):
        push_data = self.compute_push_cost(obstacle, push_distance)
        if not self.can_robot_push(robot, obstacle):
            return "detour", push_data
        weighted_energy = self.energy_weight * push_data["energy"]
        if weighted_energy < detour_cost:
            return "push", push_data
        return "detour", push_data


class WeightedBeaconInteractionCost(InteractionCost):
    def __init__(self, energy_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.energy_weight = float(energy_weight)

    def compute_push_cost(self, obstacle, push_distance, **kwargs):
        return self.compute_push_work(obstacle, push_distance, **kwargs)

    def compute_push_work(self, obstacle, push_distance, **kwargs):
        physical_work = super().compute_push_work(obstacle, push_distance, **kwargs)
        return self.energy_weight * physical_work


class AStarPlanner:
    def __init__(self, environment: Environment):
        self.environment = environment

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.dist(a, b)

    def cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        if not self.environment.in_bounds(a) or not self.environment.in_bounds(b):
            return float("inf")
        if self.environment.is_blocked(a) or self.environment.is_blocked(b):
            return float("inf")
        return planner_step_cost(a, b)

    def plan_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        open_set: list[tuple[float, float, tuple[int, int]]] = [(self.heuristic(start, goal), 0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        g_score: dict[tuple[int, int], float] = defaultdict(lambda: float("inf"))
        g_score[start] = 0.0

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current_g > g_score[current]:
                continue

            if current == goal:
                path = [current]
                while came_from[current] is not None:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.environment.neighbors8(current):
                step_cost = self.cost(current, neighbor)
                if math.isinf(step_cost):
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None


def line_clear(environment: Environment, a: tuple[int, int], b: tuple[int, int]) -> bool:
    samples = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 4 + 1
    for i in range(samples + 1):
        t = i / max(1, samples)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        cell = (int(round(x)), int(round(y)))
        if environment.in_bounds(cell) and environment.is_blocked(cell):
            return False
    return True


def grid_heading_candidates(current: tuple[int, int], previous: tuple[int, int] | None) -> list[tuple[int, int]]:
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]
    if previous is None or previous == current:
        return directions

    heading = (current[0] - previous[0], current[1] - previous[1])
    if heading not in directions:
        return directions
    idx = directions.index(heading)
    order = [directions[(idx + offset) % len(directions)] for offset in (-2, -1, 0, 1, 2, 3, 4, 5)]
    return order


class BugPlanner:
    def __init__(self, environment: Environment, mode: str):
        self.environment = environment
        self.mode = mode

    def cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return planner_step_cost(a, b)

    def _goal_step(self, current: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int] | None:
        candidates = [
            neighbor
            for neighbor in self.environment.neighbors8(current)
            if not self.environment.is_blocked(neighbor)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda node: (math.dist(node, goal), self.cost(current, node)))

    def _boundary_step(
        self,
        current: tuple[int, int],
        previous: tuple[int, int] | None,
        goal: tuple[int, int],
    ) -> tuple[int, int] | None:
        candidates = []
        for dx, dy in grid_heading_candidates(current, previous):
            nxt = (current[0] + dx, current[1] + dy)
            if not self.environment.in_bounds(nxt) or self.environment.is_blocked(nxt):
                continue
            neighbor_blocks = sum(
                1
                for near in self.environment.neighbors8(nxt)
                if self.environment.is_blocked(near)
            )
            candidates.append((neighbor_blocks, math.dist(nxt, goal), nxt))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _on_mline(self, current: tuple[int, int], start: tuple[int, int], goal: tuple[int, int], threshold: float = 0.75) -> bool:
        sx, sy = start
        gx, gy = goal
        cx, cy = current
        dx = gx - sx
        dy = gy - sy
        denom = dx * dx + dy * dy
        if denom == 0:
            return False
        t = ((cx - sx) * dx + (cy - sy) * dy) / denom
        if not (0.0 <= t <= 1.0):
            return False
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        return math.dist((cx, cy), (proj_x, proj_y)) <= threshold

    def plan_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        current = start
        previous = None
        path = [current]
        follow_boundary = False
        hit_point = None
        hit_distance = float("inf")
        max_steps = self.environment.width * self.environment.height * 8

        for _ in range(max_steps):
            if current == goal:
                return path

            if not follow_boundary and line_clear(self.environment, current, goal):
                nxt = self._goal_step(current, goal)
                if nxt is None:
                    return None
                current, previous = nxt, current
                path.append(current)
                continue

            if not follow_boundary:
                follow_boundary = True
                hit_point = current
                hit_distance = math.dist(current, goal)

            if follow_boundary:
                if self.mode == "bug1":
                    moved_away = hit_point is None or math.dist(current, hit_point) > 1.0
                    if moved_away and line_clear(self.environment, current, goal):
                        follow_boundary = False
                        continue
                else:
                    moved_away = hit_point is None or math.dist(current, hit_point) > 2.0
                    if moved_away and self._on_mline(current, start, goal) and math.dist(current, goal) < hit_distance - 0.5:
                        follow_boundary = False
                        continue

                nxt = self._boundary_step(current, previous, goal)
                if nxt is None or nxt in path[-8:]:
                    return None
                current, previous = nxt, current
                path.append(current)

        return None


@dataclass
class TrialMetrics:
    environment: str
    seed: int
    planner: str
    factor: str
    factor_level: str
    robot_radius: float
    obstacle_density_scale: float
    obstacle_size_scale: float
    friction_scale: float
    energy_weight: float
    obstacle_count_scale: float
    success: bool
    total_path_cost: float
    path_length: float
    time_to_goal: int
    energy_consumption: float
    obstacle_interactions: int
    pushed_obstacles: int
    unique_pushed_obstacles: int
    secondary_obstacle_movements: int
    moved_obstacles: int
    initial_obstacle_count: int


class InstrumentedExecutor:
    def __init__(self, environment: Environment, planner_name: str, planner, evaluator: WeightedInteractionEvaluator, default_push_distance: float = 1.0, max_steps: int = 250):
        self.environment = environment
        self.planner_name = planner_name
        self.planner = planner
        self.evaluator = evaluator
        self.default_push_distance = default_push_distance
        self.max_steps = max_steps
        self.robot_path = [environment.start]
        self.reached_goal = False
        self.path_cost = 0.0
        self.energy_consumption = 0.0
        self.navigation_energy = 0.0
        self.push_energy = 0.0
        self.obstacle_interactions = 0
        self.push_actions = 0
        self.pushed_ids: list[str] = []
        self.secondary_obstacle_movements = 0

    def move_robot(self, next_position: tuple[int, int]) -> None:
        current = self.environment.robot.position
        self.environment.robot.position = next_position
        self.robot_path.append(next_position)
        step_cost = self.planner.cost(current, next_position)
        self.path_cost += step_cost
        self.navigation_energy += step_cost
        self.energy_consumption = self.navigation_energy + self.push_energy

    def _step_direction(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        dx = 0 if b[0] == a[0] else (1 if b[0] > a[0] else -1)
        dy = 0 if b[1] == a[1] else (1 if b[1] > a[1] else -1)
        return (dx, dy)

    def _path_cost(self, path: list[tuple[int, int]] | None) -> float:
        if not path or len(path) < 2:
            return 0.0
        return sum(self.planner.cost(a, b) for a, b in zip(path, path[1:]))

    def _simulate_push_target(
        self,
        obstacle: Obstacle,
        direction: tuple[int, int],
        distance: float,
        occupied: dict[str, tuple[int, int]],
    ) -> tuple[tuple[int, int] | None, int]:
        steps = max(1, int(round(distance)))
        current = occupied[obstacle.id]
        secondary_moves = 0
        for _ in range(steps):
            nxt = (current[0] + direction[0], current[1] + direction[1])
            if not self.environment.in_bounds(nxt):
                return None, secondary_moves
            blocker = None
            for candidate in self.environment.obstacles:
                if candidate.id != obstacle.id and occupied.get(candidate.id) == nxt:
                    blocker = candidate
                    break
            if blocker is not None:
                moved_target, nested_secondary = self._simulate_push_target(
                    blocker,
                    direction,
                    1.0,
                    occupied,
                )
                secondary_moves += nested_secondary
                if moved_target is None:
                    return None, secondary_moves
                occupied[blocker.id] = moved_target
                secondary_moves += 1
            current = nxt
        return current, secondary_moves

    def push_obstacle(self, obstacle: Obstacle, direction: tuple[int, int], distance: float, primary: bool) -> tuple[bool, int]:
        occupied = {candidate.id: candidate.position for candidate in self.environment.obstacles}
        new_position, secondary_moves = self._simulate_push_target(obstacle, direction, distance, occupied)
        if new_position is None:
            return False, secondary_moves
        original_positions = {candidate.id: candidate.position for candidate in self.environment.obstacles}
        for candidate in self.environment.obstacles:
            new_candidate_position = occupied.get(candidate.id, candidate.position)
            if new_candidate_position != original_positions[candidate.id]:
                self.environment.update_obstacle_position(candidate.id, new_candidate_position)
        self.environment.update_obstacle_position(obstacle.id, new_position)
        if primary:
            self.push_actions += 1
            self.pushed_ids.append(obstacle.id)
            self.push_energy += PushPhysics.push_energy(obstacle, distance)
            self.energy_consumption = self.navigation_energy + self.push_energy
            self.secondary_obstacle_movements += secondary_moves
        return True, secondary_moves

    def execute(self) -> None:
        for _ in range(self.max_steps):
            if self.environment.robot.position == self.environment.goal:
                self.reached_goal = True
                return

            current = self.environment.robot.position
            obstacle_free_path = self.planner.plan_path(current, self.environment.goal)
            if not obstacle_free_path:
                self.reached_goal = False
                return

            if len(obstacle_free_path) < 2:
                self.reached_goal = True
                return

            next_position = obstacle_free_path[1]
            self.move_robot(next_position)

            if next_position == self.environment.goal:
                self.reached_goal = True
                return

            direct_step = self._step_direction(next_position, self.environment.goal)
            blocked_forward = (next_position[0] + direct_step[0], next_position[1] + direct_step[1])
            obstacle = self.environment.get_obstacle_at(blocked_forward) if self.environment.in_bounds(blocked_forward) else None
            if obstacle is None:
                continue

            self.obstacle_interactions += 1
            saved_position = obstacle.position
            detour_path = self.planner.plan_path(next_position, self.environment.goal)
            detour_cost = self._path_cost(detour_path) if detour_path else float("inf")

            occupied = {candidate.id: candidate.position for candidate in self.environment.obstacles}
            candidate_target, _ = self._simulate_push_target(obstacle, direct_step, self.default_push_distance, occupied)
            obstacle.position = saved_position
            if candidate_target is None:
                continue

            decision, _ = self.evaluator.decide_push_or_detour(
                self.environment.robot,
                obstacle,
                self.default_push_distance,
                detour_cost,
            )
            if decision == "push":
                self.push_obstacle(obstacle, direct_step, self.default_push_distance, primary=True)

        self.reached_goal = self.environment.robot.position == self.environment.goal


def run_beacon_human_like_trial(environment_name: str, seed: int, factor: str, factor_level: str, params: dict) -> TrialMetrics:
    environment = configure_environment(environment_name, seed, params)
    initial_obstacle_count = len(environment.obstacles)
    scene = semantics_env_to_beacon_scene(environment)
    # Tuned BEACON configuration: a slightly finer step lowers traversal energy
    # while remaining tractable across the full sweep.
    config = PlannerConfig(step=BEACON_TUNED_STEP, robot_radius=float(params["robot_radius"]))
    world = World.from_scene(scene)
    dstar_planner = BeaconDStarLitePlanner(world, config=config)
    push_policy = PushPolicy(
        WeightedBeaconInteractionCost(
            energy_weight=float(params["energy_weight"]),
            push_duration=config.step_duration,
        )
    )
    planner = HumanLikePlanner(dstar_planner, push_policy, world, alpha=min(environment.robot.sensing_radius, BEACON_TUNED_ALPHA))
    env = PyBulletEnv(world)

    executed_trajectory = [[float(world.start[0]), float(world.start[1]), 0.0]]
    cumulative_push_work = 0.0
    cumulative_navigation_work = 0.0
    cumulative_energy = 0.0
    action_log = []
    flowchart_trace = []
    stop_reason = "max_steps_reached"

    goal_tolerance = max(config.step, config.robot_radius)
    # Match the other experiment runners by scaling horizon with environment area
    # instead of using a tiny fixed cap that truncates otherwise valid rollouts.
    max_steps = max(50, environment.width * environment.height * 4)

    for step_index in range(max_steps):
        robot_position = env.get_robot_position()
        if math.dist(robot_position, world.goal) <= goal_tolerance:
            stop_reason = "goal_reached"
            break

        action = planner.step(robot_position)
        decision_trace = copy.deepcopy(planner.last_decision_trace)
        decision_trace["step_index"] = step_index

        if action is None:
            decision_trace["selected_action"] = decision_trace.get("selected_action", "stop")
            flowchart_trace.append(decision_trace)
            stop_reason = decision_trace["selected_action"]
            break

        execution = env.execute(action) or {}
        action_type = (
            "move" if isinstance(action, MoveAction)
            else "push" if isinstance(action, PushAction)
            else type(action).__name__.lower()
        )

        if isinstance(action, MoveAction):
            path_segment = action.path_segment or [
                [robot_position[0], robot_position[1], 0.0],
                [action.target[0], action.target[1], 0.0],
            ]
            if executed_trajectory[-1][:2] != path_segment[0][:2]:
                executed_trajectory.append([float(path_segment[0][0]), float(path_segment[0][1]), float(path_segment[0][2])])
            for point in path_segment[1:]:
                executed_trajectory.append([float(point[0]), float(point[1]), float(point[2])])
            move_distance = compute_path_length(path_segment)
            navigation_work = push_policy.cost_model.compute_navigation_work(move_distance)
            cumulative_navigation_work += navigation_work
            cumulative_energy += navigation_work
        else:
            move_distance = 0.0
            navigation_work = 0.0
            cumulative_push_work += float(getattr(action, "estimated_work", 0.0))
            cumulative_energy += float(getattr(action, "estimated_work", 0.0))

        updated_obstacle_positions = []
        if execution.get("updated_obstacle_ids"):
            for obstacle in world.obstacles:
                if obstacle.id in execution["updated_obstacle_ids"]:
                    updated_obstacle_positions.append(
                        {
                            "obstacle_id": int(obstacle.id),
                            "position": [round(obstacle.position[0], 4), round(obstacle.position[1], 4)],
                        }
                    )

        decision_trace["selected_action"] = action_type if action_type != "move" or decision_trace.get("selected_action") != "reroute" else "reroute"
        decision_trace["cumulative_energy"] = round(cumulative_energy, 4)
        decision_trace["updated_obstacle_positions"] = updated_obstacle_positions
        flowchart_trace.append(decision_trace)

        action_entry = {
            "step_index": step_index,
            "action_type": decision_trace["selected_action"],
            "robot_position_before": [round(robot_position[0], 4), round(robot_position[1], 4)],
            "robot_position_after": [round(env.get_robot_position()[0], 4), round(env.get_robot_position()[1], 4)],
            "navigation_distance": round(move_distance, 4),
            "navigation_work": round(navigation_work, 4),
            "cumulative_navigation_work": round(cumulative_navigation_work, 4),
            "cumulative_push_work": round(cumulative_push_work, 4),
            "cumulative_energy": round(cumulative_energy, 4),
            "updated_obstacle_positions": updated_obstacle_positions,
        }
        if isinstance(action, PushAction):
            action_entry.update(
                {
                    "obstacle_id": int(action.obstacle_id),
                    "push_distance": round(float(action.distance), 4),
                    "push_force": round(float(action.force), 4),
                    "push_work": round(float(action.estimated_work), 4),
                    "push_power": round(float(action.estimated_power), 4),
                    "push_duration": round(float(action.push_duration), 4),
                }
            )
        action_log.append(action_entry)
    else:
        robot_position = env.get_robot_position()
        if math.dist(robot_position, world.goal) <= goal_tolerance:
            stop_reason = "goal_reached"

    final_position = env.get_robot_position()
    success = math.dist(final_position, world.goal) <= goal_tolerance
    path_length = compute_path_length(executed_trajectory)
    obstacle_interactions = sum(1 for trace in flowchart_trace if trace.get("blocking_obstacle_id") is not None)
    pushed_entries = [entry for entry in action_log if entry["action_type"] == "push"]
    secondary_movements = sum(max(0, len(entry.get("updated_obstacle_positions", [])) - 1) for entry in pushed_entries)
    unique_moved = {
        update["obstacle_id"]
        for entry in action_log
        for update in entry.get("updated_obstacle_positions", [])
    }

    return TrialMetrics(
        environment=environment_name,
        seed=seed,
        planner="beacon_human_like",
        factor=factor,
        factor_level=factor_level,
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
        obstacle_interactions=obstacle_interactions,
        pushed_obstacles=len(pushed_entries),
        unique_pushed_obstacles=len({entry["obstacle_id"] for entry in pushed_entries if "obstacle_id" in entry}),
        secondary_obstacle_movements=secondary_movements,
        moved_obstacles=len(unique_moved),
        initial_obstacle_count=initial_obstacle_count,
    )


def planner_factory(planner_name: str, environment: Environment):
    if planner_name == "dstar_lite":
        return DStarLitePlanner(environment)
    if planner_name == "astar":
        return AStarPlanner(environment)
    if planner_name == "bug1":
        return BugPlanner(environment, mode="bug1")
    if planner_name == "bug2":
        return BugPlanner(environment, mode="bug2")
    raise ValueError(f"Unknown planner: {planner_name}")


def run_trial(environment_name: str, seed: int, planner_name: str, factor: str, factor_level: str, params: dict) -> TrialMetrics:
    if planner_name == "beacon_human_like":
        return run_beacon_human_like_trial(environment_name, seed, factor, factor_level, params)

    environment = configure_environment(environment_name, seed, params)
    initial_obstacle_count = len(environment.obstacles)
    evaluator = WeightedInteractionEvaluator(environment, energy_weight=params["energy_weight"])
    planner = planner_factory(planner_name, environment)
    executor = InstrumentedExecutor(environment, planner_name, planner, evaluator)
    executor.execute()
    path_length = euclidean_path_length(executor.robot_path)
    total_path_cost = path_length + params["energy_weight"] * executor.push_energy
    return TrialMetrics(
        environment=environment_name,
        seed=seed,
        planner=planner_name,
        factor=factor,
        factor_level=factor_level,
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


def build_scenarios() -> list[tuple[str, str, dict]]:
    scenarios = [("baseline", "baseline", dict(BASE_PARAMETERS))]
    for factor, values in FACTOR_LEVELS.items():
        for value in values:
            if value == BASE_PARAMETERS[factor]:
                continue
            params = dict(BASE_PARAMETERS)
            params[factor] = value
            scenarios.append((factor, str(value), params))
    return scenarios


def write_raw_csv(trials: list[TrialMetrics]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0].__dict__.keys()))
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial.__dict__)


def summarize(trials: list[TrialMetrics]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[TrialMetrics]] = defaultdict(list)
    for trial in trials:
        grouped[(trial.environment, trial.planner, trial.factor, trial.factor_level)].append(trial)

    summary_rows = []
    by_planner = defaultdict(list)
    for key, items in sorted(grouped.items()):
        environment, planner, factor, factor_level = key
        success_rate = sum(1 for item in items if item.success) / len(items)
        row = {
            "environment": environment,
            "planner": planner,
            "factor": factor,
            "factor_level": factor_level,
            "trials": len(items),
            "success_rate": round(success_rate, 4),
            "mean_total_path_cost": round(sum(item.total_path_cost for item in items) / len(items), 4),
            "mean_path_length": round(sum(item.path_length for item in items) / len(items), 4),
            "mean_time_to_goal": round(sum(item.time_to_goal for item in items) / len(items), 4),
            "mean_energy_consumption": round(sum(item.energy_consumption for item in items) / len(items), 4),
            "mean_obstacle_interactions": round(sum(item.obstacle_interactions for item in items) / len(items), 4),
            "mean_pushed_obstacles": round(sum(item.pushed_obstacles for item in items) / len(items), 4),
            "mean_unique_pushed_obstacles": round(sum(item.unique_pushed_obstacles for item in items) / len(items), 4),
            "mean_secondary_obstacle_movements": round(sum(item.secondary_obstacle_movements for item in items) / len(items), 4),
        }
        summary_rows.append(row)
        by_planner[planner].append(row)

    planner_rollup = []
    for planner, items in sorted(by_planner.items()):
        planner_rollup.append(
            {
                "planner": planner,
                "mean_success_rate": round(sum(item["success_rate"] for item in items) / len(items), 4),
                "mean_total_path_cost": round(sum(item["mean_total_path_cost"] for item in items) / len(items), 4),
                "mean_path_length": round(sum(item["mean_path_length"] for item in items) / len(items), 4),
                "mean_time_to_goal": round(sum(item["mean_time_to_goal"] for item in items) / len(items), 4),
                "mean_energy_consumption": round(sum(item["mean_energy_consumption"] for item in items) / len(items), 4),
                "mean_obstacle_interactions": round(sum(item["mean_obstacle_interactions"] for item in items) / len(items), 4),
                "mean_pushed_obstacles": round(sum(item["mean_pushed_obstacles"] for item in items) / len(items), 4),
                "mean_secondary_obstacle_movements": round(sum(item["mean_secondary_obstacle_movements"] for item in items) / len(items), 4),
            }
        )
    return summary_rows, planner_rollup


def write_summary(summary_rows: list[dict], planner_rollup: list[dict]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "assumptions": {
            "design": "one-factor-at-a-time sweep around a single baseline",
            "energy_consumption": "navigation work + push energy",
            "total_path_cost": "path_length + energy_weight * push_energy",
            "robot_radius_effect": "applied through expanded obstacle blocking in the experiment harness",
            "obstacle_density": "material density scale multiplier",
            "obstacle_size": "volume and polygon scale multiplier",
        },
        "planners": PLANNERS,
        "environments": ENVIRONMENTS,
        "baseline_parameters": BASE_PARAMETERS,
        "factor_levels": FACTOR_LEVELS,
        "summary_rows": summary_rows,
        "planner_rollup": planner_rollup,
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    scenarios = build_scenarios()
    trials: list[TrialMetrics] = []
    total_runs = len(ENVIRONMENTS) * len(PLANNERS) * len(scenarios)

    run_index = 0
    for env_index, environment_name in enumerate(ENVIRONMENTS):
        seed = 11 + env_index
        for factor, factor_level, params in scenarios:
            for planner_name in PLANNERS:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] env={environment_name} planner={planner_name} "
                    f"factor={factor}:{factor_level}",
                    flush=True,
                )
                trials.append(
                    run_trial(environment_name, seed, planner_name, factor, factor_level, params)
                )

    write_raw_csv(trials)
    summary_rows, planner_rollup = summarize(trials)
    write_summary(summary_rows, planner_rollup)

    print(f"Saved raw results to {RAW_CSV}")
    print(f"Saved summary CSV to {SUMMARY_CSV}")
    print(f"Saved summary JSON to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
