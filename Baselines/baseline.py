#!/usr/bin/env python3

import argparse
import json
import math
import sys
from heapq import heappop, heappush
from itertools import count
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
ENV_DIR = REPO_ROOT / "enviornment"
DATA_DIR = REPO_ROOT / "data"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from scene_generator import generate_scene, save_scene_json  # noqa: E402
from validator import build_grid  # noqa: E402


INF = float("inf")
SQRT2 = math.sqrt(2.0)
VALID_FAMILIES = [
    "narrow_passage",
    "sparse_clutter",
    "dense_clutter",
    "semantic_trap",
    "perturbed",
]


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = count()

    def insert(self, node, key):
        self.remove(node)
        entry = [key, next(self.counter), node]
        self.entry_finder[node] = entry
        heappush(self.heap, entry)

    def remove(self, node):
        entry = self.entry_finder.pop(node, None)
        if entry is not None:
            entry[-1] = None

    def pop(self):
        while self.heap:
            _, _, node = heappop(self.heap)
            if node is not None:
                self.entry_finder.pop(node, None)
                return node
        raise KeyError("pop from empty priority queue")

    def top_key(self):
        while self.heap:
            key, _, node = self.heap[0]
            if node is not None:
                return key
            heappop(self.heap)
        return (INF, INF)

    def contains(self, node):
        return node in self.entry_finder


class DStarLite:
    def __init__(self, xs, ys, blocked, start, goal):
        self.xs = xs
        self.ys = ys
        self.width = len(xs)
        self.height = len(ys)
        self.blocked = set(blocked)
        self.start = start
        self.goal = goal
        self.last_start = start
        self.km = 0.0
        self.U = PriorityQueue()
        self.g = {}
        self.rhs = {}

        for x in range(self.width):
            for y in range(self.height):
                node = (x, y)
                self.g[node] = INF
                self.rhs[node] = INF

        self.rhs[self.goal] = 0.0
        self.U.insert(self.goal, self.calculate_key(self.goal))

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        diag = min(dx, dy)
        straight = max(dx, dy) - diag
        return diag * SQRT2 + straight

    def calculate_key(self, node):
        best = min(self.g[node], self.rhs[node])
        return (best + self.heuristic(self.start, node) + self.km, best)

    def in_bounds(self, node):
        return 0 <= node[0] < self.width and 0 <= node[1] < self.height

    def traversable(self, node):
        return self.in_bounds(node) and node not in self.blocked

    def neighbors(self, node):
        x, y = node
        results = []
        for dx, dy in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            nxt = (x + dx, y + dy)
            if self.traversable(nxt):
                results.append(nxt)
        return results

    def predecessors(self, node):
        return self.neighbors(node)

    def cost(self, a, b):
        if not self.traversable(a) or not self.traversable(b):
            return INF
        if a[0] != b[0] and a[1] != b[1]:
            return SQRT2
        return 1.0

    def update_vertex(self, node):
        if node != self.goal:
            successors = self.neighbors(node)
            self.rhs[node] = min(
                (self.cost(node, nxt) + self.g[nxt] for nxt in successors),
                default=INF,
            )

        if self.U.contains(node):
            self.U.remove(node)

        if self.g[node] != self.rhs[node]:
            self.U.insert(node, self.calculate_key(node))

    def compute_shortest_path(self):
        while (
            self.U.top_key() < self.calculate_key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            k_old = self.U.top_key()
            node = self.U.pop()
            k_new = self.calculate_key(node)
            if k_old < k_new:
                self.U.insert(node, k_new)
                continue

            if self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                for pred in self.predecessors(node):
                    self.update_vertex(pred)
            else:
                self.g[node] = INF
                self.update_vertex(node)
                for pred in self.predecessors(node):
                    self.update_vertex(pred)

    def extract_path(self):
        if self.g[self.start] == INF and self.rhs[self.start] == INF:
            return None

        path = [self.start]
        current = self.start
        max_steps = self.width * self.height

        for _ in range(max_steps):
            if current == self.goal:
                return path

            candidates = self.neighbors(current)
            if not candidates:
                return None

            next_node = min(
                candidates,
                key=lambda node: (
                    self.cost(current, node) + self.g[node],
                    self.heuristic(node, self.goal),
                ),
            )
            if self.g[next_node] == INF:
                return None

            path.append(next_node)
            current = next_node

        return None


def load_scene(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent_dir(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def default_plan_path(scene_path):
    return DATA_DIR / "plans" / f"{scene_path.stem}_dstar.json"


def default_plan_image_path(scene_path):
    return DATA_DIR / "plan_images" / f"{scene_path.stem}_dstar.png"


def default_scene_output_paths(family, seed):
    scenes_dir = DATA_DIR / "scenes"
    images_dir = DATA_DIR / "images"
    if seed is None:
        name = f"{family}_scene"
    else:
        name = f"{family}_seed{seed}"
    return scenes_dir / f"{name}.json", images_dir / f"{name}.png"


def nearest_free_index(xs, ys, blocked, point_xyz):
    px, py = point_xyz[:2]
    best = None
    best_dist = INF

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            if (ix, iy) in blocked:
                continue
            dist = (x - px) ** 2 + (y - py) ** 2
            if dist < best_dist:
                best = (ix, iy)
                best_dist = dist

    return best


def grid_path_to_world(xs, ys, grid_path, start_xyz, goal_xyz):
    if not grid_path:
        return None

    start_z = float(start_xyz[2]) if len(start_xyz) > 2 else 0.0
    goal_z = float(goal_xyz[2]) if len(goal_xyz) > 2 else start_z
    total_segments = max(1, len(grid_path) - 1)

    points = []
    for i, (ix, iy) in enumerate(grid_path):
        t = i / total_segments
        z = start_z + (goal_z - start_z) * t
        points.append([float(xs[ix]), float(ys[iy]), z])

    points[0] = [float(start_xyz[0]), float(start_xyz[1]), start_z]
    points[-1] = [float(goal_xyz[0]), float(goal_xyz[1]), goal_z]
    return points


def compute_path_length(points):
    if not points or len(points) < 2:
        return 0.0

    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.dist(a, b)
    return total


def plan_scene(scene, step=0.15, robot_radius=0.12, mode="collision_free"):
    xs, ys, blocked = build_grid(
        scene,
        step=step,
        mode=mode,
        robot_radius=robot_radius,
    )
    start = nearest_free_index(xs, ys, blocked, scene["start"])
    goal = nearest_free_index(xs, ys, blocked, scene["goal"])

    if start is None or goal is None:
        return None, {
            "success": False,
            "reason": "Start or goal is not reachable on the discretized grid.",
        }

    planner = DStarLite(xs, ys, blocked, start, goal)
    planner.compute_shortest_path()
    grid_path = planner.extract_path()
    if not grid_path:
        return None, {
            "success": False,
            "reason": "No path found.",
        }

    world_path = grid_path_to_world(xs, ys, grid_path, scene["start"], scene["goal"])
    return world_path, {
        "success": True,
        "grid_start": list(start),
        "grid_goal": list(goal),
        "num_waypoints": len(world_path),
        "path_length": round(compute_path_length(world_path), 4),
    }


def build_plan_payload(scene, scene_path, world_path, metadata, step, robot_radius, mode):
    return {
        "algorithm": "d_star_lite",
        "scene_file": str(scene_path),
        "scene_family": scene["family"],
        "workspace": list(scene["workspace"]),
        "planning_mode": mode,
        "grid_step": step,
        "robot_radius": robot_radius,
        "start": scene["start"],
        "goal": scene["goal"],
        "success": metadata["success"],
        "summary": metadata,
        "path": world_path or [],
    }


def command_plan(args):
    scene_path = Path(args.scene).resolve()
    output_path = Path(args.output).resolve() if args.output else default_plan_path(scene_path)
    scene = load_scene(scene_path)
    world_path, metadata = plan_scene(
        scene,
        step=args.step,
        robot_radius=args.robot_radius,
        mode=args.mode,
    )
    payload = build_plan_payload(
        scene,
        scene_path,
        world_path,
        metadata,
        args.step,
        args.robot_radius,
        args.mode,
    )
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved plan JSON to {output_path}")
    print(f"Success: {payload['success']}")
    if payload["success"]:
        print(f"Waypoints: {payload['summary']['num_waypoints']}")
        print(f"Path length: {payload['summary']['path_length']}")
    else:
        print(f"Reason: {payload['summary']['reason']}")


def command_render(args):
    from draw_scene import draw_scene

    scene_path = Path(args.scene).resolve()
    plan_path = Path(args.plan).resolve()
    output_path = Path(args.output).resolve() if args.output else default_plan_image_path(scene_path)

    scene = load_scene(scene_path)
    plan = load_scene(plan_path)

    path_points = None
    if plan.get("success"):
        path_points = [point[:2] for point in plan.get("path", [])]

    ensure_parent_dir(output_path)
    draw_scene(
        scene,
        save_path=output_path,
        path_points=path_points,
        title_suffix="(D* Lite)",
    )
    print(f"Saved PNG to {output_path}")


def command_generate_scene(args):
    from draw_scene import draw_scene

    scene = generate_scene(args.family, seed=args.seed)
    default_scene_path, default_image_path = default_scene_output_paths(
        args.family, scene["seed"] if args.seed is not None else None
    )

    scene_output = Path(args.scene_output).resolve() if args.scene_output else default_scene_path
    image_output = Path(args.image_output).resolve() if args.image_output else default_image_path

    ensure_parent_dir(scene_output)
    ensure_parent_dir(image_output)
    save_scene_json(scene, scene_output)
    draw_scene(scene, image_output)

    print(f"Seed: {scene['seed']}")
    print(f"Saved scene JSON to {scene_output}")
    print(f"Saved image to {image_output}")


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run a D* Lite planning baseline on environment scene JSON files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Plan over a scene JSON and save a result JSON.",
    )
    plan_parser.add_argument("--scene", required=True, help="Path to the input scene JSON.")
    plan_parser.add_argument(
        "--output",
        help="Path to save the plan JSON. Defaults to data/plans/<scene>_dstar.json.",
    )
    plan_parser.add_argument(
        "--mode",
        default="collision_free",
        choices=["collision_free", "contact_allowed"],
        help="Obstacle semantics to use during planning.",
    )
    plan_parser.add_argument(
        "--step",
        type=float,
        default=0.15,
        help="Grid resolution used to discretize the workspace.",
    )
    plan_parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.12,
        help="Robot radius used while rasterizing obstacle occupancy.",
    )
    plan_parser.set_defaults(func=command_plan)

    render_parser = subparsers.add_parser(
        "render",
        help="Render a scene and its D* Lite path to a PNG.",
    )
    render_parser.add_argument("--scene", required=True, help="Path to the input scene JSON.")
    render_parser.add_argument("--plan", required=True, help="Path to the plan JSON produced by the plan command.")
    render_parser.add_argument(
        "--output",
        help="Path to save the PNG. Defaults to data/plan_images/<scene>_dstar.png.",
    )
    render_parser.set_defaults(func=command_render)

    generate_parser = subparsers.add_parser(
        "generate-scene",
        help="Generate a scene JSON and PNG for a supported family.",
    )
    generate_parser.add_argument(
        "--family",
        required=True,
        choices=VALID_FAMILIES,
        help="Scene family to generate.",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible scene generation.",
    )
    generate_parser.add_argument(
        "--scene-output",
        help="Optional path for the generated scene JSON.",
    )
    generate_parser.add_argument(
        "--image-output",
        help="Optional path for the generated scene PNG.",
    )
    generate_parser.set_defaults(func=command_generate_scene)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
