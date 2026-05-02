import json
import math
import random
from shapely.geometry import Point, Polygon, box, LineString
from shapely.affinity import translate, rotate

WORKSPACE = (0, 6, 0, 6)  # xmin, xmax, ymin, ymax
CLASSES = ["safe", "movable", "fragile"]
SHAPE_TYPES = [
    "circle",
    "rectangle",
    "triangle",
    "trapezoid",
    "parallelogram",
    "pentagon",
    "hexagon",
]
MATERIAL_DENSITIES = {
    "safe": 1.0,
    "movable": 0.8,
    "fragile": 0.35,
}
MATERIAL_PROPERTIES = {
    "safe": {
        "friction": 0.9,
        "fragility": 0.05,
        "pushable": False,
    },
    "movable": {
        "friction": 0.55,
        "fragility": 0.2,
        "pushable": True,
    },
    "fragile": {
        "friction": 0.75,
        "fragility": 0.95,
        "pushable": False,
    },
}


class Material:
    def __init__(
        self,
        name: str,
        mass: float,
        friction: float,
        fragility: float,
        pushable: bool = True,
    ):
        self.name = name
        self.mass = mass
        self.friction = friction
        self.fragility = fragility
        self.pushable = pushable

    def to_dict(self):
        return {
            "name": self.name,
            "density": MATERIAL_DENSITIES[self.name],
            "mass": round(self.mass, 4),
            "friction": self.friction,
            "fragility": self.fragility,
            "pushable": self.pushable,
        }


class Obstacle:
    def __init__(
        self,
        id: int,
        shape: str,
        position: tuple[float, float],
        size: dict,
        material: Material,
        movable: bool = True,
    ):
        self.id = id
        self.shape = shape
        self.position = position
        self.size = size
        self.material = material
        self.movable = movable

    def to_dict(self, vertices, class_true):
        return {
            "id": self.id,
            "shape": self.shape,
            "shape_type": self.shape,
            "class_true": class_true,
            "center": [round(self.position[0], 4), round(self.position[1], 4)],
            "position": [round(self.position[0], 4), round(self.position[1], 4)],
            "size": self.size,
            "material": self.material.to_dict(),
            "movable": self.movable,
            "vertices": vertices,
        }


def polygon_to_list(poly):
    coords = list(poly.exterior.coords)[:-1]
    return [[float(x), float(y)] for x, y in coords]


def polygon_size(poly):
    minx, miny, maxx, maxy = poly.bounds
    return {
        "area": round(poly.area, 4),
        "width": round(maxx - minx, 4),
        "height": round(maxy - miny, 4),
    }


def make_material(material_name, size):
    density = MATERIAL_DENSITIES[material_name]
    props = MATERIAL_PROPERTIES[material_name]
    mass = density * size["area"]
    return Material(
        name=material_name,
        mass=mass,
        friction=props["friction"],
        fragility=props["fragility"],
        pushable=props["pushable"],
    )


def within_workspace(poly):
    xmin, xmax, ymin, ymax = WORKSPACE
    workspace_poly = box(xmin, ymin, xmax, ymax)
    return workspace_poly.contains(poly)


def valid_candidate(candidate, placed, start_buffer=None, goal_buffer=None):
    if not candidate.is_valid:
        return False
    if not within_workspace(candidate):
        return False

    # touching allowed, overlap forbidden
    if start_buffer is not None and candidate.intersection(start_buffer).area > 1e-9:
        return False
    if goal_buffer is not None and candidate.intersection(goal_buffer).area > 1e-9:
        return False

    for p in placed:
        if candidate.intersection(p).area > 1e-9:
            return False

    return True


def make_rectangle(w, h):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2, h / 2),
        (-w / 2, h / 2),
    ])


def make_triangle(w, h):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (0, h / 2),
    ])


def make_trapezoid(bottom, top, h):
    return Polygon([
        (-bottom / 2, -h / 2),
        (bottom / 2, -h / 2),
        (top / 2, h / 2),
        (-top / 2, h / 2),
    ])


def make_parallelogram(w, h, skew):
    return Polygon([
        (-w / 2, -h / 2),
        (w / 2, -h / 2),
        (w / 2 + skew, h / 2),
        (-w / 2 + skew, h / 2),
    ])


def make_regular_ngon(n, r):
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return Polygon(pts)


def make_shape(shape_type, scale_min=0.15, scale_max=0.45):
    if shape_type == "circle":
        r = random.uniform(scale_min, scale_max)
        return Point(0, 0).buffer(r, resolution=32)

    if shape_type == "rectangle":
        w = random.uniform(scale_min * 1.5, scale_max * 2.2)
        h = random.uniform(scale_min * 1.2, scale_max * 1.8)
        return make_rectangle(w, h)

    if shape_type == "triangle":
        w = random.uniform(scale_min * 1.6, scale_max * 2.1)
        h = random.uniform(scale_min * 1.4, scale_max * 1.9)
        return make_triangle(w, h)

    if shape_type == "trapezoid":
        bottom = random.uniform(scale_min * 1.6, scale_max * 2.3)
        top = bottom * random.uniform(0.45, 0.8)
        h = random.uniform(scale_min * 1.2, scale_max * 1.8)
        return make_trapezoid(bottom, top, h)

    if shape_type == "parallelogram":
        w = random.uniform(scale_min * 1.6, scale_max * 2.2)
        h = random.uniform(scale_min * 1.1, scale_max * 1.7)
        skew = random.uniform(0.08, 0.28)
        return make_parallelogram(w, h, skew)

    if shape_type == "pentagon":
        r = random.uniform(scale_min, scale_max)
        return make_regular_ngon(5, r)

    if shape_type == "hexagon":
        r = random.uniform(scale_min, scale_max)
        return make_regular_ngon(6, r)

    raise ValueError(f"Unknown shape_type: {shape_type}")


def place_shape_at(poly, x, y, angle_deg=None):
    if angle_deg is None:
        angle_deg = random.uniform(0, 180)

    poly = rotate(poly, angle_deg, origin=(0, 0), use_radians=False)
    poly = translate(poly, xoff=x, yoff=y)
    return poly


def make_random_polygon(scale_min=0.15, scale_max=0.45):
    xmin, xmax, ymin, ymax = WORKSPACE

    shape_type = random.choice(SHAPE_TYPES)
    base_poly = make_shape(shape_type, scale_min=scale_min, scale_max=scale_max)

    minx, miny, maxx, maxy = base_poly.bounds
    x = random.uniform(xmin - minx, xmax - maxx)
    y = random.uniform(ymin - miny, ymax - maxy)

    poly = place_shape_at(base_poly, x, y)
    return poly, shape_type


def random_start_goal(min_dist=3.5, margin=0.45):
    xmin, xmax, ymin, ymax = WORKSPACE

    for _ in range(500):
        sx = random.uniform(xmin + margin, xmax - margin)
        sy = random.uniform(ymin + margin, ymax - margin)
        gx = random.uniform(xmin + margin, xmax - margin)
        gy = random.uniform(ymin + margin, ymax - margin)

        if ((sx - gx) ** 2 + (sy - gy) ** 2) ** 0.5 >= min_dist:
            return (sx, sy), (gx, gy)

    return (0.6, 0.6), (5.4, 5.4)


def obstacle_record(poly, idx, cls, shape_type):
    center = poly.centroid
    size = polygon_size(poly)
    material = make_material(cls, size)
    obstacle = Obstacle(
        id=idx,
        shape=shape_type,
        position=(center.x, center.y),
        size=size,
        material=material,
        movable=material.pushable,
    )
    return obstacle.to_dict(
        vertices=polygon_to_list(poly),
        class_true=cls,
    )


def try_add_obstacle(poly, cls, shape_type, placed, obstacles, start=None, goal=None):
    start_buffer = Point(start).buffer(0.35) if start is not None else None
    goal_buffer = Point(goal).buffer(0.35) if goal is not None else None

    if valid_candidate(poly, placed, start_buffer, goal_buffer):
        placed.append(poly)
        obstacles.append(obstacle_record(poly, len(obstacles), cls, shape_type))
        return True

    return False


def sample_background_obstacles(
    n_min,
    n_max,
    start,
    goal,
    placed=None,
    class_weights=None,
    scale_min=0.15,
    scale_max=0.45,
):
    if placed is None:
        placed = []

    if class_weights is None:
        class_weights = {
            "safe": 0.2,
            "movable": 0.6,
            "fragile": 0.2,
        }

    classes = list(class_weights.keys())
    weights = list(class_weights.values())

    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    n_obs = random.randint(n_min, n_max)
    obstacles = []
    attempts = 0
    max_attempts = 15000

    while len(obstacles) < n_obs and attempts < max_attempts:
        attempts += 1

        candidate, shape_type = make_random_polygon(
            scale_min=scale_min,
            scale_max=scale_max,
        )

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)
            cls = random.choices(classes, weights=weights, k=1)[0]
            obstacles.append(obstacle_record(candidate, len(obstacles), cls, shape_type))

    return placed, obstacles


def generate_sparse(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal()

    _, obstacles = sample_background_obstacles(
        8,
        15,
        start,
        goal,
        scale_min=0.14,
        scale_max=0.32,
    )

    return {
        "family": "sparse",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles,
    }


def generate_cluttered(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal()

    _, obstacles = sample_background_obstacles(
        25,
        40,
        start,
        goal,
        scale_min=0.12,
        scale_max=0.24,
    )

    return {
        "family": "cluttered",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles,
    }


def generate_collision_required(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal(min_dist=4.4)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    midpoint = line.interpolate(0.5, normalized=True)

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5

    if length == 0:
        dx, dy, length = 1.0, 0.0, 1.0

    ux = dx / length
    uy = dy / length
    px = -dy / length
    py = dx / length

    # Long wall from one side of the workspace to the other.
    # More columns = longer wall. More rows = thicker wall.
    row_offsets = [-1.05, -0.72, -0.39, -0.06, 0.27, 0.60, 0.93]
    col_offsets = [
        -4.0, -3.55, -3.10, -2.65, -2.20, -1.75, -1.30, -0.85, -0.40,
         0.05,  0.50,  0.95,  1.40,  1.85,  2.30,  2.75,  3.20,  3.65,  4.10
    ]

    for row in row_offsets:
        for col in col_offsets:
            jitter_u = random.uniform(-0.07, 0.07)
            jitter_p = random.uniform(-0.07, 0.07)

            x = midpoint.x + ux * (row + jitter_u) + px * (col + jitter_p)
            y = midpoint.y + uy * (row + jitter_u) + py * (col + jitter_p)

            shape_type = random.choice(SHAPE_TYPES)
            base = make_shape(shape_type, scale_min=0.16, scale_max=0.28)
            poly = place_shape_at(base, x, y)

            if not within_workspace(poly):
                continue

            # Middle of the wall is mostly movable so collision is useful.
            # Edges are more fragile so random side-collision is costly.
            if abs(row) <= 0.45:
                cls = random.choices(
                    ["movable", "fragile", "safe"],
                    weights=[0.70, 0.20, 0.10],
                    k=1
                )[0]
            else:
                cls = random.choices(
                    ["fragile", "movable", "safe"],
                    weights=[0.55, 0.30, 0.15],
                    k=1
                )[0]

            try_add_obstacle(poly, cls, shape_type, placed, obstacles, start, goal)

    # Add extra clutter away from the wall so the scene does not look empty.
    start_buffer = Point(start).buffer(0.35)
    goal_buffer = Point(goal).buffer(0.35)

    extra_targets = random.randint(12, 20)
    attempts = 0
    max_attempts = 20000

    while extra_targets > 0 and attempts < max_attempts:
        attempts += 1

        candidate, shape_type = make_random_polygon(scale_min=0.12, scale_max=0.24)
        c = candidate.centroid

        rel_x = c.x - midpoint.x
        rel_y = c.y - midpoint.y

        along = rel_x * ux + rel_y * uy
        across = rel_x * px + rel_y * py

        # Keep extra obstacles outside the main wall band
        in_wall_band = abs(along) < 1.25
        if in_wall_band:
            continue

        if valid_candidate(candidate, placed, start_buffer, goal_buffer):
            placed.append(candidate)

            cls = random.choices(
                ["safe", "movable", "fragile"],
                weights=[0.25, 0.35, 0.40],
                k=1
            )[0]

            obstacles.append(obstacle_record(candidate, len(obstacles), cls, shape_type))
            extra_targets -= 1

    return {
        "family": "collision_required",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles,
    }


def generate_collision_shortcut(seed=None):
    if seed is not None:
        random.seed(seed)

    start, goal = random_start_goal(min_dist=4.0)
    placed = []
    obstacles = []

    line = LineString([start, goal])
    fractions = [0.35, 0.5, 0.65]

    for frac in fractions:
        p = line.interpolate(frac, normalized=True)

        shape_type = random.choice(SHAPE_TYPES)
        base = make_shape(shape_type, scale_min=0.18, scale_max=0.30)
        poly = place_shape_at(base, p.x, p.y)

        try_add_obstacle(poly, "movable", shape_type, placed, obstacles, start, goal)

    placed, bg = sample_background_obstacles(
        14,
        22,
        start,
        goal,
        placed=placed,
        class_weights={
            "safe": 0.25,
            "movable": 0.5,
            "fragile": 0.25,
        },
        scale_min=0.12,
        scale_max=0.24,
    )

    for obs in bg:
        obs["id"] = len(obstacles)
        obstacles.append(obs)

    return {
        "family": "collision_shortcut",
        "workspace": WORKSPACE,
        "start": [start[0], start[1], 0.0],
        "goal": [goal[0], goal[1], 0.0],
        "obstacles": obstacles,
    }


def generate_scene(family="sparse", seed=None):
    if family == "sparse":
        return generate_sparse(seed=seed)

    if family == "cluttered":
        return generate_cluttered(seed=seed)

    if family == "collision_required":
        return generate_collision_required(seed=seed)

    if family == "collision_shortcut":
        return generate_collision_shortcut(seed=seed)

    raise ValueError(
        "family must be one of: "
        "'sparse', 'cluttered', 'collision_required', 'collision_shortcut'"
    )


def save_scene_json(scene, path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)
