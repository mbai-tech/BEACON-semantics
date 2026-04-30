import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


CLASS_COLORS = {
    "safe": "#A8D5BA",
    "movable": "#F6D186",
    "fragile": "#F4A6A6",
    "forbidden": "#9AA5B1",
}


def draw_scene(scene, save_path=None, path_points=None, title_suffix=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    xmin, xmax, ymin, ymax = scene["workspace"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    for obstacle in scene["obstacles"]:
        vertices = obstacle["vertices"]
        color = CLASS_COLORS.get(obstacle.get("class_true"), "lightgray")
        patch = MplPolygon(
            vertices,
            closed=True,
            edgecolor="black",
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(patch)

    sx, sy, *_ = scene["start"]
    gx, gy, *_ = scene["goal"]
    ax.plot(sx, sy, "go", markersize=8, label="Start")
    ax.plot(gx, gy, "ro", markersize=8, label="Goal")

    if path_points:
        path_x = [point[0] for point in path_points]
        path_y = [point[1] for point in path_points]
        ax.plot(path_x, path_y, color="#1D4ED8", linewidth=2.5, label="Path")

    title = f"Environment: {scene['family']}"
    if title_suffix:
        title = f"{title} {title_suffix}"

    ax.set_title(title)
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
