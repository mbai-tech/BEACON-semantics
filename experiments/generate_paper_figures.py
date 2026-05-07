#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/ishita/Documents/GitHub/BEACON-semantics")
RAW_CSV = ROOT / "data/experiments/semantics/planner_sweep_raw.csv"
OUT_DIR = ROOT / "data/experiments/semantics/paper_figures"
EXAMPLE_DIR = ROOT / "data/examples/example_runs_beacon_success_small"

PLANNER_ORDER = ["bug1", "bug2", "dstar_lite", "beacon_human_like"]
PLANNER_LABELS = {
    "bug1": "Bug1",
    "bug2": "Bug2",
    "dstar_lite": "D* Lite",
    "beacon_human_like": "BEACON",
}
PLANNER_COLORS = {
    "bug1": "#f28e2b",
    "bug2": "#e15759",
    "dstar_lite": "#4e79a7",
    "beacon_human_like": "#59a14f",
}


def load_rows() -> list[dict]:
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def planner_rollup(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["planner"]].append(row)

    output = []
    for planner in PLANNER_ORDER:
        items = grouped[planner]
        output.append(
            {
                "planner": planner,
                "label": PLANNER_LABELS[planner],
                "success_rate": sum(1 for item in items if item["success"].lower() == "true") / len(items),
                "mean_time": sum(float(item["time_to_goal"]) for item in items) / len(items),
                "mean_path": sum(float(item["path_length"]) for item in items) / len(items),
                "mean_energy": sum(float(item["energy_consumption"]) for item in items) / len(items),
                "nonzero_energy_rate": sum(1 for item in items if float(item["energy_consumption"]) > 0.0) / len(items),
            }
        )
    return output


def environment_matrices(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    envs = sorted({row["environment"] for row in rows})
    success = np.zeros((len(envs), len(PLANNER_ORDER)), dtype=float)
    energy = np.zeros((len(envs), len(PLANNER_ORDER)), dtype=float)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["planner"])].append(row)

    for env_index, env in enumerate(envs):
        for planner_index, planner in enumerate(PLANNER_ORDER):
            items = grouped[(env, planner)]
            success[env_index, planner_index] = sum(1 for item in items if item["success"].lower() == "true") / len(items)
            energy[env_index, planner_index] = sum(float(item["energy_consumption"]) for item in items) / len(items)
    return envs, success, energy


def save_overview_figure(summary: list[dict]) -> None:
    labels = [item["label"] for item in summary]
    colors = [PLANNER_COLORS[item["planner"]] for item in summary]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    metrics = [
        ("success_rate", "Success Rate", (0.0, 1.08), "{:.2f}"),
        ("mean_time", "Avg. Steps", None, "{:.1f}"),
        ("mean_energy", "Avg. Energy", None, "{:.2f}"),
    ]

    for ax, (field, title, ylim, fmt) in zip(axes, metrics):
        values = [item[field] for item in summary]
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0.0, max(values) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (ax.get_ylim()[1] * 0.02),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Planner Comparison on the Current BEACON Sweep", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "planner_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_figure(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for item in summary:
        ax.scatter(
            item["mean_energy"],
            item["success_rate"],
            s=80 + item["mean_time"] * 2.5,
            color=PLANNER_COLORS[item["planner"]],
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
        )
        ax.text(
            item["mean_energy"] + 0.2,
            item["success_rate"] + 0.01,
            item["label"],
            fontsize=10,
        )

    ax.set_xlabel("Average Traversal Energy")
    ax.set_ylabel("Success Rate")
    ax.set_title("Energy–Success Tradeoff")
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "energy_success_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_figure(envs: list[str], success: np.ndarray, energy: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    datasets = [
        (success, "Success Rate by Environment", "Success Rate"),
        (energy, "Average Energy by Environment", "Avg. Energy"),
    ]

    for ax, (matrix, title, cbar_label) in zip(axes, datasets):
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
        ax.set_title(title)
        ax.set_xticks(range(len(PLANNER_ORDER)))
        ax.set_xticklabels([PLANNER_LABELS[p] for p in PLANNER_ORDER], rotation=20, ha="right")
        ax.set_yticks(range(len(envs)))
        ax.set_yticklabels(envs)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "environment_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_panel() -> None:
    stems = [
        "02_open_room_seed03_bug1.png",
        "02_open_room_seed03_bug2.png",
        "02_open_room_seed03_dstar_lite.png",
        "02_open_room_seed03_beacon_human_like.png",
    ]
    paths = [EXAMPLE_DIR / stem for stem in stems]
    if not all(path.exists() for path in paths):
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 10.0))
    titles = ["Bug1", "Bug2", "D* Lite", "BEACON"]
    for ax, path, title in zip(axes.flat, paths, titles):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Representative Multi-Planner Trajectories (Open Room, Seed 03)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qualitative_example_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(summary: list[dict]) -> None:
    lines = [
        "# Paper Figures",
        "",
        "Generated from `data/experiments/semantics/planner_sweep_raw.csv`.",
        "",
        "## Files",
        "- `planner_overview.png`: success, average steps, and average traversal energy by planner.",
        "- `energy_success_tradeoff.png`: success rate vs. energy scatter, marker size scales with average steps.",
        "- `environment_heatmaps.png`: environment-by-planner heatmaps for success and energy.",
        "- `qualitative_example_panel.png`: 2x2 qualitative panel of saved example trajectories.",
        "",
        "## Rollup",
    ]
    for item in summary:
        lines.append(
            f"- {item['label']}: success={item['success_rate']:.3f}, "
            f"steps={item['mean_time']:.2f}, energy={item['mean_energy']:.3f}"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = planner_rollup(rows)
    envs, success, energy = environment_matrices(rows)

    save_overview_figure(summary)
    save_tradeoff_figure(summary)
    save_heatmap_figure(envs, success, energy)
    save_qualitative_panel()
    write_manifest(summary)
    print(f"Saved paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
