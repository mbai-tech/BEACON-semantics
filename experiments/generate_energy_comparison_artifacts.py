#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/ishita/Documents/GitHub/BEACON-semantics")
RAW_CSV = ROOT / "data/experiments/semantics/planner_sweep_raw.csv"
OUTPUT_DIR = ROOT / "data/experiments/semantics/energy_comparison"

PLANNER_ORDER = ["dstar_lite", "bug1", "bug2", "beacon_human_like"]
PLANNER_LABELS = {
    "dstar_lite": "D* Lite",
    "bug1": "Bug1",
    "bug2": "Bug2",
    "beacon_human_like": "BEACON",
}


def load_rows() -> list[dict]:
    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def planner_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["planner"]].append(row)

    summary = []
    for planner in PLANNER_ORDER:
        items = grouped[planner]
        energies = [float(item["energy_consumption"]) for item in items]
        successes = sum(1 for item in items if item["success"].lower() == "true")
        summary.append(
            {
                "planner": planner,
                "planner_label": PLANNER_LABELS[planner],
                "runs": len(items),
                "successes": successes,
                "success_rate": successes / max(len(items), 1),
                "mean_energy": statistics.fmean(energies) if energies else 0.0,
                "median_energy": statistics.median(energies) if energies else 0.0,
                "max_energy": max(energies) if energies else 0.0,
                "nonzero_energy_runs": sum(1 for energy in energies if energy > 0.0),
            }
        )
    return summary


def environment_summary(rows: list[dict]) -> tuple[list[str], list[list[float]]]:
    envs = sorted({row["environment"] for row in rows})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["planner"])].append(float(row["energy_consumption"]))

    matrix = []
    for env in envs:
        matrix.append(
            [
                statistics.fmean(grouped[(env, planner)]) if grouped[(env, planner)] else 0.0
                for planner in PLANNER_ORDER
            ]
        )
    return envs, matrix


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_overall_table(path: Path, summary: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Energy usage comparison across planners for the recorded experiment sweep. Push energy remained zero in every run, so all planners show identical energy statistics.}",
        r"\label{tab:energy-comparison-overall}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"\textbf{Planner} & \textbf{Runs} & \textbf{Success Rate} & \textbf{Mean Energy} & \textbf{Max Energy} & \textbf{Nonzero Runs} \\",
        r"\hline",
    ]
    for row in summary:
        lines.append(
            f"{row['planner_label']} & "
            f"{row['runs']} & "
            f"{row['success_rate']:.3f} & "
            f"{row['mean_energy']:.1f} & "
            f"{row['max_energy']:.1f} & "
            f"{row['nonzero_energy_runs']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_environment_table(path: Path, envs: list[str], matrix: list[list[float]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean recorded energy consumption by environment and planner. All entries are zero because no planner executed a push in this sweep.}",
        r"\label{tab:energy-comparison-by-env}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"\textbf{Environment} & \textbf{D* Lite} & \textbf{Bug1} & \textbf{Bug2} & \textbf{BEACON} \\",
        r"\hline",
    ]
    for env, values in zip(envs, matrix):
        env_label = env.replace("_", r"\_")
        lines.append(
            f"{env_label} & "
            f"{values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f} & {values[3]:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overall_bar(path: Path, summary: list[dict]) -> None:
    labels = [row["planner_label"] for row in summary]
    values = [row["mean_energy"] for row in summary]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_ylabel("Mean Energy Consumption")
    ax.set_title("Planner Energy Usage Across the Sweep")
    ax.set_ylim(0.0, 1.0 if max(values, default=0.0) == 0.0 else max(values) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            (bar.get_height() + 0.03) if value > 0.0 else 0.03,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.text(
        0.5,
        0.92,
        "All planners recorded zero push energy in this sweep",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.3"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_environment_heatmap(path: Path, envs: list[str], matrix: list[list[float]]) -> None:
    data = np.array(matrix, dtype=float)
    vmax = 1.0 if math.isclose(float(np.max(data)), 0.0) else float(np.max(data))

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(PLANNER_ORDER)))
    ax.set_xticklabels([PLANNER_LABELS[planner] for planner in PLANNER_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels(envs)
    ax.set_title("Mean Energy Consumption by Environment and Planner")

    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            ax.text(col_index, row_index, f"{data[row_index, col_index]:.1f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Energy Consumption")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = planner_summary(rows)
    envs, matrix = environment_summary(rows)

    write_csv(OUTPUT_DIR / "energy_usage_by_planner.csv", summary)
    write_overall_table(OUTPUT_DIR / "energy_usage_overall_table.tex", summary)
    write_environment_table(OUTPUT_DIR / "energy_usage_by_environment_table.tex", envs, matrix)
    plot_overall_bar(OUTPUT_DIR / "energy_usage_overall_bar.png", summary)
    plot_environment_heatmap(OUTPUT_DIR / "energy_usage_by_environment_heatmap.png", envs, matrix)

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
