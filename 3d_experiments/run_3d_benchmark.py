"""Run Bug1, Bug2, D* Lite, and BEACON on the 3D-environment scene families.

"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from beacon.environment.scene_generator_shapely import generate_scene
from beacon.core.bug_algorithm        import run_bug
from beacon.core.bug2_algorithm       import run_bug2
from beacon.core.dstar_lite_algorithm import run_dstar_lite
from beacon.planning.beacon_planner   import run_beacon

FAMILIES = [
    "sparse_clutter",
    "dense_clutter",
    "narrow_passage",
    "semantic_trap",
]

ALGORITHMS = {
    "Bug1":    run_bug,
    "Bug2":    run_bug2,
    "D* Lite": run_dstar_lite,
    "BEACON":  run_beacon,
}


def _path_length(path: list) -> float:
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        total += (dx ** 2 + dy ** 2) ** 0.5
    return round(total, 3)


def run_one(scene: dict, alg_name: str, alg_fn) -> dict:
    t0 = time.perf_counter()
    result = alg_fn(scene)
    elapsed = time.perf_counter() - t0

    damage  = 0.0
    replans = 0
    if result.scene_summary is not None:
        damage  = result.scene_summary.total_semantic_damage
        replans = result.scene_summary.n_cibp_replans

    return {
        "algorithm": alg_name,
        "success":   result.success,
        "path_len":  _path_length(result.path),
        "damage":    damage,
        "replans":   replans,
        "time_s":    round(elapsed, 3),
    }


def run_family(family: str, n_scenes: int, seed_offset: int = 0) -> list[dict]:
    rows = []
    for i in range(n_scenes):
        scene = generate_scene(family=family, seed=seed_offset + i)
        for alg_name, alg_fn in ALGORITHMS.items():
            row = run_one(scene, alg_name, alg_fn)
            row["family"] = family
            row["scene"]  = i
            rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    from collections import defaultdict
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r["family"], r["algorithm"])].append(r)

    families  = list(dict.fromkeys(r["family"]    for r in rows))
    alg_names = list(dict.fromkeys(r["algorithm"] for r in rows))

    header = f"{'Family / Algorithm':<30}  {'Success':>8}  {'Path Len':>9}  {'Damage':>8}  {'Replans':>8}  {'Time(s)':>8}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for fam in families:
        print(f"\n  {fam}")
        for alg in alg_names:
            bucket = agg[(fam, alg)]
            if not bucket:
                continue
            n       = len(bucket)
            suc_pct = 100 * sum(r["success"] for r in bucket) / n
            avg_len = sum(r["path_len"] for r in bucket) / n
            avg_dmg = sum(r["damage"]   for r in bucket) / n
            avg_rep = sum(r["replans"]  for r in bucket) / n
            avg_t   = sum(r["time_s"]   for r in bucket) / n
            print(
                f"    {alg:<26}  {suc_pct:>7.1f}%  {avg_len:>9.2f}"
                f"  {avg_dmg:>8.2f}  {avg_rep:>8.2f}  {avg_t:>8.3f}s"
            )

    print()
    print("=" * len(header))
    print("\nOverall (all families):")
    for alg in alg_names:
        all_rows = [r for r in rows if r["algorithm"] == alg]
        n       = len(all_rows)
        suc     = 100 * sum(r["success"]  for r in all_rows) / n
        avg_len = sum(r["path_len"] for r in all_rows) / n
        avg_dmg = sum(r["damage"]   for r in all_rows) / n
        avg_rep = sum(r["replans"]  for r in all_rows) / n
        print(
            f"  {alg:<10}  success={suc:.1f}%  "
            f"path_len={avg_len:.2f}  damage={avg_dmg:.2f}  replans={avg_rep:.2f}"
        )
    print()


def show_pybullet(scene: dict) -> None:
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("[WARN] pybullet not installed — skipping 3D viewer")
        return

    sys.path.insert(0, str(_ROOT / "enviornment"))
    from render_scene_pybullet import setup_pybullet_scene

    for obs in scene["obstacles"]:
        if "class_true" not in obs and "true_class" in obs:
            obs["class_true"] = obs["true_class"]

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    setup_pybullet_scene(scene)

    print("PyBullet 3D viewer open — press Q to quit.")
    while True:
        keys = p.getKeyboardEvents()
        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes",   type=int, default=5, help="Scenes per family")
    parser.add_argument("--family",   type=str, default=None,
                        choices=FAMILIES, help="Run one family only")
    parser.add_argument("--pybullet", action="store_true",
                        help="Open PyBullet 3D viewer for first scene after benchmark")
    args = parser.parse_args()

    families = [args.family] if args.family else FAMILIES

    print(f"\nRunning {', '.join(ALGORITHMS)} on {args.scenes} scenes × {len(families)} families …")

    all_rows: list[dict] = []
    for fam in families:
        print(f"  {fam} …", end=" ", flush=True)
        rows = run_family(fam, args.scenes)
        all_rows.extend(rows)
        print("done")

    print_table(all_rows)

    if args.pybullet:
        first_scene = generate_scene(family=families[0], seed=0)
        show_pybullet(first_scene)
