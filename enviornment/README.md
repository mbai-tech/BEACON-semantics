# Environment Generator

This folder generates planning environments as both JSON scene files and PNG
images.

## Setup

From the repo root:

```bash
cd /Users/ishita/Documents/GitHub/SURP/enviornment
python3 -m venv .venv-mac
source .venv-mac/bin/activate
pip install -r requirements.txt
```

## Generate All Families

To generate 100 scenes for each family:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run.py
```

This writes files to:

- `data/images/<family>/`
- `data/scenes/<family>/`

Important:

- Running `run.py` clears the existing files in each family subfolder first.
- Output filenames include the seed, so each file can be reproduced later.

Example output names:

```text
data/images/sparse_clutter/000_seed123456789.png
data/scenes/sparse_clutter/000_seed123456789.json
```

## Generate One Family

To generate one scene for a single family:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter
```

Valid family names:

- `sparse_clutter`
- `dense_clutter`
- `narrow_passage`
- `semantic_trap`
- `perturbed`

This writes to:

- `data/images/<family>/scene.png`
- `data/scenes/<family>/scene.json`

Running `run_family.py` without a seed also clears the normal `data/` output
folders first.

## Save A Specific Seed

To regenerate the exact same environment from a known seed and save it in a
separate folder that is not cleared by `run.py`:

```bash
cd /Users/ishita/Documents/GitHub/SURP
python3 enviornment/run_family.py sparse_clutter 12345
```

This writes to:

- `enviornment/saved_enviornments/images/`
- `enviornment/saved_enviornments/scenes/`

Example saved files:

```text
enviornment/saved_enviornments/images/sparse_clutter/seed12345.png
enviornment/saved_enviornments/scenes/sparse_clutter/seed12345.json
```

These saved environments are not deleted when you rerun `python3 enviornment/run.py`.

If you save the same seed again, a numbered suffix is added instead of
overwriting the previous saved copy.

## Clear Saved Environments

To clear every saved scene and image created under
`enviornment/saved_enviornments/`:

```bash
python3 enviornment/clear_saved_envs.py
```

To clear saved files for only one family:

```bash
python3 enviornment/clear_saved_envs.py --family sparse_clutter
```

## Reproduce From A Seed In Batch Output

If you see a batch-generated filename like:

```text
sparse_clutter_003_seed1989186592.json
```

you can reproduce that same scene with:

```bash
python3 enviornment/run_family.py sparse_clutter 1989186592
```

## Run The D* Lite Baseline

`Baselines/baseline.py` plans over the scene JSON files produced in this folder. It
discretizes the workspace into a grid, runs D* Lite on that grid, and writes a
path JSON whose waypoints keep the scene's 3D `[x, y, z]` format.

## Generate A Scene With `baseline.py`

You can also create a scene for a specific family directly from the baseline
CLI:

```bash
python3 Baselines/baseline.py generate-scene \
  --family sparse_clutter \
  --seed 12345
```

By default this writes:

- `data/scenes/sparse_clutter/seed12345.json`
- `data/images/sparse_clutter/seed12345.png`

Supported families:

- `sparse_clutter`
- `dense_clutter`
- `narrow_passage`
- `semantic_trap`
- `perturbed`

Example:

```bash
python3 Baselines/baseline.py plan \
  --scene data/scenes/sparse_clutter/seed12345.json
```

If you omit `--output`, `baseline.py` now creates the folder automatically and
saves to:

- `data/plans/<family>/<scene_name>_dstar.json`

This produces a JSON file containing:

- planner metadata
- success/failure status
- path length and waypoint count
- a `path` array with 3D waypoints

Useful flags:

- `--mode collision_free`
- `--mode contact_allowed`
- `--step 0.15`
- `--robot-radius 0.12`

## Render The Planned Path To PNG

After generating a plan JSON, you can draw the path on top of the original
scene:

```bash
python3 Baselines/baseline.py render \
  --scene data/scenes/sparse_clutter/seed12345.json \
  --plan data/plans/sparse_clutter/seed12345_dstar.json
```

If you omit `--output`, `baseline.py` automatically creates:

- `data/plan_images/<family>/<scene_name>_dstar.png`

That command creates a PNG overlay with:

- the original obstacles
- the start and goal
- the D* Lite path
