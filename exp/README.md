# Experiments

Use the [experiment guide](../entry.md). This directory has four task workflows, one shared ablation workflow and four analysis workflows. `index.json` defines their display order; each named directory contains its guide and module/configuration index. Models and training algorithms remain in the shared package.

## Training and data

```bash
hrm list
hrm recipes --experiment sudoku
hrm train --experiment sudoku --recipe RECIPE_KEY --fresh-seed --output exp/sudoku/records/my-run --prepare-only
```

Replace `RECIPE_KEY` with a listed key. Remove `--prepare-only` and use the documented distributed launcher to train. SAT, Maze and CREMP retain their task-specific launchers; follow their guides. Omitted data paths use HF or the documented public upstream dataset. Explicit `--dataset-path` and `--eval-dataset-path` take priority and must exist.

`recipes.json` stores shared configurations, including retained comparison variants. Names identify the task, model and settings; `-variant-NN` distinguishes saved configurations sharing the displayed settings. Seeds are stored in each configuration. New repetitions use `--fresh-seed` to freeze a Unix-second seed; reusing a prepared run retains it. Use this option to prepare a new repetition when a configuration lacks a seed.

For sampled cohorts, retain the source population, filters, sample size and train/test separation. Record the sampling seed and source revision for each new sample. Fixed benchmarks retain their defined splits.

## Analysis records

```bash
hrm analyse dynamics.msd.analyze --experiment dynamics
hrm analyse dynamics.msd.analyze --experiment dynamics --records /path/to/msd-records
```

The dispatcher selects the named stage under `exp/<topic>/records/`, such as `exp/dynamics/records/msd/`. Modules may expect a further input subdirectory: MSD reads `rollouts/` within that root. Explicit `--records` overrides the stage root. Prepare the required input records before invoking the analyzer; missing or empty inputs cause an error.

Each analyzer reads the layout specified in its guide within the selected records root. Some use subdirectories such as `data/...`, `checkpoint/...` and `hrm/analyse/<module>/...`. Use absolute paths for explicit module-specific overrides. Use the input schema required by each analysis module.

Training runs should write under `exp/<topic>/records/<run>/`. Shared controllers save `evaluation.jsonl` with seed, optimizer step, denominator, inference budget and EMA/raw weight type. Summarize existing runs with:

```bash
hrm analyse training_summary --experiment sudoku --split test_hard
```

Select `--cycles` if records contain multiple budgets. The summary reports both peak and last observations. Check the run completion record for the completed training budget. Specialized trainers may use their own records formats.

## Compatibility

`compatibility.json` supports legacy `E##` identifiers. Existing nonempty `exp/E##/records/` directories remain usable through those identifiers; new default paths use named topics and stages. Old configuration files have moved into the corresponding topic under `configs/`. Use the updated task guides for their paths. Existing record directories remain available.

Only code, configuration and documentation are tracked. Records, trajectories, generated data and weights stay outside Git. Compatibility metadata records the original experimental protocols.
