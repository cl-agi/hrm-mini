# Mean squared displacement

[Back to dynamics](README.md)

## Compute and summarize

```bash
python -m hrm.analyse.dynamics.msd.collect --prepare-only
python -m hrm.analyse.dynamics.msd.collect
```

The default run loads the public HF HRM Sudoku checkpoint and test split, samples 256 puzzles, and computes full-state MSD over 4,096 L updates. Preparation freezes the sampling seed, puzzle row IDs, checkpoint hash and configuration. The run writes curves under `exp/dynamics/records/msd/rollouts/` and calls the statistics module to write `statistics/` in the same directory.

Use `--checkpoint /path/to/model.pt`, `--config /path/to/model_config.json`, and `--dataset /path/to/test.csv` for local inputs. An explicit checkpoint uses its adjacent configuration by default. Supported architectures are HRM, shared-weight TRM, equal-frequency H2L1 HRM, and RT. Each checkpoint supplies its own weights and architecture. `--output-dir` selects the run directory, `--boards` sets the cohort size, and `--batch-size` controls working memory. The default batch size is one puzzle. A standard-width HRM's largest window uses approximately 1 GiB of CPU state buffers per puzzle, in addition to model and runtime memory. State buffers are released after each window; saved outputs contain numeric curves, statistical summaries and metadata.

A short CPU run:

```bash
python -m hrm.analyse.dynamics.msd.collect \
  --device cpu --boards 1 --boundaries 0 12 24 36 48 \
  --output-dir exp/dynamics/records/msd-small
```

Custom windows use the `available` fit profile: log2–log2 OLS over the available positive MSD values at lags of at least six updates. The standard run uses all four profiles below. Short runs check the complete execution path; long-run statistics use the full horizon and cohort.

## Calculation

State zero is the initialized carry. Each subsequent clock tick completes one L update and, at the configured L interval, an H update. RT advances its recurrent core once per tick. H stays constant between its updates. For every lag, MSD averages squared differences over all valid time origins within the window, puzzles, token positions (including BOS), and hidden coordinates. Both endpoints lie in the same half-open window; a window of length N has N − lag origins. Curves use equal puzzle weighting across batches.

Default windows are `[0,48)`, `[48,192)`, `[192,768)` and `[768,4096)`. The standard profiles use a shared lag grid:

- `common6_47`: 21 lags from 6 through 47, in all windows.
- `common6_24`: 12 lags from 6 through 22, in all windows.
- `late6_3327`: 39 lags from 6 through 3327, in the last window.
- `archived`: condition-specific minimum lag, equal to the L update interval.

Each checkpoint is fitted separately with a free intercept and equal weight per lag point. Curves are normalized by their own MSD(6) before aggregation. H–L contrasts pair states from the same checkpoint and window. Standard deviation uses `ddof=1` across independent checkpoints and is undefined for one checkpoint. Zero MSD is retained in curves and omitted from logarithmic fitting with explicit counts.

## Inputs and outputs

`rollout_metadata.csv` records the checkpoint identity, known training seed, sampling seed, cohort size, update schedule, windows and lag grid. `msd_curves.csv` contains one mean for each checkpoint/state/window/lag. Fresh records use checkpoint identity as the replication unit; resampling a checkpoint leaves that identity unchanged. The default public checkpoint and cohort are recorded in `run_config.json`. For an Easy-k55 comparison, provide the corresponding trained checkpoints and evaluation cohort through the same local overrides.

The statistics stage can also be run separately:

```bash
python -m hrm.analyse.dynamics.msd.analyze \
  --records /path/to/rollouts --output-dir /path/to/new/statistics
# Custom short windows:
python -m hrm.analyse.dynamics.msd.analyze \
  --records /path/to/short/rollouts --output-dir /path/to/new/statistics \
  --profile available
```

Outputs include `msd_curves.csv`, `msd_curve_summary.csv`, `beta_per_seed.csv`, `beta_summary.csv`, and, for H/L models, `delta_per_seed.csv` and `delta_summary.csv`. The per-seed filenames also accept fresh checkpoint-indexed results; those rows identify `checkpoint_id`. `analysis_manifest.json` records the replication unit, input hashes, fit profiles and output hashes. Legacy curve inputs retain their standard-window, epoch-19 selection and training-seed checks.
