# Command options

[Quick start](../readme.md) · [Task and analysis guides](../entry.md)

## Models and datasets

`hrm evaluate` loads published models and data from Hugging Face by default.
The model and dataset repositories are `cl-agi/sudoku`, `cl-agi/sat`,
`cl-agi/maze` and `cl-agi/cremp`. Revisions are pinned in `hrm/hub.py`.

```bash
hrm evaluate --task sudoku --ckpt /path/to/model.pt --output result.json
hrm evaluate --task sudoku --dataset-path /path/to/test.csv --output result.json
```

`--ckpt-dir` selects a directory containing `model.pt`. Local checkpoint
configuration is read from an adjacent `model_config.json` or
`resolved_config.json`; `--config` selects an explicit configuration. SAT also
reads embedded checkpoint configuration and supports `--weight-state ema`.
CREMP stores its configuration in the checkpoint.

Puzzle data directories contain `<split>.csv`; SAT directories contain
`data_4.2.npz` and `data_4.3.npz`; CREMP directories contain `<split>.npz`.
The task guides describe their schemas and split selection.

`HF_HOME` controls downloaded assets and `HRM_ASSET_CACHE` controls converted
inputs. The Docker setup places both on the persistent work volume.

## Training configurations

```bash
hrm recipes --experiment sudoku
hrm recipes --experiment sudoku --json
```

Use a listed `--recipe` or supply `--config FILE`. `--prepare-only` resolves inputs
and saves the run configuration. `--fresh-seed` selects one new timestamp-seeded
repetition; subsequent execution reuses the saved seed. Use a new output directory
for each independent run. `--dataset-path` and `--eval-dataset-path` override
training and evaluation inputs independently on the shared training entry point.
Task-specific entry points document their path options in their guides.
Shared training entry points save checkpoints under
`OUTPUT/<run_name or run>/seed_SEED/`. Set `run_name` in the configuration to
choose the group directory.

## Evaluation and analysis

`--limit` selects a small evaluation subset; SAT applies it per density and CREMP
per molecule. Use the task guide's complete-split command for a full evaluation.
CREMP evaluation consists of generation, coordinate reconstruction and scoring,
with all three commands in the [CREMP guide](../exp/cremp/README.md).

Analysis guides provide commands to generate their input records and compute
statistics. The dispatcher uses `exp/<topic>/records/<stage>/` by default;
`--records PATH` overrides it. Generate the records first; missing inputs produce
an error. See [record layouts](../exp/README.md#analysis-records).

## Tests

```bash
python -m unittest discover -s tests -q
```
