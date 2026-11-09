# Maze

[Experiment guide](../../entry.md)

## Published mixed-size inputs

The public `cl-agi/maze` training split already contains 1,000 selected
examples: 250 each at sizes 15, 20, 25 and 30. Use this selected split directly. The test split contains 1,000 examples at each of
15/20/25/30/32/35/40/45/50/60/80; `test_control` is the separate 30-size control.

Check the pinned public inputs (downloads only on the machine running this command):

```bash
python -m hrm.training.maze_data --split train
python -m hrm.training.maze_data --split test
python -m hrm.training.maze_data --split test_control
```

`--dataset-path` accepts a local Parquet file, or a directory containing
`train.parquet`, `test.parquet` and `test_control.parquet`. Explicit missing
inputs cause an error.

The Python `hrm.training.maze_data.create_train_loader(seed=...)` adapter
uses the existing size-bucket sampler: 25 repetitions, 32 global batches,
768 examples per global batch, fixed ordering across epochs. The caller must
supply a saved run seed. Use the launcher below for fresh training.

## Portable HRM/RT runs

The launcher prepares data, freezes the run configuration and starts training.
Its configurations are [hrm.json](configs/extrapolation/hrm.json) and [rt.json](configs/extrapolation/rt.json).
Each run uses a seed frozen from Unix seconds at preparation. Separate
repetitions must be prepared in separate directories at different seconds.
Repeated preparation verifies the same saved seed, inputs, configuration and
source hashes. Local overrides require a directory containing all three splits.

```bash
python -m hrm.training.maze prepare --family HRM --run-dir exp/maze/records/extrapolation/hrm_run01
python -m hrm.training.maze prepare --family RT --run-dir exp/maze/records/extrapolation/rt_run01

bash scripts/distributed.sh -m hrm.training.maze train --run-dir exp/maze/records/extrapolation/hrm_run01
bash scripts/distributed.sh -m hrm.training.maze train --run-dir exp/maze/records/extrapolation/rt_run01

python -m hrm.training.maze evaluate --run-dir exp/maze/records/extrapolation/hrm_run01 --eval-batch-size 1
hrm analyse training_summary --experiment maze --split test_80
```

Training uses 20 epochs × 32 batches × 16 training cycles = 10,240 optimizer
updates. Both models use original 2D RoPE. Evaluation uses 16 inference cycles,
legal shortest-path accuracy, and all 12 test cohorts at the final epoch.
`--eval-batch-size` affects evaluation memory only. `last.pt` contains EMA
weights; `final_metrics.json` is written only after final evaluation. Checkpoint selection uses the final epoch. To reproduce
the ten-run comparison, prepare and train ten independent runs per architecture.

Start each training run in a fresh directory. The launcher rejects a directory
containing `training_started.json`. Preserve interrupted runs and prepare a new
directory for a fresh start.
