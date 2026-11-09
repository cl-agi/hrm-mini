# Search dynamics and timescales

[Experiment guide](../../entry.md)

This workflow examines local refinement and continued exploration by H/L. The primary numerical analysis is MSD; trajectory comparisons and conditional updates provide supporting diagnostics.

## MSD

```bash
python -m hrm.analyse.dynamics.msd.collect --prepare-only
python -m hrm.analyse.dynamics.msd.collect
```

The collector loads the public HRM checkpoint and Sudoku puzzles, computes full-state displacement curves, and writes statistical summaries under `exp/dynamics/records/msd/`. See [MSD calculation and commands](msd.md) for checkpoint overrides, sampling and short tests.

## Trajectory generation and search comparisons

```bash
python -m hrm.analyse.dynamics.perturbation_recovery.native
```

This small native-trajectory entry defaults to public HF model/data and writes under `exp/dynamics/records/recurrence/`. The resulting trajectory describes the sampled puzzle and selected checkpoint. [Same-puzzle search comparison](search.md) documents HRM, DFS and simulated annealing with shared inputs.

Other retained modules support H-update and historical-distance controls. Their outputs are supporting records within this workflow. Long recovery and perturbation analyses require their corresponding trajectory and intervention records.
