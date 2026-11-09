# Experiment guide

Follow the three steps in the [README](readme.md), then select a workflow below.
Run commands from the repository root. Each guide includes its training,
evaluation or record-generation commands and required dependencies.

## Tasks

| Task | Workflow |
|---|---|
| [Sudoku](exp/sudoku/README.md) | Train HRM, evaluate checkpoints and vary the inference budget |
| [SAT and graph coloring](exp/sat/README.md) | Train and evaluate SAT models; evaluate graph coloring |
| [Maze](exp/maze/README.md) | Train on mazes and evaluate valid paths across sizes |
| [Molecular conformations](exp/cremp/README.md) | Train, sample conformations, reconstruct coordinates and score |

## Comparisons and analyses

| Workflow | What it runs |
|---|---|
| [Architecture and training ablations](exp/ablations/README.md) | Timescale depth, shared modules, update frequencies and supervision |
| [Search dynamics](exp/dynamics/README.md) | Trajectories, displacement statistics and perturbation recovery |
| [Memory](exp/memory/README.md) | State sensitivity and Jacobian analyses |
| [Error readout and intervention](exp/error_readout/README.md) | Error predictors and interventions on model states |
| [Generalization](exp/generalization/README.md) | Clue edits and difficulty transfer |

Use `hrm list` to list workflows and `hrm recipes --experiment sudoku` to list
training configurations. See [experiment usage](exp/README.md) for configuration,
seed and record-directory options.
