# Local Jacobian

[Back to memory](README.md)

## Generate fresh records

Run this directly from the checkout to generate records (the `hrm analyse`
dispatcher deliberately requires existing records). Omitted checkpoint/data inputs
resolve the pinned public Sudoku HRM and test split. Explicit `--checkpoint`,
`--config`, and `--dataset` overrides are supported; a missing explicit input fails.

```bash
python -m hrm.analyse.dynamics.output_sensitivity.h_l_joint_transition_jacobian --selection all --probes 1 --prepare-only
python -m hrm.analyse.dynamics.output_sensitivity.h_l_joint_transition_jacobian --selection all --probes 1
```

The default output is `exp/memory/records/local_jacobian/`; `--output-dir DIR` overrides it. Preparation freezes a Unix-second seed, selected board IDs, data/checkpoint/code hashes, and settings in `run_config.json`. Execution reuses that plan. Changes require a fresh output directory. Each execution uses a fresh output directory; for a same-seed rerun use a new directory and `--seed` from the saved plan. Use `--boards 1 --selection sequential --device cpu` in place of the selection options above for a small CPU check. All 32 H updates are still computed.

This estimator uses FP32 with math attention, six L updates per H update, and 32 H updates (16 inference cycles of the 2H6L schedule). BOS tangents are zero and BOS is excluded from the output RMS. Other recurrence schedules are rejected. The four blocks are H→H, H→L, L→H, and L→L; the reported RMS gain is the square root of the mean squared probe gain.

Outputs are `per_board_probe.csv`, `summary_long.csv`, and `stage_matrix.csv` plus a completion manifest. The published best checkpoint evaluates the selected checkpoint and cohort. To reproduce multiple trained seeds, supply each independently trained checkpoint. These are local transition/Taylor analyses, distinct from the output-Jacobian triangular matrix described in the memory workflow. Numerical kernels were recovered from the original analysis; see `hrm/analyse/dynamics/output_sensitivity/LOCAL_SOURCE.json`.