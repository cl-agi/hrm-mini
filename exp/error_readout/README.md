# Error readout and intervention

[Experiment guide](../../entry.md)

The commands below generate model states, fit readouts and evaluate interventions. The experiment asks whether constraint/error information is readable from H/L and whether interventions along those directions change subsequent dynamics. Configure the task, target error and perturbed module within this workflow.

| Choice | Supported analysis families |
|---|---|
| Task | Sudoku; SAT as a cross-task extension |
| Target | Constraint violations; answer CE/MSE as alternative targets |
| State | H, L or joint readout; H- or L-direction interventions |
| Control | Matched random displacements; native/no-intervention branch |

## Stages

1. Collect states from a frozen checkpoint and split by whole puzzle/formula.
2. Fit the readout on training instances; select settings on validation instances.
3. Evaluate held-out readout performance and apply the specified intervention with matched controls.
4. Summarize next-update dynamics and task performance separately, with instances as the statistical unit.

The implementation families are `hrm.analyse.dynamics.constraint_readout`, `hrm.analyse.dynamics.constraint_intervention`, `hrm.analyse.dynamics.answer_error`, and `hrm.analyse.sat.constraint_intervention`. Each module implements its target-specific calculations on these records.

## Collect and fit constraint readouts

```bash
python -m hrm.analyse.dynamics.constraint_readout.run --prepare-only
python -m hrm.analyse.dynamics.constraint_readout.run
hrm analyse dynamics.constraint_readout.summarize --experiment error_readout
hrm analyse dynamics.constraint_readout.audit_definition --experiment error_readout
```

The collector defaults to the public HF Sudoku checkpoint and test puzzles. `--checkpoint`, `--config`, and `--dataset` select local inputs. Preparation freezes a Unix-second seed and the sampled row IDs. The default 512-puzzle cohort is split by whole puzzle into 256 training, 128 validation and 128 held-out puzzles. The affine OLS readout uses the training partition. Each puzzle contributes 192 pre-L-update observations over 16 inference cycles with the checkpoint's 2H6L schedule. H/L features are averaged over positions; a read-only H pass supplies digit probabilities for soft and hard constraint targets. Whole-puzzle resampling supplies uncertainty and permutation statistics.

Outputs are `results.json`, `predictions.npz`, `probes.pt` and run metadata under `exp/error_readout/records/sudoku_answers/hrm/analyse/dynamics/constraint_readout/`. `--output-dir` selects another directory; use `python -m hrm.analyse.dynamics.constraint_readout.summarize --records OUTPUT` for that output. Undefined correlations are stored as null.

For a short CPU check, use a fresh output directory and `--device cpu --boards 8 --cycles 1 --bootstrap-puzzles 20 --permutation-puzzles 19`. Full statistical runs use the defaults above.

## Analyze existing records

```bash
hrm analyse dynamics.constraint_readout.summarize --experiment error_readout
hrm analyse sat.constraint_intervention.summarize --experiment error_readout
```

Defaults route Sudoku answer/readout records to `exp/error_readout/records/sudoku_answers/`, Sudoku population/dose records to `records/sudoku_constraints/`, and SAT records to `records/sat_constraints/`. Legacy analyzers retain their module-relative file layout within each root; see [records conventions](../README.md#analysis-records). Use `--records /path/to/records` to override the root. Use the generated records from the corresponding collection command.

These commands summarize previously generated records. CE/MSE targets, dose scans and frozen-H controls are optional settings.

Answer-error report inputs use `intervention_statistics.json`; summary records store its contents under `intervention_statistics`. Reports are written to `REPORT.md`.

## Generate answer-error readouts

```bash
python -m hrm.analyse.dynamics.answer_error.collect_fit --prepare-only
python -m hrm.analyse.dynamics.answer_error.collect_fit
```

This collects native pre-L states from the public Sudoku checkpoint and data, then fits CE and probability-MSE readouts. Whole puzzles are split 50%/25%/25% for training, validation and test. Linear probes use unregularized affine OLS; three MLP probes use validation early stopping. Prepared metadata records the cohort and training seeds. Outputs are numerical metrics, fitted probes and generated observations under `exp/error_readout/records/answer_error/`. Local overrides are `--checkpoint`, `--config` and `--dataset`. A short check uses `--boards 8 --cycles 1 --epochs 1 --bootstrap 2`.

## Generate Sudoku count interventions

```bash
python -m hrm.analyse.dynamics.constraint_intervention.intervention --prepare-only
python -m hrm.analyse.dynamics.constraint_intervention.intervention
```

The command fits an affine duplicate-excess probe from L states on training puzzles, computes the training feature covariance, and evaluates a disjoint held-out cohort after four H updates. Eligible states have a nonnegative-clipped probe score of at least one. Fractional score changes use covariance-preconditioned tangent directions and four signed random controls with matched token norms and displacement; every branch checks its score and geometry tolerances. The output records included/excluded puzzle IDs and next-update norms. Defaults use 256 training puzzles, 128 test puzzles and eight collection cycles. `--train-boards`, `--test-boards`, `--fit-cycles`, `--device` and local asset overrides configure a new run. Results are written under `exp/error_readout/records/sudoku_constraints/`.

## Generate SAT readouts and interventions

```bash
python -m hrm.analyse.sat.constraint_intervention.prepare
CUDA_VISIBLE_DEVICES=0 python -m hrm.analyse.sat.constraint_intervention.preflight --parity
CUDA_VISIBLE_DEVICES=0 python -m hrm.analyse.sat.constraint_intervention.preflight --precision --alphas .002 .01 .03
python -m hrm.analyse.sat.constraint_intervention.run_experiment --gpus 0
python -m hrm.analyse.sat.constraint_intervention.summarize
```

Preparation resolves the public SAT checkpoint and train/validation/test formulas, freezes seeds and hashes, and saves the analysis protocol. Parity and precision checks run on an idle visible GPU; subsequent stages require both checks to pass. The unchanged analysis uses FP32 native trajectories, validation-selected ridge H/L probes and matched signed interventions after the fourth H update. Defaults use 64 inference cycles. Preparation accepts `--checkpoint`, `--config` and `--data-root`; `--limit 2 --cycles 3` prepares a small execution check. Set `HRM_ANALYSIS_RECORDS` to select a separate run root; the default is `exp/error_readout/records/sat_constraints/`.


Summarize the generated Sudoku interventions with:

```bash
python -m hrm.analyse.dynamics.constraint_intervention.analyze_intervention
```

For population dynamics, collect and summarize with:

```bash
python -m hrm.analyse.dynamics.constraint_intervention.population
python -m hrm.analyse.dynamics.constraint_intervention.population_readout
```

The population collector defaults to 1,056 puzzles and accepts `--boards`, `--batch-size`, `--device` and local input overrides. The numerical summary fits and scores the population OLS readout on these same observations and reports whole-puzzle bootstrap intervals. Output directories can be selected with `--output` for collection and `--records` for statistics.
