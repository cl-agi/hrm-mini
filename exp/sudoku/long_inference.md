# Long Inference

[Back to sudoku](README.md)

## Evaluate one fixed checkpoint at all budgets

The public-data evaluator supports single-device and distributed execution from this checkout:

```bash
# Public HRM checkpoint and Sudoku test data; single available GPU (CPU fallback).
python -m hrm.analyse.sudoku.stable16.evaluate_distributed --max-cycles 16 64 256 1024 4096 16384 --stable-cycles 16
# Small CPU input and execution check:
python -m hrm.analyse.sudoku.stable16.evaluate_distributed --device cpu --limit 2 --batch-size 1 --max-cycles 1 16 20 --stable-cycles 16 --output-dir exp/sudoku/records/stable_stopping/smoke
# A locally trained HRM or RT, optionally overriding its saved configuration:
python -m hrm.analyse.sudoku.stable16.evaluate_distributed --checkpoint /path/to/best.pt --config /path/to/model_config.json --dataset-path /path/to/data --max-cycles 16 64 256 1024 4096 16384 --stable-cycles 16 --output-dir exp/sudoku/records/stable_stopping/run1
# Multiple GPUs on one host; change the worker count to match available devices:
NPROC_PER_NODE=2 bash scripts/distributed.sh -m hrm.analyse.sudoku.stable16.evaluate_distributed --device cuda --max-cycles 16 64 256 1024 4096 16384 --stable-cycles 16 --output-dir exp/sudoku/records/stable_stopping/distributed
```

Omitted inputs use pinned HF assets. Saved configurations naming the official public Sudoku source retain that source and its sampling policy; other saved dataset paths use the published default. Local checkpoint configurations are read adjacent to the checkpoint; omitted data uses the public default. An explicit missing path fails. Default records are `exp/sudoku/records/stable_stopping/evaluation/`, independent of the working directory. CUDA uses compilation unless `--no-compile`; CPU runs eagerly. CPU multi-process tests use the Gloo backend, while CUDA uses NCCL. All ranks require shared output storage.

`stable-cycles 16` means **16 consecutive unchanged complete predictions**, requiring at least 17 predictions. Stopping depends on prediction stability. Each budget gets the prediction from its cycle, or the previously frozen prediction if the stability rule stopped earlier. Budgets share one continuous trajectory, with no reset between budgets; finished batch slots can be refilled. `--stable-cycles 0` disables this rule. Correctness and the denominator follow the original evaluator, including the BOS token in the encoded sequence.

`run_config.json` is frozen before inference and records checkpoint hash, selected input-order hash, data options, seed and budgets. Explicit `--seed` overrides the saved evaluation seed; absent a saved seed, Unix seconds are used and broadcast from rank zero. `--limit` selects a smaller shuffled cohort; `--data-seed` controls that selection. A saved subsample seed is preserved; otherwise a new limited cohort uses the evaluation seed. The full evaluation preserves the saved test-size/shuffle policy. Each budget has a summary and ordered per-rank correctness/cycle records, checked for exact once-only coverage. `--save-predictions` additionally stores predictions. All records stay outside Git. Use a fresh directory and the recorded seeds to rerun inference from initialization.

For the inference-budget curve, keep one checkpoint per training run fixed across all budgets, selected using the prescribed **16-inference-cycle evaluation**. Run three independently trained HRMs and three RTs to estimate across-run variation. The published HRM checkpoint supports a single-run evaluation. The legacy `run_tts_curve` remains a campaign/archive utility. The training path below uses its original fixed-update controller, integrated with shared models, optimizer and inference code.

## Train the six fixed checkpoints

The checked-in `configs/stable_stopping/hrm.json` and `configs/stable_stopping/rt.json` restore the original stable-stopping configuration: 83,200 optimizer updates, 20 evaluations at 4,160-update intervals, 16 training and inference cycles, EMA 0.999, learning rate 1e-4, weight decay 1.0, and batch size 96 per GPU on eight GPUs (global batch 768). HRM uses 2H6L with two layers per stack; RT uses seven recurrent steps with four layers. Training uses full-token cross entropy and the shared AdamATan2 optimizer. Reducing GPU count changes the global batch. Use eight GPUs for the configured global batch of 768.

```bash
# Resolve public source and freeze one new training seed, without starting GPUs:
hrm train --experiment sudoku --config exp/sudoku/configs/stable_stopping/hrm.json --fresh-seed --output exp/sudoku/records/stable_stopping/hrm-run1 --prepare-only
# Run with the prepared configuration and saved seed:
bash scripts/distributed.sh -m hrm train --experiment sudoku --config exp/sudoku/records/stable_stopping/hrm-run1/resolved_config.json --output exp/sudoku/records/stable_stopping/hrm-run1
```

Repeat preparation separately for three HRM and three RT runs, using `rt.json` for RT and distinct output directories. Check that the frozen Unix-second training seeds differ; prepare sequential repetitions at least one second apart. Each training run requires a fresh directory; reusing a started directory raises an error.

All six runs share the same data draw. Default input is pinned public `sapientinc/sudoku-extreme`: shuffle each official split with seed 42, take the first 1,000 training puzzles and first 20,000 test puzzles. This is a separate draw from the released standard 1k training set. All six runs share this cohort. You may resample with `--data-seed UNIX_SECONDS` during preparation; use the same frozen data seed for all six runs and subsequent curve evaluation. Record the selected data seed with the new cohort. Explicit `--dataset-path DIR` requires `train.csv` and `test.csv` with `question`/`answer` columns and adequate sizes; the configured shuffle and prefix selection still apply.

Each run writes under `OUTPUT/run/seed_SEED/`. `training_summary.json` identifies `best_eval.checkpoint` and `final_eval`; `metrics.jsonl` preserves the exact evaluation schedule and denominator. Use the **highest fixed-16 test accuracy, latest optimizer-step tie** checkpoint for every curve budget. The original selection evaluation sets stable16, which runs all 16 cycles because 16 stability comparisons require 17 predictions. Reuse the selected checkpoint at every inference budget. `evaluation.jsonl` is also written for the common scalar analysis tools. Evaluation and selection use the same 20,000 puzzles, as in the original experiment.

## Aggregate saved curves

Create `exp/sudoku/records/stable_stopping/curve_runs.json` describing the evaluation directories and actual training seeds. For example, replace the placeholders below with your recorded seed integers and run directories:

```json
{"runs": [
  {"family": "HRM", "training_seed": 1794189001, "directory": "hrm-run1"},
  {"family": "HRM", "training_seed": 1794189002, "directory": "hrm-run2"},
  {"family": "HRM", "training_seed": 1794189003, "directory": "hrm-run3"}
]}
```

Add the three RT entries with `"family": "RT"` for the comparison. Replace the example seeds with your saved training seeds. Relative directories are resolved against this manifest. Each directory must contain the evaluator's frozen plan, combined/per-budget summaries and all per-rank `.npz` results.

```bash
hrm analyse sudoku.stable16.summarize --experiment sudoku
# Explicit record root (or use the module's --manifest FILE option):
hrm analyse sudoku.stable16.summarize --experiment sudoku --records /path/to/records
```

The resulting `curve_summary.json` reports arithmetic mean accuracy in percent and **sample SD across runs (ddof=1)**, separately for HRM and RT. It recomputes numerators, denominators and actual cycles from the shards and rejects mismatched budgets, stopping rules, ordered test cohorts, reused checkpoint hashes, repeated training seeds, and missing/inconsistent records. Defaults require three runs per declared family; `--expected-runs 1` permits an explicitly limited single-run report with SD `null`. Training independence and checkpoint selection are checked using the training and 16-cycle selection records described below.

When comparing limited subsets, pass the same `--data-seed` to every evaluation
and retain the same data source/split; independently resampling each run would
change the cohort and is rejected by this paired curve summary.

## Selection evidence

Preserve the fixed-update controller's `training_summary.json`, `metrics.jsonl`, model configuration and selected checkpoint with each curve. The generic standard trainer's `best.selection.json` uses **earliest** ties; the fixed-update stable-stopping controller uses **latest** ties. Choose the receipt check for the controller that produced the run. For new fixed-update runs, add a `training_directory` to each curve entry, pointing to the seed directory that contains `model_config.json`, `input_provenance.json`, `metrics.jsonl`, `training_summary.json` and the selected checkpoint. Relative paths resolve against `curve_runs.json`:

```json
{"family": "HRM", "training_seed": 1794191000, "directory": "hrm-run1-curve",
 "training_directory": "hrm-run1/reproduction/seed_1794191000"}
```

Use your saved training seed. Then run:

```bash
hrm analyse sudoku.stable16.summarize --experiment sudoku --require-e10-training
```

This verifies the full 83,200-step/20-evaluation accounting, eight-worker/global batch configuration, numerical model dimensions, fixed-16 EMA protocol, latest-tie selection, actual selected checkpoint SHA-256, shared training draw and identical selection/curve test cohort. The training controller records dataset order hashes before training and a weight hash at each save. Evidence files are hashed in the report. `e10_training_verified` reports agreement among the supplied training records. The report also gives `fixed16_reevaluation_correct_delta`; inspect any nonzero difference between the original selection score and current fixed-16 reevaluation. The delta quantifies numerical agreement with the selection score.

Older archives lacking input identities or checkpoint hashes remain explicitly unverified unless matching evidence is recovered. The strict check will reject a small smoke run; use the evaluator/summary without this flag for execution checks, retaining the unverified status. The generic `--require-selection` flag is separate and validates the standard trainer's earliest-tie scalar receipt.