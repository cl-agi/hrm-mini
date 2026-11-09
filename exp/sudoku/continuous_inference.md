# Continuous Inference

[Back to sudoku](README.md)

## Generate continuous-inference records

```bash
# Public HRM checkpoint + public Sudoku test split, 4,096 sampled puzzles,
# 16,384 inference cycles without early stopping:
python -m hrm.analyse.sudoku.continuous_inference.runner_snapshot
# Small CPU execution check (two puzzles, five inference cycles):
python -m hrm.analyse.sudoku.continuous_inference.runner_snapshot --device cpu --max-examples 2 --max-cycles 5 --batch-size 1 --output exp/sudoku/records/continuous/smoke.json
# A locally trained RT or TRM checkpoint; explicit paths must exist:
python -m hrm.analyse.sudoku.continuous_inference.runner_snapshot /path/to/model.pt --config /path/to/model_config.json --dataset-path /path/to/data/test_hard.csv --seed SAVED_SEED --output exp/sudoku/records/continuous/rt.json
```

The default output is `exp/sudoku/records/continuous/continuous.json`, independent of the current
working directory. The module is a record generator and may be invoked before a
records directory exists; the generic analysis dispatcher intentionally requires
existing records. Only main-task HRM weights are published. Train comparison
models using the recipes listed by `hrm recipes --experiment ablations` and
keep their saved configuration adjacent to the selected checkpoint. Recipe names identify model settings; inspect their architecture fields to select HRM, RT or TRM.

With no explicit seed, a single-shard run freezes Unix seconds before inference. Use the same recorded seed, CSV and sample count for each model's paired comparison. The sample is the first `--max-examples` rows of a NumPy seeded permutation of the CSV. `--indices FILE.npy` supplies exact row IDs instead. Manual sharding requires an explicit shared seed and unique outputs; each output reports its own shard. A single process covers the entire cohort by default. Historical numerical reproduction additionally requires the original checkpoint and selected IDs; a newly sampled/public-checkpoint run is a new result.

The `.plan.json` freezes seed, checkpoint/config identity, input CSV hash and selected-ID hash. Start each inference run in a fresh output location. CSV inputs require 81-character `question` and `answer` fields and preserve leading zeros. An explicit config is loaded strictly, including separate backbone/architecture sections. Explicit paths are checked for existence.

Every inference cycle makes one native `model(carry, input_ids)` call. The input is unchanged, the carry is continuous, targets are used only for measurement, and execution continues through the configured inference budget. CUDA uses graph replay; CPU runs the same buffered transition eagerly. Each run first compares logits, carry and validity records with direct native eager execution and fails on parity errors. Parity checks run on the selected device and execution backend.

The JSON reports fixed-budget exact accuracy (all 82 tokens and board-only), independent Sudoku/clue validity, ever-solved fractions and the number of examples with **any correct-to-incorrect or valid-to-invalid reversal before that budget**. Thus a board that fails temporarily and recovers at the endpoint is still counted. The `.npz` retains per-cycle Boolean measurements, IDs and final predictions under the selected records directory. Reversal statistics use every-cycle measurements; endpoint accuracy uses the final prediction at each budget.

A longer run can verify a previous run's entire observed prefix:

```bash
python -m hrm.analyse.sudoku.continuous_inference.runner_snapshot --device cpu --max-examples 2 --max-cycles 8 --batch-size 1 --seed SAVED_SEED --prefix-reference exp/sudoku/records/continuous/smoke.npz --output exp/sudoku/records/continuous/longer.json
```

Keep checkpoint, input, seed and batch size unchanged. This reruns inference from initialization and verifies all prefix metrics and the final prefix prediction. Tests cover partial final batches, HRM/RT/TRM config loading, native CPU carry parity, prefix equality, and a solved-failed-recovered example.