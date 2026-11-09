# Sudoku

[Experiment guide](../../entry.md)

Train and evaluate HRM on Sudoku. Model variants use the shared [ablation configurations](../ablations/README.md).

## Evaluate the published checkpoint

```bash
hrm evaluate --task sudoku --device cpu --limit 1 --cycles 1 --output exp/sudoku/records/smoke.json
hrm evaluate --task sudoku --cycles 16 --output exp/sudoku/records/test.json
```

Omitted model/data paths resolve from HF. Explicit `--ckpt` and `--dataset-path` take priority. The full public test split has 20,000 puzzles. Small checks verify execution on the selected subset.

## Train

```bash
hrm train --experiment sudoku \
  --recipe sudoku-hrm \
  --fresh-seed --output exp/sudoku/records/train --prepare-only
bash scripts/distributed.sh -m hrm train --experiment sudoku \
  --recipe sudoku-hrm \
  --fresh-seed --output exp/sudoku/records/train
```

Preparation freezes the seed and resolves public data. Other recipe settings are retained. Use the same output directory for the prepared run. Model selection and evaluation must use the same split, denominator, weight type and inference budget as the comparison.

## Inference budgets and stopping

[Long inference with stable stopping](long_inference.md) and [continuous inference](continuous_inference.md) evaluate the same trained checkpoint across inference budgets.

## Optional comparisons

RT and shared-H/L variants use the shared training package. IRED and Qwen use `third_party/ired` and `third_party/qwen`; frontier-model evaluations require API access. These methods have separate dependencies and may require locally trained checkpoints. Saved run records provide timing and steps-to-peak statistics.
## IRED baselines

Install the main checkout and the IRED dependencies in the selected environment:

```bash
python -m pip install -e . -e ./third_party/ired
python -m hrm.analyse.baselines.ired.train_extreme_aug \
  --initialization scratch --output "$HRM_WORKDIR/ired-extreme" --prepare-only
python -m hrm.analyse.baselines.ired.train_extreme_aug \
  --initialization scratch --output "$HRM_WORKDIR/ired-extreme"
```

Preparation resolves the public Extreme-1k training data and freezes the seed. Execution trains for 50,000 updates and writes `latest.pt`; it then evaluates its final EMA weights. `--data-path DIR` supplies local CSVs. Pretrained initialization additionally requires `--initialization pretrained --initial-checkpoint FILE`.

For the public SAT-Net/RRN training protocol:

```bash
python -m hrm.analyse.baselines.ired.run \
  --output "$HRM_WORKDIR/ired-satnet" --steps 60000 --prepare-only
python -m hrm.analyse.baselines.ired.run \
  --output "$HRM_WORKDIR/ired-satnet" --steps 60000 --train-only
```

`--satnet-dir` and `--rrn-dir` override the upstream input directories. Use `--help` on `hrm.analyse.baselines.ired.eval_extreme` for explicit-checkpoint evaluation, shard assignment and merging. Match `--expected-step` and `--training-description` to the checkpoint's actual training run.
