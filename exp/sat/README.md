# SAT and graph coloring

[Experiment guide](../../entry.md)

## Fresh training and local-checkpoint evaluation

The training configuration retains the experiment's architecture and update
budget, but omits the historical seed. Preparation saves a new Unix-second seed
in `training_seed.json` and the complete configuration in `resolved_config.json`.
Reusing a prepared output preserves its seed; changing the configuration requires
a new output directory. Explicit seeds in a custom configuration remain supported
for same-seed replays.

```bash
hrm train-sat --config exp/sat/configs/inference/train.json --output exp/sat/records/inference/hrm_run01 --prepare-only
bash scripts/distributed.sh -m hrm.cli train-sat --config exp/sat/configs/inference/train.json --output exp/sat/records/inference/hrm_run01

hrm evaluate --task sat --checkpoint exp/sat/records/inference/hrm_run01/weights/s61440.pt \
  --weight-state raw --n-vars 1024 --cycles 1024 --starts 64 --output exp/sat/records/inference/hrm_run01/public1024.json
```

Omitted training data resolves to the published HF splits. `--data-root` overrides
training data; evaluation has its own optional `--dataset-path` public-test
bundle. Raw weights are the default and match the reported raw-checkpoint
protocol. `--weight-state ema` explicitly chooses the EMA parameters from a
bundled training checkpoint; requesting EMA from a plain weight file is an error.
Evaluation finds configuration embedded in `latest.pt`, adjacent to a local
checkpoint, or one level above `weights/s*.pt`. `--config` explicitly overrides
this lookup. The model configuration determines the architecture family reported
in the results.

The example checkpoint exists only after that training step and validation have completed. Full-test accuracy requires completing this evaluation. The published HF model remains the default when no checkpoint is supplied.

## Graph coloring and optional baselines

SAT weights also support graph coloring through the retained `hrm.analyse.sat.graph_coloring` analysis package. QuerySAT and NeuroSAT are optional third-party comparisons, with separate dependencies and weights. Graph-coloring data preparation and inference budgets are task settings; no additional HRM model release is required. Records belong under `exp/sat/records/graph_coloring/`.

When overriding evaluation settings with `--config`, use the prepared `resolved_config.json`, which includes the frozen training seed. The training template requires preparation before evaluation.
## Public graph-coloring baseline evaluation

NeuroSAT loads its public pretrained weights and the HF graph split:

```bash
python -m pip install -e . -e ./third_party/neurosat
python -m hrm.analyse.sat.graph_coloring.evaluate_neurosat \
  --device cpu --iterations 3 --limit 1 \
  --output "$HRM_WORKDIR/neurosat-graph-smoke" --prepare-only
python -m hrm.analyse.sat.graph_coloring.evaluate_neurosat \
  --device cpu --iterations 3 --limit 1 \
  --output "$HRM_WORKDIR/neurosat-graph-smoke"
```

For the complete generated-graph protocol use `--iterations 2048`, omit `--limit`, and choose a new output directory. `--benchmark flat200` selects the Flat200 split. `--checkpoint FILE` and `--dataset-path FILE.parquet` override weights and inputs.

QuerySAT uses its separate Python 3.11 environment. From the checkout root:

```bash
/path/to/querysat-venv/bin/python -m pip install -e './third_party/querysat[evaluation]'
PYTHONPATH="$PWD" /path/to/querysat-venv/bin/python \
  -m hrm.analyse.sat.graph_coloring.evaluate_querysat \
  --checkpoint /path/to/querysat/checkpoint --device cpu --limit 1 \
  --rounds 3 --starts 1 --output "$HRM_WORKDIR/querysat-graph-smoke" --prepare-only
```

This command imports the lightweight evaluation adapter directly from the checkout. Remove `--prepare-only` to evaluate; the full protocol uses `--rounds 2048 --starts 4` and all cases. Supply a trained TensorFlow checkpoint prefix or checkpoint directory. Generated graphs and Flat200 use the same HF data adapter.

HRM graph-coloring evaluation loads the published SAT checkpoint and reconstructs the CNF from public graph records:

```bash
python -m hrm.analyse.sat.graph_coloring.recurrent --colors 3 --output exp/sat/records/graph_coloring/hrm3
python -m hrm.analyse.sat.graph_coloring.recurrent --colors 4 --output exp/sat/records/graph_coloring/hrm4
```

Defaults use 1,024 inference cycles and 64 starts. A short check adds `--limit 1 --cycles 1 --starts 1 --device cpu`. Local overrides are `--checkpoint`, `--config` (saved training configuration) and `--dataset-path` (published-schema Parquet). `--prepare-only` freezes the configuration and input identities. SAT training uses `hrm train-sat`.
