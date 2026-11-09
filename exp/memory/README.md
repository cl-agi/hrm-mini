# Memory and sensitivity

[Experiment guide](../../entry.md)

Measure how earlier H/L states influence later predictions. Local Jacobians and removal of the H-to-H link are supporting variants of this analysis.

## Output sensitivity

```bash
python -m hrm.analyse.dynamics.jacobian.cli \
  --checkpoint /path/to/checkpoint.pt --config /path/to/model_config.json \
  --dataset /path/to/test.csv --output-dir exp/memory/records/sensitivity/new-run
```

This standalone estimator currently requires explicit checkpoint and dataset paths. Use compatible HRM/RT weights and the corresponding saved architecture configuration. Its `--help` describes board selection, probes and inference horizon. Use the configured board cohort, probes and inference horizon to compute the memory curve.

## Supporting controls

See [local transition Jacobians](local_jacobian.md). The retained `dynamics.old_h_recovery` modules evaluate reconstruction of old H from later L; their records belong under `exp/memory/records/old_h/`. These analyses share the model framework. Missing records must be generated or supplied explicitly.
Generate and fit a new old-H readout from the default HF checkpoint and data:

```bash
python -m hrm.analyse.dynamics.old_h_recovery.run --prepare-only
python -m hrm.analyse.dynamics.old_h_recovery.run
python -m hrm.analyse.dynamics.old_h_recovery.fit
python -m hrm.analyse.dynamics.old_h_recovery.audit
```

The default stage directories are `features/`, `decoders/`, and `audit/` under `exp/memory/records/old_h/`. Use separate output directories for additional conditions.
