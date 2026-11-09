# Core architecture and training ablations

[Experiment guide](../../entry.md)

These comparisons share the HRM training framework. Select comparisons through the configuration.

| Question | Configuration choices |
|---|---|
| Does hierarchy depth matter? | Two, three or four timescales |
| Does separation of modules matter? | Separate versus shared H/L weights |
| Does update-frequency separation matter? | Standard schedule versus equal-frequency H/L |
| Does the training method matter? | Deep supervision versus final-output-only supervision |

```bash
hrm recipes --experiment ablations
hrm train --experiment ablations --recipe sudoku-hrm-3scale \
  --fresh-seed --output exp/ablations/records/hrm3 --prepare-only
```

Run training with the same recipe and output directory, omitting `--prepare-only` and using the configured distributed launcher. This menu contains 14 core configurations. Additional retained configurations are available through the unfiltered `hrm recipes` catalog; they provide additional comparison settings. New repetitions use frozen timestamp seeds; historical seeds are preserved unless `--fresh-seed` is requested.

Compare matched training data, parameter/compute budgets, inference budgets and checkpoint-selection rules. Use `hrm analyse training_summary --experiment ablations --split test_hard` to summarize generated training records; select `--cycles` when logs contain several inference budgets. Records must exist. The summary reports peak and last evaluations. The completion record gives the completed training budget.
