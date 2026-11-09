# Generalization

[Experiment guide](../../entry.md)

## Single-clue edits

The evaluator joins the public `test_single_clue` split to its original puzzles using `parent_id`, evaluates both members, and writes per-pair predictions and numerical summaries. Results are separated by the parent training/test split.

```bash
python -m hrm.analyse.sudoku.single_clue.evaluate
hrm analyse sudoku.single_clue.prepare_statistics --experiment generalization
```

Use `--checkpoint` and `--config` for another model, or `--dataset-path` for a paired NPZ containing `question_old`, `question_new`, `solution_old` and `solution_new`. `--output` selects the record directory. A small check uses `--limit 2 --cycles 1 --batch-size 1 --device cpu`. The default inference budget is 16 cycles. Statistics can also run directly with `python -m hrm.analyse.sudoku.single_clue.prepare_statistics --records DIR`.

## Easy/Hard transfer

Preparation uses the shared trainer and records a new training seed. The 80,000-puzzle workflow samples the public upstream training population, excludes the published easy-validation puzzles, and records the sampling seed. The 1,000-puzzle workflow uses the released `train_easy` split. Both evaluate the published easy-validation and difficulty groups.

```bash
python -m hrm.analyse.sudoku.difficulty.prepare --model hrm --output exp/generalization/records/easy80k
python -m hrm.analyse.sudoku.difficulty.run_experiment --output exp/generalization/records/easy80k
python -m hrm.analyse.sudoku.difficulty_1k.prepare --model hrm --output exp/generalization/records/easy1k
python -m hrm.analyse.sudoku.difficulty_1k.run_experiment --output exp/generalization/records/easy1k
```

Use `--model rt` for the recurrent Transformer. Preparation accepts `--dataset-path` and `--data-seed`; the distributed launcher uses eight GPUs. Each run saves `resolved_config.json`, checkpoints and evaluation records. Summarize training with `hrm analyse training_summary --experiment generalization --records RUN`.
