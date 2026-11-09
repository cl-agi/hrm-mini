# Teacher-CoT corpus

Run from the checkout after installing `third_party/qwen[gpu,cot]`. All outputs below belong in `exp/sudoku/records/qwen/teacher/` and are excluded from Git. These commands prepare teacher traces from public inputs and record the resulting yield. Paid execution requires your own provider credentials and budget.

Without `--execute`, every generation/repair command only prepares and freezes
its inputs. To execute a stage, repeat that same command with `--execute`.
Complete a stage before preparing any stage that depends on its output: the
consumer freezes source hashes and rejects subsequent source changes. New seeds
are Unix seconds at preparation; rerunning a stage reuses its saved seed.

## 1. Original puzzles with DeepSeek

Generate traces for the initial 1,000 puzzles through OpenRouter, then retry unsolved puzzles through the native DeepSeek API. Use a separate output for each backend configuration:

```bash
python -m hrm_qwen.cot.generate --backend openrouter --model deepseek/deepseek-v4-pro \
  --out exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl
```

Execute with `OPENROUTER_API_KEY`. Then prepare native retries on the remaining puzzles, execute once, and repeat with `--retry-errors --retry-wrong` for another attempt. Retry the remaining unsolved puzzles:

```bash
python -m hrm_qwen.cot.generate --backend deepseek --model deepseek-v4-pro \
  --only-unsolved-in exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl \
  --out exp/sudoku/records/qwen/teacher/deepseek_retry.jsonl
```

Native execution uses `DEEPSEEK_API_KEY`. API errors are distinct from wrong answers. Record counts include transport retries; cohort size is determined by unique selected puzzles. Both entries default to 131072 completion tokens, 120000 requested reasoning tokens and temperature 1. Record any provider/budget override in a new prepared output.

## 2. Repair wrong original traces

Once both outputs are final, concatenate them into a new, frozen repair input:

```bash
cat exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl \
    exp/sudoku/records/qwen/teacher/deepseek_retry.jsonl \
    > exp/sudoku/records/qwen/teacher/repair_source.jsonl
python -m hrm_qwen.cot.repair --source exp/sudoku/records/qwen/teacher/repair_source.jsonl \
  --out exp/sudoku/records/qwen/teacher/repair.jsonl
```

Repeat the repair command with `--execute`. Keep `repair_source` frozen after preparation. Repair writes its results to a separate output file.

## 3. Second teacher on the unsolved tail

The following command uses `anthropic/claude-opus-5` through OpenRouter and Amazon Bedrock, selects up to 189 unsolved puzzles ordered by ascending rating, and requests up to two continuations. Confirm endpoint availability and pricing with the provider before execution.

```bash
python -m hrm_qwen.cot.generate --backend openrouter --model anthropic/claude-opus-5 \
  --providers amazon-bedrock --visible-cot --continuations 2 --order rating_asc \
  --select-count 189 \
  --only-unsolved-in exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl,exp/sudoku/records/qwen/teacher/deepseek_retry.jsonl,exp/sudoku/records/qwen/teacher/repair.jsonl \
  --out exp/sudoku/records/qwen/teacher/claude_visible.jsonl
```

Ratings default to the pinned public source. `--select-count` selects at most 189 eligible puzzles **once**, after exclusions and ordering. The frozen selected indices remain fixed across resumed attempts. Ties use the saved shuffle seed. To change the teacher, provider or cap, prepare a different output and describe that substitution when reporting results.

## 4. Hinted variants

After the second teacher finishes, prepare only the remaining unsolved source puzzles with at most 40 empty cells. Teacher generation uses the training split:

```bash
python -m hrm_qwen.cot.generate --empties 40 \
  --only-unsolved-in exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl,exp/sudoku/records/qwen/teacher/deepseek_retry.jsonl,exp/sudoku/records/qwen/teacher/repair.jsonl,exp/sudoku/records/qwen/teacher/claude_visible.jsonl \
  --out exp/sudoku/records/qwen/teacher/hinted40.jsonl
```

Complete the full-difficulty stages before generating hinted traces. Training
selects correct full-difficulty traces first, followed by hinted traces for
the remaining source puzzles.

## 5. Train and evaluate

Keep the listed source priority: one correct trace per source puzzle, original before repaired/second-teacher before hinted. The trainer reports the resulting verified and overlength counts.

```bash
python -m hrm_qwen.cot.train \
  --traces exp/sudoku/records/qwen/teacher/deepseek_initial.jsonl \
           exp/sudoku/records/qwen/teacher/deepseek_retry.jsonl \
           exp/sudoku/records/qwen/teacher/repair.jsonl \
           exp/sudoku/records/qwen/teacher/claude_visible.jsonl \
           exp/sudoku/records/qwen/teacher/hinted40.jsonl \
  --output-dir exp/sudoku/records/qwen/student --prepare-only
```

Launch the same arguments under `bash scripts/distributed.sh` (eight ranks), omitting `--prepare-only`.
Then follow the [Qwen evaluation commands](../../../../third_party/qwen/README.md#teacher-cot--instruct-evaluation)
and [paired statistical report](../../../../third_party/qwen/README.md#statistics-from-a-fresh-evaluation).
