# hrm-qwen

Vendored implementation, independently installable with `pip install -e third_party/qwen` from the HRM repository root.

Base repository: https://github.com/ZW471/hrm-mini.git

Base commit: `1553d84dae8bae452c01b218799f45fafab48d02`. The vendored source is licensed under [Apache-2.0](LICENSE).

Use a separate environment for Qwen/vLLM and install the main checkout there first: `python -m pip install -e .`, then `python -m pip install -e 'third_party/qwen[gpu,cot]'`. This environment needs a Transformers version supporting the selected Qwen model; CREMP uses its own pinned Transformers environment.

See `SOURCE.json` for the supplied code hashes and documented source adaptations.

## Data defaults and protocol scope

Omitting `--data-dir` in the supplied SFT, prompt-evaluation and rollout commands loads the pinned HF Sudoku splits through `hrm.hub`: `train` is the released 1,000-puzzle training set and `test_hard` aliases the 20,000-puzzle `test` split. An explicit directory must contain the requested CSV (`test.csv` and `test_hard.csv` are accepted aliases); the path must exist. CSV strings preserve leading blanks and normalize `.` to `0`. Source rows and augmentation logic are retained; the adapter resolves paths for the current environment.

The scripts implement different protocols. `sft/train.py`/`sft/eval.py` train and evaluate direct digit predictions. `instruct/rollout.py` and `instruct/train.py` use the model's **own** traces with answer-only supervision and budget forcing. The teacher-CoT pipeline below trains on verified teacher reasoning and answers. Its generation, training and evaluation commands use that objective.

Data-interface checks (no model weights, vLLM or GPU required):

```bash
python -m unittest discover -s third_party/qwen/tests -v
```

## Teacher-CoT training

`hrm_qwen.cot.train` preserves the recovered teacher-CoT objective: all completion tokens (reasoning and answer) are supervised; prompt tokens are masked. It keeps the first verified-correct trace for each index, drops wrong and overlength examples, and repeats the retained examples for the configured epochs. The input JSONL must have `index`, `question`, `reasoning`, `content` and `exact`; optional `layout` defaults to `rows`, and `repaired` records repair provenance. `exact` records the result of verifying the teacher’s answer. Pass original-puzzle trace files before hinted files: training keeps one correct trace per source index, preferring the original puzzle when available. The trainer reports the retained count for each generated corpus.

Install with `pip install -e 'third_party/qwen[gpu,cot]'` to include Liger. Check the GPU dependency combination in the target environment before training. Run from the checkout with verified traces:

```bash
python -m hrm_qwen.cot.train --traces exp/sudoku/records/qwen/teacher/train.jsonl \
  --output-dir exp/sudoku/records/qwen/student --prepare-only
# Same arguments, all ranks read the saved seed and trace hashes:
bash scripts/distributed.sh -m hrm_qwen.cot.train \
  --traces exp/sudoku/records/qwen/teacher/train.jsonl --output-dir exp/sudoku/records/qwen/student
```

Defaults: Qwen/Qwen3.5-4B, max sequence length 131072, two epochs, learning rate 1e-5, gradient accumulation 4, cosine schedule, warmup 10, fp32 master weights, bf16 compute, full-shard FSDP and gradient checkpointing. W&B is off unless `--wandb-project` is supplied. Preparation freezes configuration/input/code hashes and the Unix-second seed. Execution loads the model and tokenizes the traces. Tokenization reports the actual retained/dropped counts before training. Changing arguments, trace contents or source requires a fresh output directory. Start training in a fresh directory. For a same-seed rerun, use a new directory with the saved `--seed`. Final weights/tokenizer are under `student/final/`.

## Teacher-CoT / Instruct evaluation

The recovered diagnostic is exposed as `hrm_qwen.cot.evaluate`. It lets the model finish its reasoning naturally. The `instruct.rollout` workflow uses answer forcing. The parser only grades a final answer after `</think>`. `pass@1` averages all sampled outputs; `pass@4` measures puzzles with at least one correct output. Truncated/unparsed outputs remain in both denominators.

Prepare the base-model cohort (CPU; public data, no model download):

```bash
python -m hrm_qwen.cot.evaluate prepare --records exp/sudoku/records/qwen/base-eval \
  --config test_hard:0:16384:256 --n 4 --dp-size 4 --seed 1
```

This uses the historical Qwen/Qwen3.5-4B revision recorded by the experiment,
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. An alternative HF model requires
`--model REPO --revision FULL_COMMIT_SHA`. A local model uses `--model-path DIR`;
missing directories fail, and all checkpoint files are hashed. The SFT condition
uses the newly trained checkpoint with a 65536-token budget and eight shards:

```bash
python -m hrm_qwen.cot.evaluate prepare --records exp/sudoku/records/qwen/sft-eval \
  --model-path exp/sudoku/records/qwen/student/final \
  --config test_hard:0:65536:256 --n 4 --dp-size 8 --seed 1
```

`--seed 1` retains the original evaluation seed; omit it for a new Unix-second
seed frozen during preparation. Each rank uses saved seed + rank, as in the
original diagnostic. Changing dp-size changes the sampling streams and should
be reported. Each condition prepares the first 256 unmodified test puzzles;
`--data-dir` overrides HF with local CSVs. Additional difficulty conditions can
be added during preparation, e.g. `--config train:8:16384:64`; they use indexed
training augmentation and reveal solution cells until at most eight blanks
remain. The full prepared questions, rank assignments, public-data revision,
code hashes and sampling settings are stored in the records directory.

Run each rank once, assigning one GPU per rank (base condition, Bash):

```bash
pids=()
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank python -m hrm_qwen.cot.evaluate run \
    --records exp/sudoku/records/qwen/base-eval --rank "$rank" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
python -m hrm_qwen.cot.evaluate score --records exp/sudoku/records/qwen/base-eval
```

For SFT use `sft-eval` and ranks 0 through 7. Score only after all ranks succeed. Missing/duplicate puzzles, wrong sample counts, changed source/input/checkpoint or altered completed shards fail explicitly. Rerun failed starts in a fresh records directory with the original seed. Evaluation records contain per-puzzle grades and aggregates. They stay in `exp/sudoku/records/qwen/`, outside Git. Default records location for each subcommand is `exp/sudoku/records/qwen/evaluation` when no override is provided.

Validation currently covers public-data preparation and a mocked inference-to- scoring run (including incomplete and duplicate shard rejection). Run GPU inference to check the installed vLLM/model combination and measure accuracy. The generation entry below prepares fresh traces; repair is documented below.

## Preparing and generating teacher traces

`hrm_qwen.cot.generate` defaults to the public 1,000-puzzle training split.
Preparation needs no API key and makes no paid calls. The seed, source revision,
input identity and request settings are frozen beside the output JSONL:

```bash
python -m hrm_qwen.cot.generate --out exp/sudoku/records/qwen/teacher/train.jsonl
# After setting DEEPSEEK_API_KEY in the environment, explicitly execute:
python -m hrm_qwen.cot.generate --out exp/sudoku/records/qwen/teacher/train.jsonl --execute
# One additional attempt at remaining unsolved puzzles:
python -m hrm_qwen.cot.generate --out exp/sudoku/records/qwen/teacher/train.jsonl \
  --execute --retry-errors --retry-wrong
```

Omit `--seed` to freeze a Unix-second seed on first preparation. Repeated commands reuse it. Changed data, settings or code require a new output path; execution requires traces matching the saved plan. A process lock prevents concurrent writers. By default four workers use the historical DeepSeek V4 Pro endpoint with thinking enabled, a 131072 completion cap and temperature 1. Transport failures, including rate limits, stop after four attempts per request. Confirm endpoint availability with the provider before execution. `--backend openrouter --model MODEL_ID` is supported, including the original visible-CoT mode and bounded continuations; provider/model substitutions must be reported.

The historical corpus used up to three original-puzzle attempts, repaired wrong
traces, visible-CoT traces from another teacher, and 40-blank hinted variants.
The retained corpus preferred a full-difficulty correct trace over a hinted one
for each source puzzle. To generate a hinted subset after freezing the original
trace files, use a new output path:

```bash
python -m hrm_qwen.cot.generate --out exp/sudoku/records/qwen/teacher/hinted40.jsonl \
  --empties 40 --only-unsolved-in exp/sudoku/records/qwen/teacher/train.jsonl
# Repeat with --execute to call the API.
```

`--only-unsolved-in` accepts comma-separated completed trace files; their hashes are frozen during preparation. Keep the source files fixed for this stage. Use the same base training split for all sources. Records retain source indices for the original one-trace-per-puzzle training policy and a separate `variant_id` for the actual puzzle. Pass original trace files before hinted files to training. Retain the original source index for each hinted variant. When data paths are omitted, rating-order/filter options recover ratings by matching each selected puzzle and solution against pinned public `sapientinc/sudoku-extreme` training data. Matching preserves selected row order; missing or conflicting ratings fail. The first lookup reads the full upstream CSV in the HF cache, then caches only the selected rating vector. Source revisions, selected-input identity and rating-vector hash are recorded. Ratings are cached separately from the published dataset. An explicit CSV override must supply its own `rating` column.

`--max-cost` is an optional stop-on-reported-cost threshold, including prior records on resume. In-flight requests can exceed this threshold, and cost tracking depends on provider-reported charges. Set provider-side limits to enforce a spending cap. Wrong answers and API failures remain in the trace log; training filters on verified `exact`. Trace bodies stay outside Git under records, and no credentials are saved. The returned provider/model fields are retained when supplied.

## Repairing wrong teacher traces

Finish generation before preparing repair: the source JSONL is frozen by hash. The repair entry preserves the original heuristic: locate a committed wrong `rXcY=D` placement against the reference solution, ignore hypothetical/retracted placements, cut at the start of that line, and ask the teacher to continue the prefix. Verification covers explicit placements and the final answer. Invalid grid coordinates are ignored.

```bash
python -m hrm_qwen.cot.repair --source exp/sudoku/records/qwen/teacher/train.jsonl \
  --out exp/sudoku/records/qwen/teacher/repair.jsonl
# Requires DEEPSEEK_API_KEY; explicit paid execution:
python -m hrm_qwen.cot.repair --source exp/sudoku/records/qwen/teacher/train.jsonl \
  --out exp/sudoku/records/qwen/teacher/repair.jsonl --execute
```

Preparation is offline and freezes source/config/code hashes and a Unix-second shuffle seed. `--seed 1` requests the historical repair ordering. New input or settings require a new output path. Existing results are checked against source puzzle identity. Repeated execution skips already attempted repairs, including failed attempts, as in the original workflow. To repeat failed repairs, use a new output path and retain the prior seed explicitly; solved original traces are always skipped. `--limit N` bounds repair attempts per invocation. Four workers and at most four transport attempts/request are the default. API keys are read from the environment. Configure a provider-side spending cap for repair requests.

The historical continuation endpoint is DeepSeek's beta prefix-completion API. Prefix length uses an estimate of three characters per token. Check the actual token budget and endpoint compatibility with the provider. Repair outputs contain reasoning text and stay outside Git.

Train with original traces first, then repairs, then any additional teacher
traces, and hints last. The first correct trace per source index wins:

```bash
python -m hrm_qwen.cot.train \
  --traces exp/sudoku/records/qwen/teacher/train.jsonl exp/sudoku/records/qwen/teacher/repair.jsonl \
           exp/sudoku/records/qwen/teacher/hinted40.jsonl \
  --output-dir exp/sudoku/records/qwen/student --prepare-only
```

Include the same original and repair files in `--only-unsolved-in` when preparing
hints. To use repair records from another output file, include that file in both
the training trace list and the hint exclusion list.

Fresh paired confidence intervals and model comparisons use the main package
[statistics commands](#statistics-from-a-fresh-evaluation).
It verifies both completed cohorts and leaves their records unchanged.

See the [ordered teacher-stage recipe](../../exp/sudoku/configs/qwen/teacher_workflow.md) for the complete sequence. `--select-count N` freezes at most N eligible source indices after exclusions and ordering; resumed attempts use the same cohort.

## Statistics from a fresh evaluation

After completing both evaluation cohorts, run:

```bash
hrm analyse baselines.qwen.report --experiment sudoku --records exp/sudoku/records/qwen
```

The records directory must contain `base-eval/` and `sft-eval/`, each with its saved plan and completed rank outputs. Use `--base-records` and `--sft-records` to override those directories. Default budgets are 16,384 and 65,536 tokens respectively; `--base-budget` and `--sft-budget` select matching recorded budgets. The report checks paired puzzle identity and full coverage before computing bootstrap intervals. It computes statistics from the saved evaluation records.