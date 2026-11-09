# Search

[Back to dynamics](README.md)

## Generate simulated-annealing records

```bash
pip install -e ".[search]"
python -m hrm.analyse.dynamics.search_baselines.run_sa --prepare-only
python -m hrm.analyse.dynamics.search_baselines.run_sa
```

The default uses row 2 of the public Sudoku test split and writes `exp/dynamics/records/search/sa`. `--dataset` accepts local `test_hard.csv` or `test.csv`; missing explicit inputs fail. `--board-id` selects a zero-based row in that split. Use the same puzzle for the DFS/HRM comparison.

Preparation freezes three distinct Unix-second inference seeds, the puzzle, reference answer, dataset hash, algorithm settings and code hashes. `--seeds` accepts three explicit distinct seeds for a matched rerun. Execution reuses the saved plan and refuses to overwrite started output. The reference solution is used for post-hoc reporting. The solver receives the puzzle clues.

The algorithm fills each box with a random permutation of its missing digits,
then swaps only non-clue cells within boxes. Energy counts missing distinct
digits in rows and columns. Its initial temperature is the standard deviation
of energies from 1,000 unconditional calibration swaps, floored at 0.01. It
starts a fresh box-valid state after calibration, cools by 0.99 after each
`number_of_blanks ** 2` proposals, and redraws a box-valid state at the original
temperature after 20 chains without an accepted strictly downhill move.
Calibration is additional to the default 10,000,000-proposal search budget.

Each seed produces an NPZ with saved boards, trace, terminal board and a 99×99 normalized 81-cell Hamming distance matrix. Trace records occur every 100 proposals, at restarts, and at success/end. Matrix frames are nearest recorded states to 99 evenly spaced proposal counts from the last restart (or initial state if no restart) to termination; blanks are distinct values. Short traces may repeat frames, and the unique-frame count is reported. The manifest retains all three outcomes, including failures; budget exhaustion is explicitly labelled as an unsolved window. Independent Sudoku validation checks clues and all 27 units.

For a short CPU execution check, use a new output directory with `--budget 100 --stride 10`. The manifest records the outcome at the requested budget.

## Generate DFS records on the same puzzle

```bash
python -m hrm.analyse.dynamics.search_baselines.run_dfs --prepare-only
python -m hrm.analyse.dynamics.search_baselines.run_dfs
```

The default public test split and row 2 match the SA entry. If you supplied `--dataset` or `--board-id` to SA, supply identical values here. Outputs default to `exp/dynamics/records/search/dfs`, with `--output-dir` available. DFS runs with the main package dependencies.

DFS chooses the blank with fewest legal digits (MRV), scans cells in row-major order to resolve ties, and tries candidate digits in ascending order. It records the initial board, every assignment, every dead-end detection and every undo; dead-end events can therefore repeat the preceding state. Search uses puzzle clues and Sudoku constraints. Terminal validity is checked independently, and matching the dataset answer is reported separately.

`trajectory.npz` contains boards, operation labels, 99 sampled indices and the normalized 81-cell Hamming matrix. Sampling chooses the nearest recorded event to each evenly spaced point from initialization to termination, with the earlier event winning a tie. The default safety cap is 1,000,000 recorded states including initialization; increase `--max-states` if necessary. Run status records solved, unsatisfiable or capped outcomes. The manifest reports whether the window actually ends at a solution. A short trace may yield repeated sampled frames. The plan freezes inputs/settings before execution. Use a fresh output directory for each run. DFS follows a deterministic search order.

## Generate the frozen HRM trajectory

```bash
python -m hrm.analyse.dynamics.search_baselines.run_hrm --prepare-only
python -m hrm.analyse.dynamics.search_baselines.run_hrm
```

Defaults use the public Sudoku HRM and the same public test row 2. Pass identical `--dataset` and `--board-id` overrides across SA, DFS and HRM. `--checkpoint` uses a local checkpoint with adjacent `model_config.json` (`--config` can explicitly override it); missing explicit paths fail. This entry requires the standard full-link dual-module attention architecture with a 2H6L schedule. It runs inference with frozen weights and the configured forward dtype. Outputs default to `exp/dynamics/records/search/hrm`, or `--output-dir`.

`--cycles 16` means 16 full inference cycles: each has two H phases, each of six L updates followed by one H update. The trajectory records initialization and every individual L/H update (225 states at the default horizon). Each full inference cycle is checked against native forward for exact H/L state equality. Predictions are direct readouts from the current H state, excluding BOS and class 0. Readouts use the saved states directly.

The selected window ends at the first prediction matching the reference answer, or at the horizon if no solution is found. The latter is explicitly marked as unsolved. The target selects the reporting window after recurrence has completed. Distances use Euclidean distance across all tokens and coordinates of zH+zL, normalized by the maximum over the entire selected window. Ninety-nine nearest update indices are sampled uniformly within that window. The stopping point depends on the checkpoint and puzzle. Zero-distance trajectories produce a zero matrix and an explicit flag.

The NPZ retains full H/L states, direct predictions, operation labels and sampled distance matrix; these outputs are stored in the selected records directory. The manifest records first success, window length, unique sampled frames, hashes and native parity. A short CPU check can use `--device cpu --cycles 2` with a new output directory. This checks execution on the selected puzzle.

## Audit the same-puzzle comparison

```bash
python -m hrm.analyse.dynamics.search_baselines.extract_source
```

This consumes `exp/dynamics/records/search/{sa,dfs,hrm}` and writes `exp/dynamics/records/search/comparison/comparison.json`. `--records` changes the common root; `--sa-records`, `--dfs-records`, `--hrm-records` and `--output-dir` allow explicit locations. Missing, incomplete or changed inputs fail, as do different puzzle, answer, row or dataset hashes across methods. Supply completed producer manifests and a fresh audit output directory.

The audit retains all three SA outcomes; it checks recorded clue/box invariants,
row/column energy, proposal/window counters, terminal solution status and sampled
Hamming distances. It deterministically replays DFS and checks the complete
board/action sequence, status and sampled distances. For HRM it checks the
2H6L operation sequence, unchanged H during L (and L during H), direct-readout
consistency during L, first-success selection and distance scale/matrix using
direct state differences independent of the producer's Gram formula.

The SA audit covers recorded proposals and states. HRM trajectory generation verifies native-forward parity; the records audit checks the saved parity receipt. Reports retain censored-run status.