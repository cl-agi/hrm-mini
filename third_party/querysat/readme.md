# QuerySAT

Vendored TensorFlow implementation. `SOURCE.json` records source and adapted file hashes. Install in a **separate Python 3.11 environment**; the main HRM package uses Python 3.12 or newer in its own environment.

From the repository root:

```bash
python3.11 -m venv /path/to/querysat-venv
/path/to/querysat-venv/bin/python -m pip install "./third_party/querysat[evaluation]"
/path/to/querysat-venv/bin/python -m hrm_querysat.main --help
CUDA_VISIBLE_DEVICES=-1 /path/to/querysat-venv/bin/python -m unittest discover \
  -s third_party/querysat/tests -v
```

For GPU execution in the documented PyTorch image, install the TensorFlow CUDA
libraries in this separate environment and expose their library directories:

```bash
source /path/to/querysat-venv/bin/activate
python -m pip install "./third_party/querysat[evaluation,gpu]"
export LD_LIBRARY_PATH="$(find "$VIRTUAL_ENV/lib" -path '*/nvidia/*/lib' -type d | paste -sd:):${LD_LIBRARY_PATH:-}"
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

The first H100 execution compiles TensorFlow kernels from PTX; allow for this
initial compilation before timing model execution.

The independent installation includes the CNFgen, NetworkX and Python-SAT
requirements used by the dataset registry. CNFgen 0.9.5 returns signed DIMACS
integers directly; this adapter preserves their order.

The upstream CLI defaults to `exp/sat/records/inference/querysat` for model runs and a `data` subdirectory inside each run for generated data, relative to the working directory. `--train_dir` and `--data_dir` override these paths. Its synthetic generators create run-local training data. Benchmark evaluation uses the specified benchmark splits. SHA2019 and external-solver comparisons additionally require their external executables.

Prepare a new synthetic run without generating data or starting training:

```bash
/path/to/querysat-venv/bin/python -m hrm_querysat.main --task ksat --prepare-only
```

The printed run directory contains `resolved_config.json`, including the frozen Unix-second seed, generator, data directory, hyperparameters, package versions and source hashes. `--seed` overrides the seed explicitly. A directory collision raises an error. Defaults keep generated data local to the run; an explicit `--data_dir` must already exist and is recorded as reused input.

Start the prepared run with `--restore RUN --train`. Restore reads the saved hyperparameters and seed; conflicting seed/data overrides and data regeneration are rejected. Evaluation with `--restore RUN --evaluate` requires a checkpoint. Restore loads model/optimizer state and initializes the random stream and data iteration for the new process.

An explicit-checkpoint graph evaluation entry is documented in [SAT and graph coloring](../../exp/sat/README.md). Supply a trained QuerySAT checkpoint and the benchmark inputs for evaluation. Installation checks cover imports and interfaces; benchmark runs produce task metrics.