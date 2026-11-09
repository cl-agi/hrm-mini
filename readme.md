# HRM

Train, evaluate and analyse hierarchical reasoning models on Sudoku, SAT, Maze
and molecular conformations. Published models and datasets load from Hugging Face
when their paths are omitted.

## 1. Prepare the environment

Start the recommended Docker image, clone this repository inside the container,
and install it with `python -m pip install -e .`.
Follow the commands in [Environment setup](docs/environment.md).
Run all commands below from the repository root in that environment.

## 2. Choose pretrained weights or train your own

There are two ways to obtain model weights:

- **Use pretrained weights:** proceed directly to step 3. Running the default
  evaluation command automatically downloads and uses the published pretrained
  checkpoint from Hugging Face. Training is optional.
- **Train your own model:** run the training commands below, then supply the
  resulting checkpoint with `--ckpt PATH` in step 3.

For a new Sudoku HRM run on eight GPUs:

```bash
hrm train --experiment sudoku --recipe sudoku-hrm --fresh-seed \
  --output exp/sudoku/records/train --prepare-only
bash scripts/distributed.sh -m hrm train --experiment sudoku \
  --config exp/sudoku/records/train/resolved_config.json \
  --output exp/sudoku/records/train
```

Preparation downloads the data and saves the configuration and seed. Training
uses that saved configuration. Choose another task or model in the
[experiment guide](entry.md); each task documents its GPU requirements and launcher.

## 3. Evaluate and analyse

Run this command to download and evaluate the pretrained Sudoku model:

```bash
hrm evaluate --task sudoku --cycles 16 --output exp/sudoku/records/test.json
```

Supply `--ckpt PATH` to evaluate your trained model and `--dataset-path PATH`
to use local data.

For analyses, follow the record-generation and analysis commands in the
[experiment guide](entry.md). Each workflow saves its inputs under
`exp/<topic>/records/`; `--records PATH` selects another input directory.

[Command options](docs/usage.md) · [Experiment guide](entry.md) · [Apache 2.0 license](LICENSE)
Third-party packages retain their upstream licenses in `third_party/`.
