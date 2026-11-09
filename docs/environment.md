# Environment setup

Use a host with Docker, NVIDIA Container Toolkit and a compatible NVIDIA driver.
Check GPU access with `nvidia-smi`. The Sudoku example uses eight GPUs; each
[task guide](../entry.md) specifies its own hardware and launcher.

## Start the container

Set `HRM_WORKDIR` to a persistent directory with space for the checkout, data,
models and outputs. Copy this repository's clone URL into `HRM_REPO_URL`.

```bash
export HRM_WORKDIR=/absolute/path/to/hrm-work
export HRM_REPO_URL='<repository-clone-url>'
export HRM_IMAGE=sapientai/pytorch-docker@sha256:51ce934445c60da5c6436928ad09cda559b38e40cd00666fe74fdc9f8ef8dd73
mkdir -p "$HRM_WORKDIR"
docker pull "$HRM_IMAGE"
docker run --rm -it --gpus all --shm-size=16g --entrypoint /bin/bash \
  -v "$HRM_WORKDIR:/work" -w /work \
  -e HRM_REPO_URL -e HRM_REPOSITORY=/work/hrm -e HRM_WORKDIR=/work \
  -e HF_HOME=/work/cache/hf -e HRM_ASSET_CACHE=/work/cache/assets \
  -e PIP_CACHE_DIR=/work/cache/pip -e TMPDIR=/work/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
  -e TRITON_CACHE_DIR=/work/cache/triton -e WANDB_MODE=disabled \
  "$HRM_IMAGE"
```

## Clone and install inside the container

```bash
mkdir -p "$TMPDIR"
git clone "$HRM_REPO_URL" "$HRM_REPOSITORY"
cd "$HRM_REPOSITORY"
python -m venv --system-site-packages /work/venv
source /work/venv/bin/activate
python -m pip install -e .
hrm list
```

The environment inherits PyTorch and CUDA from the image. Keep the checkout for
its experiment configurations. When returning to the container, activate
`/work/venv/bin/activate` and run commands from `/work/hrm`.

Continue with [training and evaluation](../readme.md#2-choose-pretrained-weights-or-train-your-own).

## Task-specific dependencies

CREMP and optional baselines have installation commands in their task guides.
QuerySAT uses a separate Python 3.11 / TensorFlow 2.15.1 environment. The main HRM
package requires Python 3.12 or newer.

The distributed launcher starts eight PyTorch workers by default and assigns
compiler-cache directories per launch and rank. Keep the configured worker count
and batch size for the selected training recipe. Cache directories from completed
launches can be removed when their processes have exited.
