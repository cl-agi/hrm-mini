# HRM-Mini

Minimalistic implementation of Hierarchical Recurrent Model (HRM) for training 3-SAT problems.

## Install requirements

Ensure Python and PyTorch is installed, and your machine have at least 1 GPU and total 80 GiB VRAM. 

Then install pip dependencies:

```bash
pip install -r requirements.txt
```

## W&B Integration

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Launch main experiment

3-SAT 256 with 1000 examples. Internet connection is required to download dataset.

If your VRAM is not sufficient, you can reduce `local_batch_size` in `config/tuned_hrm_sat.yaml`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name tuned_hrm_sat
```
