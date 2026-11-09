"""Shared launcher for single-device and torchrun training policies."""
import os
import tempfile
from pathlib import Path
import torch
import torch.distributed as dist

def run_training(config_dict, config_class, train_single_seed):
    if "seeds" not in config_dict:
        raise ValueError("Save explicit seeds before launching training")
    config = config_class(**config_dict)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    owned = not dist.is_initialized()
    rendezvous = None
    if owned:
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
        else:
            fd, rendezvous = tempfile.mkstemp(prefix="ddp_", dir=output)
            os.close(fd)
            dist.init_process_group("nccl", init_method="file://" + rendezvous, rank=0, world_size=1)
    try:
        group = getattr(config, "run_name", None) or "run"
        for seed in config.seeds:
            train_single_seed(config, seed, group, dist.get_world_size(), dist.get_rank())
    finally:
        if owned:
            dist.destroy_process_group()
            if rendezvous:
                Path(rendezvous).unlink(missing_ok=True)
