from hrm.training.trainer import _exclude_from_code_artifact
from hrm.training.trainer import ArchConfig, DataConfig, load_module
from hrm.training.steps import train_step, run_inference
from typing import Any, Optional
import os
import importlib
import json
import math
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import pydantic
import hydra
import tqdm
import wandb
import coolname
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hrm.training.optimizer import AdamATan2
from hrm.models.layers import Carry
from hrm.evaluation.streaming import iter_inference

OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("round_div", lambda x, y: round(x / y), replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)
OmegaConf.register_new_resolver("stp_lr", lambda lr, width, reference, exponent: lr * (width / reference) ** -exponent, replace=True)

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    cycles_per_data: int

class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    eval_split: str

class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    output_dir: str = "runs"
    compile_model: bool = True
    arch: ArchConfig
    data: DataConfig
    transformer: dict[str, Any]
    scaling: dict[str, Any]

    seeds: list[int]

    cycles_per_data: int
    eval_cycles: Optional[int]
    eval_stable_cycles: int = pydantic.Field(ge=0)
    epochs: int

    local_batch_size: int

    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float]

    log_interval: int

    def model_config_dict(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return self.transformer | self.arch.__pydantic_extra__ | metadata

# [Utils]


def write_json(path: str, value: Any):
    with open(path + ".tmp", "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(path + ".tmp", path)

# [Training and Inference Step]

def update_lr(config: TrainConfig, optim: torch.optim.Optimizer, step: int, total_steps: int) -> float:
    # Linear warmup cosine schedule
    if step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, step / config.lr_warmup_steps)
    else:
        progress = (step - config.lr_warmup_steps) / max(1, total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in optim.param_groups:
        param_group["lr"] = tensor_lr

    return lr


def train_single_seed(config: TrainConfig, seed: int, group_name: str, WORLD_SIZE: int, RANK: int):
    """Run a full training run for a single seed."""
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    train_loader, train_metadata = create_dataloader("train", config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]
    if len(train_loader) == 0:
        raise ValueError("Training data must fill a global batch; increase train_size/repeat or reduce local_batch_size.")
    eval_loader, _ = create_dataloader(config.data.eval_split, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]

    total_steps = int(config.cycles_per_data * len(train_loader) * config.epochs)

    # Initialize Model and Optimizer
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.model_config_dict(train_metadata))
        if config.compile_model:
            model = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

        # DDP Wrap
        model = DDP(model, static_graph=True)

    optim = AdamATan2(
        model.parameters(),
        lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        ema=config.ema
    )

    # Initialize checkpointing
    run_name = f"{group_name}/seed_{seed}"
    checkpoint_dir = os.path.join(config.output_dir, run_name)
    if RANK == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        write_json(os.path.join(checkpoint_dir, "model_config.json"), config.model_dump())

    # -----Train & Eval loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=total_steps, desc=f"seed={seed}")

        wandb.init(project=config.data.name,
                   name=run_name,
                   group=group_name,
                   config=config.model_dump() | {"seed": seed, "world_size": WORLD_SIZE},
                   settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None:
            wandb.run.log_code(exclude_fn=_exclude_from_code_artifact)

    step = 0
    trained_examples = 0
    training_started = time.monotonic()
    epoch_record = best_record = None
    for epoch in range(config.epochs):
        epoch_started = time.monotonic()
        local_epoch_examples = 0
        model.train()
        for x, y in train_loader:
            local_epoch_examples += x.shape[0]
            x = x.cuda()
            y = y.cuda()

            metrics = {}
            lr = None
            carry: Carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                step += 1
                lr = update_lr(config, optim, step, total_steps)

                carry, metrics = train_step(model, carry, optim, x, y)

            if RANK == 0 and progress_bar is not None and step - progress_bar.n >= config.log_interval:
                progress_bar.update(step - progress_bar.n)
                wandb.log({f"train/{k}": v.item() for k, v in metrics.items()} | {"train/lr": lr}, step=step)

            del x, y, carry, metrics

        # Eval
        model.eval()
        optim.swap_ema()

        counts = torch.zeros(4, dtype=torch.long, device="cuda")
        counts[3] = local_epoch_examples
        # Completion times differ across ranks, so evaluate without DDP collectives.
        for result in iter_inference(
            model.module, eval_loader, config.local_batch_size,
            config.eval_cycles or config.cycles_per_data, config.eval_stable_cycles,
        ):
            counts[0] += torch.all(result.predictions == result.targets, dim=-1).sum()
            counts[1] += result.targets.shape[0]
            counts[2] += result.cycles.sum()

        dist.reduce(counts, dst=0)
        if RANK == 0:
            correct, total, cycles, epoch_examples = counts.tolist()
            trained_examples += epoch_examples
            from hrm.training.records import record_evaluation
            record_evaluation(os.path.join(checkpoint_dir,'evaluation.jsonl'),seed=seed,step=step,split=config.data.eval_split,correct=correct,total=total,cycles=config.eval_cycles or config.cycles_per_data,weights='EMA' if config.ema is not None else 'raw',stable_cycles=config.eval_stable_cycles,mean_cycles=cycles/total)
            accuracy = correct / total
            # Save the evaluated EMA weights without torch.compile's prefix.
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_acc_{accuracy:.6f}.pt")
            torch.save({k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()},
                       checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
            epoch_record = {
                "epoch": epoch, "seed": seed, "world_size": WORLD_SIZE,
                "global_batch_size": config.local_batch_size * WORLD_SIZE,
                "epoch_trained_examples": epoch_examples,
                "trained_examples": trained_examples, "optimizer_steps": step,
                "optimizer_example_exposures": trained_examples * config.cycles_per_data,
                "cycles_per_data": config.cycles_per_data, "checkpoint": checkpoint_path,
                "eval": {config.data.eval_split: {
                    "correct": correct, "total": total, "exact_match": accuracy,
                    "mean_cycles": cycles / total,
                    "max_cycles": config.eval_cycles or config.cycles_per_data,
                    "stable_cycles": config.eval_stable_cycles,
                }},
            }
            wandb.log({
                "epoch": epoch,
                f"eval/{config.data.eval_split}_exact_match": accuracy,
                f"eval/{config.data.eval_split}_mean_cycles": cycles / total,
            }, step=step)

        optim.swap_ema()  # Swap EMA back
        if RANK == 0:
            epoch_record["elapsed_seconds"] = time.monotonic() - epoch_started
            with open(os.path.join(checkpoint_dir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(epoch_record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if best_record is None or accuracy >= best_record["eval"][config.data.eval_split]["exact_match"]:
                best_record = epoch_record

    if RANK == 0:
        write_json(os.path.join(checkpoint_dir, "training_summary.json"), {
            "status": "complete", "seed": seed, "epochs": config.epochs,
            "trained_examples": trained_examples, "optimizer_steps": step,
            "optimizer_example_exposures": trained_examples * config.cycles_per_data,
            "elapsed_seconds": time.monotonic() - training_started,
            "final_epoch": epoch_record, "best_epoch": best_record,
        })

    # Close progress bar and wandb run for this seed
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()


# [Training Loop]


def train(config_dict):
    from hrm.training.runtime import run_training
    run_training(config_dict, TrainConfig, train_single_seed)
