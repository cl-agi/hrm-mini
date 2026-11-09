from hrm.training.trainer import _exclude_from_code_artifact
"""Fixed-update Sudoku data-scaling controller used for E10's training runs."""
from typing import Any, Optional
import os
import json
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import pydantic
import tqdm
import wandb
from hrm.training.optimizer import AdamATan2
from hrm.models.layers import Carry
from hrm.evaluation.streaming import iter_inference
from hrm.training.trainer import load_module
from hrm.training.steps import train_step
from hrm.training.mixer import update_lr

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    cycles_per_data: int

class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    eval_split: str

class EvalConfig(pydantic.BaseModel):
    arch: ArchConfig
    data: DataConfig
    transformer: dict[str, Any]
    seeds: list[int] = pydantic.Field(default_factory=lambda: [42])
    cycles_per_data: int = pydantic.Field(gt=0)
    eval_cycles: Optional[int]
    eval_stable_cycles: int = pydantic.Field(ge=0)
    local_batch_size: int = pydantic.Field(gt=0)

    def model_config_dict(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return self.transformer | self.arch.__pydantic_extra__ | metadata

class TrainConfig(EvalConfig):
    model_config = pydantic.ConfigDict(extra='forbid')
    output_dir: str
    compile_model: bool = True
    scaling: dict[str, Any]
    seeds: list[int]
    total_steps: int = pydantic.Field(gt=0)
    num_evals: int = pydantic.Field(gt=0)

    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float]

    log_interval: int

    @pydantic.model_validator(mode="after")
    def validate_eval_schedule(self):
        if self.total_steps % self.num_evals:
            raise ValueError("total_steps must be divisible by num_evals for equal evaluation intervals")
        return self

def write_json(path: str, value: Any):
    with open(path + ".tmp", "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(path + ".tmp", path)

def train_single_seed(config: TrainConfig, seed: int, group_name: str, WORLD_SIZE: int, RANK: int):
    """Run a full training run for a single seed."""
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    train_loader, train_metadata = create_dataloader("train", config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, training=True, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]
    eval_loader, _ = create_dataloader(config.data.eval_split, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE, seed=seed, **config.data.__pydantic_extra__)  # pyright: ignore[reportCallIssue]

    eval_interval = config.total_steps // config.num_evals
    global_batch_size = config.local_batch_size * WORLD_SIZE

    # Initialize Model and Optimizer
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.model_config_dict(train_metadata))
        model = torch.compile(model, dynamic=False, fullgraph=True) if config.compile_model else model  # pyright: ignore[reportAssignmentType]

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
        if os.path.exists(os.path.join(checkpoint_dir, "model_config.json")):
            raise FileExistsError("Existing training run; use a new output directory (resume not implemented)")
        os.makedirs(checkpoint_dir, exist_ok=True)
        write_json(os.path.join(checkpoint_dir, "model_config.json"), config.model_dump())
        from hrm.training.data_scaling_records import ordered_dataset_identity
        write_json(os.path.join(checkpoint_dir, 'input_provenance.json'), dict(
            seed=seed,world_size=WORLD_SIZE,data=config.data.model_dump(),
            train=ordered_dataset_identity(train_loader.dataset),
            test=ordered_dataset_identity(eval_loader.dataset)))

    # -----Train & Eval loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=config.total_steps, desc=f"seed={seed}")

        wandb.init(project=config.data.name,
                   name=run_name,
                   group=group_name,
                   config=config.model_dump() | {"seed": seed, "world_size": WORLD_SIZE},
                   settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None:
            wandb.run.log_code(exclude_fn=_exclude_from_code_artifact)

    trained_examples = 0
    training_started = time.monotonic()
    interval_started = training_started
    local_interval_examples = 0
    eval_record = best_record = None
    batches = iter(train_loader)
    model.train()
    for step in range(1, config.total_steps + 1):
        if (step - 1) % config.cycles_per_data == 0:
            x, y = next(batches)
            local_interval_examples += x.shape[0]
            x = x.cuda()
            y = y.cuda()
            carry: Carry = model.module.initial_carry

        lr = update_lr(config, optim, step, config.total_steps)
        carry, metrics = train_step(model, carry, optim, x, y)

        if RANK == 0 and progress_bar is not None and (
            step - progress_bar.n >= config.log_interval or step == config.total_steps
        ):
            progress_bar.update(step - progress_bar.n)
            wandb.log({f"train/{k}": v.item() for k, v in metrics.items()} | {"train/lr": lr}, step=step)

        if step % eval_interval:
            continue

        # Eval
        model.eval()
        optim.swap_ema()

        counts = torch.zeros(4, dtype=torch.long, device="cuda")
        counts[3] = local_interval_examples
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
            correct, total, cycles, interval_examples = counts.tolist()
            trained_examples += interval_examples
            accuracy = correct / total
            # Save the evaluated EMA weights without torch.compile's prefix.
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}_acc_{accuracy:.6f}.pt")
            torch.save({k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()},
                       checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
            from hrm.training.data_scaling_records import file_hash
            checkpoint_digest=file_hash(checkpoint_path)
            eval_record = {"checkpoint_sha256": checkpoint_digest,
                "evaluation": step // eval_interval, "seed": seed, "world_size": WORLD_SIZE,
                "global_batch_size": global_batch_size,
                "interval_trained_examples": interval_examples,
                "trained_examples": trained_examples, "optimizer_steps": step,
                "optimizer_example_exposures": step * global_batch_size,
                "cycles_per_data": config.cycles_per_data, "checkpoint": checkpoint_path,
                "eval": {config.data.eval_split: {
                    "correct": correct, "total": total, "exact_match": accuracy,
                    "mean_cycles": cycles / total,
                    "max_cycles": config.eval_cycles or config.cycles_per_data,
                    "stable_cycles": config.eval_stable_cycles,
                }},
            }
            wandb.log({
                "evaluation": step // eval_interval,
                f"eval/{config.data.eval_split}_exact_match": accuracy,
                f"eval/{config.data.eval_split}_mean_cycles": cycles / total,
            }, step=step)

        optim.swap_ema()  # Swap EMA back
        model.train()
        if RANK == 0:
            eval_record["elapsed_seconds"] = time.monotonic() - interval_started
            with open(os.path.join(checkpoint_dir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(eval_record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            from hrm.training.records import record_evaluation
            record_evaluation(os.path.join(checkpoint_dir, 'evaluation.jsonl'),seed=seed,step=step,
                split=config.data.eval_split,correct=correct,total=total,
                cycles=config.eval_cycles or config.cycles_per_data,
                weights='EMA' if config.ema is not None else 'raw',
                stable_cycles=config.eval_stable_cycles,mean_cycles=cycles/total)
            if best_record is None or accuracy >= best_record["eval"][config.data.eval_split]["exact_match"]:
                best_record = eval_record
        local_interval_examples = 0
        interval_started = time.monotonic()

    if RANK == 0:
        write_json(os.path.join(checkpoint_dir, "training_summary.json"), {
            "status": "complete", "seed": seed,
            "total_steps": config.total_steps, "num_evals": config.num_evals,
            "trained_examples": trained_examples, "optimizer_steps": step,
            "optimizer_example_exposures": step * global_batch_size,
            "elapsed_seconds": time.monotonic() - training_started,
            "final_eval": eval_record, "best_eval": best_record,
        })

    # Close progress bar and wandb run for this seed
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()

def train(config_dict):
    from hrm.training.runtime import run_training
    run_training(config_dict, TrainConfig, train_single_seed)
