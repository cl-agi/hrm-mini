from hrm.training.trainer import _exclude_from_code_artifact
from hrm.training.trainer import ArchConfig, DataConfig, load_module
from hrm.training.steps import train_step, run_inference
from typing import Any, Optional
import os
import importlib
import math
import json
import hashlib
import time
import yaml

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

from hrm.training.optimizer import AdamATan2
from hrm.models.layers import Carry
from hrm.training.checkpointing import capture_rng, restore_rng
from hrm.training.early_stopping import EarlyStopping



class TrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    output_dir: str = "runs"
    compile_model: bool = True
    arch: ArchConfig
    data: DataConfig

    seeds: list[int] = [42]
    eval_dataset_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    cycles_per_data: int
    epochs: int
    eval_interval_steps: int = 4160
    eval_outer_steps: list[int] = [2, 4, 8, 16, 32]
    early_stop_patience_epochs: float = float("inf")
    early_stop_eval_outer: int = 16

    local_batch_size: int

    lr: float
    lr_warmup_steps: int
    lr_min_ratio: float

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None

    log_interval: int = 5

# [Utils]

# [Training and Inference Step]


def update_lr(config: TrainConfig, optim: torch.optim.Optimizer, step: int, total_steps: int) -> float:
    # Linear warmup cosine schedule
    if step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, step / config.lr_warmup_steps)
    else:
        progress = (step - config.lr_warmup_steps) / (total_steps - config.lr_warmup_steps)
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

    resume = None
    rank_resume = None
    if config.resume_from_checkpoint is not None:
        resume = torch.load(config.resume_from_checkpoint, map_location="cpu", weights_only=True)
        assert resume["format_version"] == 2 and resume["world_size"] == WORLD_SIZE and resume["seed"] == seed
        current_config = config.model_dump()
        saved_config = dict(resume["config"])
        # Only the explicitly requested evaluation/termination policy may change.
        for key in ["output_dir", "compile_model", "resume_from_checkpoint", "early_stop_patience_epochs", "early_stop_eval_outer", "eval_interval_steps"]:
            current_config.pop(key, None)
            saved_config.pop(key, None)
        assert current_config == saved_config, "Exact restart requires the same training configuration"
        rank_resume = resume["rank_states"][RANK]
        assert rank_resume["rank"] == RANK

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    data_kwargs = dict(config.data.__pydantic_extra__ or {})
    train_loader, train_metadata = create_dataloader(
        "train", config.local_batch_size, rank=RANK, world_size=WORLD_SIZE,
        seed=seed, resume_worker_state=rank_resume["data_worker_rng"] if rank_resume else None,
        skip_batches=resume["cursor"]["next_batch"] if resume else 0, **data_kwargs
    )  # pyright: ignore[reportCallIssue]
    eval_data_kwargs = dict(data_kwargs)
    if config.eval_dataset_name is not None:
        eval_data_kwargs["dataset_name"] = config.eval_dataset_name
    eval_loaders = {
        split_name: create_dataloader(
            split_name, config.local_batch_size, rank=RANK, world_size=WORLD_SIZE,
            seed=seed, **eval_data_kwargs
        )[0]
        for split_name in ["test_hard"]
    }  # pyright: ignore[reportCallIssue]

    total_steps = int(config.cycles_per_data * len(train_loader) * config.epochs)
    # Dataset size and device count determine the epoch length.

    assert total_steps % config.eval_interval_steps == 0
    assert config.eval_interval_steps % config.cycles_per_data == 0
    assert config.eval_outer_steps == [2, 4, 8, 16, 32]

    # Initialize Model and Optimizer
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model: nn.Module = model_cls(config.arch.__pydantic_extra__ | train_metadata)
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

    step = 0
    start_epoch = 0
    start_batch = 0
    if resume is not None:
        assert resume["total_steps"] == total_steps
        assert train_loader.sampler.state_dict() == rank_resume["sampler"]
        model.module.load_state_dict(resume["model_state_dict"])
        optim.load_state_dict(resume["optimizer_state_dict"])
        for state in optim.state.values():
            state["step"] = state["step"].cpu()
            assert all(k in state for k in ["exp_avg", "exp_avg_sq", "param_ema"])
        step = resume["step"]
        start_epoch = resume["cursor"]["next_epoch"]
        start_batch = resume["cursor"]["next_batch"]
        del resume
        if RANK == 0:
            print(f"EXACT_RESUME step={step} epoch={start_epoch} batch={start_batch} world_size={WORLD_SIZE}", flush=True)

    initial_hash = hashlib.sha256()
    for name, value in model.module.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.cpu().view(torch.uint8).numpy().tobytes())

    # Initialize checkpointing
    run_name = f"{group_name}/seed_{seed}"
    checkpoint_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if RANK == 0:
        with open(os.path.join(checkpoint_dir, "initialization.json"), "w") as f:
            json.dump(dict(sha256=initial_hash.hexdigest()), f)
    assert len(optim.param_groups) == 1
    if RANK == 0:
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)
        with open(os.path.join(checkpoint_dir, "optimizer_groups.json"), "w") as f:
            json.dump(dict(groups=1, peak_lr=config.lr, weight_decay=config.weight_decay,
                           parameters=sum(p.numel() for p in model.parameters()),
                           parameter_names=[n for n,p in model.named_parameters()]), f, indent=2)

    early_stopping = EarlyStopping(config.early_stop_patience_epochs)
    history_file = os.path.join(checkpoint_dir, f"metrics_outer{config.early_stop_eval_outer}.jsonl")
    if os.path.exists(history_file):
        with open(history_file) as f:
            for line in f:
                row = json.loads(line)
                if row["step"] <= step:
                    early_stopping.update(row["correct"], row["epoch_fraction"], row["step"])
    initial_step = step
    started_at = time.time()
    # -----Train & Eval loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=total_steps, initial=step, desc=f"seed={seed}")

        wandb.init(project=config.data.name,
                   name=run_name,
                   group=group_name,
                   config=config.model_dump() | {"seed": seed},
                   settings=wandb.Settings(x_disable_stats=True))
        if wandb.run is not None:
            wandb.run.log_code(exclude_fn=_exclude_from_code_artifact)

    def evaluate_checkpoint(step, epoch, epoch_fraction, next_batch, worker_rng):
        # Eval
        model.eval()
        optim.swap_ema()

        # Save model
        # Clean '_orig_mod.' prefix added by torch.compile for easier downstream loading
        if RANK == 0:
            torch.save({k.replace("_orig_mod.", ""): v for k, v in model.module.state_dict().items()},
                       os.path.join(checkpoint_dir, f"step_{step}.pt"))

        for eval_name, eval_loader in eval_loaders.items():
            eval_depths = config.eval_outer_steps
            counts = {depth: torch.zeros(2, dtype=torch.long, device="cuda") for depth in eval_depths}
            for x, y in eval_loader:
                # Run inference
                carry: Carry = model.module.initial_carry
                y_hat = None
                for depth in range(1, max(eval_depths) + 1):
                    carry, y_hat = run_inference(model, carry, x.cuda())

                    if depth in counts:
                        counts[depth][0] += torch.all(y_hat == y.cuda(), dim=-1).sum()
                        counts[depth][1] += y.shape[0]

                del carry, y_hat

            for eval_depth, num_total_correct in counts.items():
                # Reduce and log
                dist.reduce(num_total_correct, dst=0)
                num_total_correct = num_total_correct.cpu().tolist()
                if RANK == 0:
                    assert num_total_correct[1] == len(eval_loader.dataset), num_total_correct
                    wandb.log({f"eval/{eval_name}_outer{eval_depth}_exact_match": num_total_correct[0] / num_total_correct[1]}, step=step)
                    print(f"EVAL_RESULT run={run_name} epoch={epoch} split={eval_name} outer={eval_depth} correct={num_total_correct[0]} total={num_total_correct[1]} exact_match={num_total_correct[0] / num_total_correct[1]:.10f}", flush=True)
                    from hrm.training.records import record_evaluation
                    record_evaluation(os.path.join(checkpoint_dir,'evaluation.jsonl'),seed=seed,step=step,split=eval_name,correct=num_total_correct[0],total=num_total_correct[1],cycles=eval_depth,weights='EMA' if config.ema is not None else 'raw')
                    with open(os.path.join(checkpoint_dir, f"metrics_outer{eval_depth}.jsonl"), "a") as metrics_file:
                        metrics_file.write(json.dumps(dict(seed=seed, epoch=epoch, epoch_fraction=epoch_fraction, step=step, training_outer=config.cycles_per_data,
                            split=eval_name, evaluation_cycles=eval_depth, correct=num_total_correct[0], total=num_total_correct[1],
                            exact_match=num_total_correct[0] / num_total_correct[1],
                            parameters=sum(p.numel() for p in model.parameters()),
                            world_size=WORLD_SIZE, local_batch_size=config.local_batch_size,
                            elapsed_seconds=time.time()-started_at)) + "\n")
                    if eval_depth == config.early_stop_eval_outer:
                        early_stopping.update(num_total_correct[0], epoch_fraction, step)

        optim.swap_ema()  # Swap EMA back

        model.train()
        cursor = dict(next_epoch=epoch + 1 if next_batch == len(train_loader) else epoch,
                      next_batch=0 if next_batch == len(train_loader) else next_batch)
        local_state = dict(rank=RANK, rng=capture_rng(), data_worker_rng=worker_rng,
                           sampler=train_loader.sampler.state_dict())
        rank_states = [None] * WORLD_SIZE
        if WORLD_SIZE > 1:
            dist.all_gather_object(rank_states, local_state)
        else:
            rank_states[0] = local_state
        # Snapshot the consumed worker batch, not its potentially prefetched future state.
        if RANK == 0:
            training_checkpoint_path = os.path.join(checkpoint_dir, f"train_step_{step}.pt")
            temporary_checkpoint_path = training_checkpoint_path + ".tmp"
            torch.save(
                {
                    "format_version": 2,
                    "resume_supported": True,
                    "config": config.model_dump(),
                    "cursor": cursor,
                    "rank_states": rank_states,
                    "epoch_fraction": epoch_fraction,
                    "epoch": epoch,
                    "step": step,
                    "total_steps": total_steps,
                    "seed": seed,
                    "world_size": WORLD_SIZE,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "early_stopping": early_stopping.state_dict(),
                },
                temporary_checkpoint_path,
            )
            os.replace(temporary_checkpoint_path, training_checkpoint_path)
            latest_link = os.path.join(checkpoint_dir, "latest_train.pt")
            temporary_link = latest_link + ".tmp"
            os.symlink(os.path.basename(training_checkpoint_path), temporary_link)
            os.replace(temporary_link, latest_link)
        if WORLD_SIZE > 1:
            dist.barrier()

        stop = torch.tensor(int(early_stopping.should_stop(epoch_fraction)) if RANK == 0 else 0,
                            dtype=torch.int32, device="cuda")
        if WORLD_SIZE > 1:
            dist.broadcast(stop, src=0)
        model.train()
        return bool(stop.item())

    early_stopped = early_stopping.should_stop(step / (config.cycles_per_data * len(train_loader)))
    for epoch in range(start_epoch, config.epochs):
        if early_stopped:
            break
        model.train()
        train_loader.sampler.skip_batches = start_batch if epoch == start_epoch else 0
        train_iterator = iter(train_loader)
        if rank_resume is not None:
            # Iterator construction consumes a new base seed; undo that extra draw.
            # The worker independently restores the state after the last consumed batch.
            restore_rng(rank_resume["rng"])
            rank_resume = None
        for batch_index, (x, y, worker_rng) in enumerate(train_iterator, start=train_loader.sampler.skip_batches):
            x = x.cuda()
            y = y.cuda()

            metrics = {}
            lr = None
            carry: Carry = model.module.initial_carry
            for _ in range(config.cycles_per_data):
                step += 1
                lr = update_lr(config, optim, step, total_steps)

                carry, metrics = train_step(model, carry, optim, x, y)
                if step == initial_step + 16 and RANK == 0:
                    with open(os.path.join(checkpoint_dir, "first_update.json"), "w") as f:
                        json.dump(dict(step=step, loss=float(metrics["loss"]), peak_memory_allocated=torch.cuda.max_memory_allocated(), peak_memory_reserved=torch.cuda.max_memory_reserved()), f)

            if step % config.log_interval == 0:
                values = torch.stack([v.float() for v in metrics.values()])
                dist.all_reduce(values)
                logged = dict(zip(metrics, (values / WORLD_SIZE).cpu().tolist()))
                if not all(math.isfinite(v) for v in logged.values()):
                    raise FloatingPointError(f"Nonfinite metrics at {step}")
                if RANK == 0:
                    record = dict(epoch=epoch, step=step, seed=seed, lr=lr,
                                  weight_decay=config.weight_decay, **logged)
                    with open(os.path.join(checkpoint_dir, "train_metrics.jsonl"), "a") as f:
                        f.write(json.dumps(record) + "\n")
                    progress_bar.update(step - progress_bar.n)
                    wandb.log({f"train/{k}": v for k,v in logged.items()} | {"train/lr": lr}, step=step)

            del x, y, carry, metrics
            if step % config.eval_interval_steps == 0:
                early_stopped = evaluate_checkpoint(step, epoch, epoch + (batch_index + 1) / len(train_loader), batch_index + 1, worker_rng)
                if early_stopped:
                    break

    assert step == total_steps or early_stopped
    # Close progress bar and wandb run for this seed
    if progress_bar is not None:
        progress_bar.close()
    if RANK == 0:
        wandb.finish()
        with open(os.path.join(checkpoint_dir, "training_complete.json"), "w") as complete_file:
            json.dump(dict(seed=seed, epoch=max(0, math.ceil(step / (config.cycles_per_data * len(train_loader))) - 1), epoch_fraction=step / (config.cycles_per_data * len(train_loader)),
                           step=step, total_steps=total_steps, world_size=WORLD_SIZE,
                           termination_reason="early_stop" if early_stopped else "max_steps",
                           early_stopping=early_stopping.state_dict()), complete_file)


# [Training Loop]


def train(config_dict):
    from hrm.training.runtime import run_training
    run_training(config_dict, TrainConfig, train_single_seed)
