"""Prepare-only delivery: invoke explicitly with torchrun to train one ABCD run.

Core update equations are copied from canonical HRM-mini fa468c. New engineering
covers mixed-size batching, both RoPE variants, full evaluation, and safe resume.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from hrm.training.optimizer import AdamATan2
from hrm.models.hrm import HRM
from hrm.analyse.maze.extrapolation.evaluation import configure_compiler, evaluate
from hrm.analyse.maze.rt_extrapolation.maze_metrics import METRIC_VERSION
from hrm.analyse.maze.extrapolation.experiment_config import DEFAULT_DATA_ROOT, resolved_config, run_identity, write_json
from hrm.analyse.maze.extrapolation.maze_dataset import create_train_loader


def train_step(model, carry, opt, x, y):
    # Preserve upstream: full-token CE, one optimizer update each outer cycle,
    # and detached recurrent carry supplied by HRM.forward (not TBPTT within it).
    carry, y_hat = model(carry, x)
    loss = F.cross_entropy(y_hat.view(-1, y_hat.shape[-1]).to(torch.float32),
                           y.view(-1).long(), reduction="mean")
    loss.backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        preds = torch.argmax(y_hat, dim=-1)
        metrics = {"loss": loss.detach(),
                   "per_position_accuracy": torch.mean(preds == y, dtype=torch.float32),
                   "exact_match": torch.mean(torch.all(preds == y, dim=-1), dtype=torch.float32)}
    return carry, metrics


def update_lr(config, optim, step, total_steps):
    # Literal upstream schedule: step is incremented before this function.
    if step < config["lr_warmup_steps"]:
        lr = config["lr"] * min(1.0, step / config["lr_warmup_steps"])
    else:
        progress = (step - config["lr_warmup_steps"]) / (total_steps - config["lr_warmup_steps"])
        lr = config["lr"] * (config["lr_min_ratio"] + max(
            0.0, (1 - config["lr_min_ratio"]) * .5 * (1.0 + math.cos(math.pi * progress))))
    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in optim.param_groups:
        param_group["lr"] = tensor_lr
    return lr


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_metric(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def restore_metric_log(path, committed_step):
    """Remove only records from an interrupted, uncommitted epoch on resume."""
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Metrics log has corruption before its final interrupted write")
            continue
        if row["step"] <= committed_step:
            kept.append(json.dumps(row, sort_keys=True))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
    temporary.replace(path)


def check_resume(checkpoint, identity, training):
    if checkpoint.get("format_version") != 1 or checkpoint.get("identity") != identity:
        raise ValueError("Checkpoint code/data/protocol identity does not match this run")
    if checkpoint.get("weights") != "non_ema_training":
        raise ValueError("Resume requires non-EMA training weights and optimizer/EMA state")
    if checkpoint.get("world_size") != training["world_size"]:
        raise ValueError("Resume world size must match the original run")
    if checkpoint.get("total_steps") != training["total_optimizer_steps"]:
        raise ValueError("Resume total optimizer steps mismatch")
    epoch = int(checkpoint["epoch"])
    expected_step = (epoch + 1) * training["batches_per_epoch"] * training["cycles_per_data"]
    if not 0 <= epoch < training["epochs"] or checkpoint["step"] != expected_step:
        raise ValueError("Only complete, internally consistent epoch-boundary checkpoints can resume")
    if len(checkpoint["rng_states"]) != training["world_size"]:
        raise ValueError("Missing per-rank RNG states")


def initialize_run_directory(run_dir, resume, rank, identity, config, train_metadata):
    status = [None]
    if rank == 0:
        try:
            if resume:
                saved = json.loads((run_dir / "run_metadata.json").read_text())
                if saved["identity"] != identity:
                    raise ValueError("Existing run was made with different code, data, or parameters")
            else:
                run_dir.mkdir(parents=True, exist_ok=False)
                (run_dir / "checkpoints").mkdir()
                write_json(run_dir / "resolved_config.json", config)
                write_json(run_dir / "run_metadata.json", {
                    "experiment": config["experiment"], "seed": config["seed"],
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "identity": identity, "training_data": train_metadata,
                    "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "optimizer_step_scalar_device": "cpu",
                })
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def run(config, data_root, run_dir, resume, rank, world_size):
    training = config["training"]
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    loader, train_metadata = create_train_loader(
        mode=config["data_mode"], batch_size=training["local_batch_size"], rank=rank,
        world_size=world_size, seed=config["seed"], data_root=data_root,
        repeat=training["repeat"], num_workers=config["runtime"]["num_workers"],
    )
    total_steps = training["cycles_per_data"] * len(loader) * training["epochs"]
    if total_steps != 10240 or train_metadata["training_examples"] != 1000:
        raise ValueError("Training budget or number of unique examples changed")
    identity = run_identity(config["experiment"], data_root, seed=config["seed"])
    initialize_run_directory(run_dir, resume, rank, identity, config, train_metadata)
    configure_compiler(config["runtime"])
    with torch.device("cuda"):
        raw_model = HRM(config["architecture"])
        compiled_model = torch.compile(raw_model, dynamic=False, fullgraph=True)
        model = DDP(compiled_model, static_graph=True)
    # Deliberately outside the CUDA default-device context: upstream's optimizer
    # step scalars and LR are on CPU; moment/EMA arrays follow their parameters.
    optim = AdamATan2(
        model.parameters(), lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
        betas=(training["beta1"], training["beta2"]),
        weight_decay=training["weight_decay"], ema=training["ema"],
    )
    step, start_epoch, resumed_final_metrics = 0, 0, None
    if resume:
        # map_location='cuda' would silently move upstream's CPU optimizer steps.
        checkpoint = torch.load(run_dir / "checkpoints/latest_train.pt", map_location="cpu", weights_only=True)
        check_resume(checkpoint, identity, training)
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optim.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"] = group["lr"].cpu()
        for state in optim.state.values():
            if state["step"].device.type != "cpu":
                raise ValueError("Optimizer scalar step must remain on CPU")
        step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        torch.set_rng_state(checkpoint["rng_states"][rank]["cpu"])
        torch.cuda.set_rng_state(checkpoint["rng_states"][rank]["cuda"])
        resumed_final_metrics = checkpoint.get("final_metrics")
        del checkpoint
        if rank == 0:
            restore_metric_log(run_dir / "metrics.jsonl", step)
            print(f"RESUME_RESULT experiment={config['experiment']} epoch={start_epoch} step={step}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    last_final_metrics = resumed_final_metrics
    start_time = time.monotonic()
    for epoch in range(start_epoch, training["epochs"]):
        model.train()
        # Intentionally no sampler.set_epoch: match the original fixed ordering.
        for batch_index, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            carry = model.module.initial_carry
            for _ in range(training["cycles_per_data"]):
                step += 1
                lr = update_lr(training, optim, step, total_steps)
                carry, metrics = train_step(model, carry, optim, x, y)
            if rank == 0:
                row = {"kind": "train", "experiment": config["experiment"], "seed": config["seed"],
                       "epoch": epoch, "batch": batch_index, "step": step, "seq_len": x.shape[1],
                       "lr": lr, "metric_scope": "rank_0_last_outer_cycle",
                       **{name: value.item() for name, value in metrics.items()}}
                append_metric(metrics_path, row)
                print(f"TRAIN_RESULT experiment={config['experiment']} epoch={epoch} "
                      f"step={step}/{total_steps} seq_len={x.shape[1]} loss={row['loss']:.6f}", flush=True)
            del x, y, carry, metrics

        model.eval()
        optim.swap_ema()
        try:
            if rank == 0:
                # Keys match the original plain EMA state_dict checkpoints.
                atomic_torch_save(raw_model.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch}.pt")
            names = (config["evaluation"]["final_epoch"] if epoch == training["epochs"] - 1
                     else config["evaluation"]["every_epoch"])
            eval_metrics = evaluate(compiled_model, names, config, data_root, rank, world_size,
                                    prediction_dir=run_dir / "evaluations" / METRIC_VERSION / f"epoch_{epoch}")
            if rank == 0:
                for name, values in eval_metrics.items():
                    append_metric(metrics_path, {"kind": "eval", "experiment": config["experiment"],
                                                "seed": config["seed"], "epoch": epoch, "step": step,
                                                "weights": "EMA", "dataset": name, **values})
            if epoch == training["epochs"] - 1:
                last_final_metrics = eval_metrics
        finally:
            optim.swap_ema()

        rng_state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
        rng_states = [None] * world_size
        dist.all_gather_object(rng_states, rng_state)
        if rank == 0:
            atomic_torch_save({
                "format_version": 1, "identity": identity, "weights": "non_ema_training",
                "epoch": epoch, "step": step, "total_steps": total_steps, "world_size": world_size,
                "model_state_dict": raw_model.state_dict(), "optimizer_state_dict": optim.state_dict(),
                "rng_states": rng_states, "final_metrics": last_final_metrics,
            }, run_dir / "checkpoints/latest_train.pt")
        dist.barrier()

    if step != total_steps or last_final_metrics is None:
        raise ValueError("Run ended without the full update budget and final evaluation")
    if rank == 0:
        # Published only after the resumable checkpoint is safely committed.
        write_json(run_dir / "final_metrics.json", {
            "experiment": config["experiment"], "seed": config["seed"], "epoch": 19,
            "step": step, "weights": "EMA", "identity": identity, "metrics": last_final_metrics,
            "elapsed_seconds_this_invocation": time.monotonic() - start_time,
        })
        print(f"COMPLETED experiment={config['experiment']} step={step}", flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=["D"])
    parser.add_argument("--seed", type=int, required=True, choices=[4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resume", action="store_true", help="Resume this run's last complete epoch")
    args = parser.parse_args()
    config = resolved_config(args.experiment, args.seed)
    if "LOCAL_RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "0")) != 8:
        parser.error("Use torchrun --nproc_per_node=8 to retain local batch 96 / global batch 768")
    if args.resume and not (args.run_dir / "checkpoints/latest_train.pt").is_file():
        parser.error("--resume requires run-dir/checkpoints/latest_train.pt")
    if not args.resume and args.run_dir.exists():
        parser.error("run-dir exists; use a new directory or explicitly --resume")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        run(config, args.data_root.resolve(), args.run_dir.resolve(), args.resume,
            dist.get_rank(), dist.get_world_size())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
