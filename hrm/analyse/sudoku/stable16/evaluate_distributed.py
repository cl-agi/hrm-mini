#!/usr/bin/env python3
"""Evaluate a Sudoku checkpoint with unpadded shards and independent GPU workers.

Run from the installed checkout, for example:
  torchrun --standalone --nproc_per_node=8 evaluate_distributed.py \
      --ckpt checkpoints/run/seed_42/epoch_0.pt \
      --output-dir results/original --max-cycles 16 --stable-cycles 0

Pass multiple cycle budgets (for example --max-cycles 16 128 2048) to record
all budgets in one continuous pass, under output-dir/cycles_<budget>/.

Use a different output directory for each checkpoint/evaluation mode. Results
are never overwritten. Inference runs without DDP; only final scalar statistics
are reduced, so uneven shards and completion order are safe.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import time

import numpy as np


def restore_global_indices(local_indices, rank, world_size, dataset_size):
    """Validate an entire completed shard and reconstruct dataset positions."""
    if world_size < 1 or not 0 <= rank < world_size or dataset_size < 1:
        raise ValueError("Invalid rank, world size, or dataset size")
    local_indices = np.asarray(local_indices, dtype=np.int64)
    expected_count = len(range(rank, dataset_size, world_size))
    if local_indices.ndim != 1 or not np.array_equal(
        np.sort(local_indices), np.arange(expected_count, dtype=np.int64)
    ):
        raise ValueError(f"Rank {rank} has missing, repeated, or invalid local indices")
    return rank + local_indices * world_size


def validate_complete_results(shards, dataset_size, max_cycles, stable_cycles):
    """Check all persisted shards, including exact coverage, and return metrics."""
    all_indices, all_correctness, all_cycles = [], [], []
    for rank, shard in enumerate(shards):
        indices = np.asarray(shard["indices"])
        correctness = np.asarray(shard["correctness"])
        cycles = np.asarray(shard["cycles"])
        if not np.array_equal(indices, np.arange(rank, dataset_size, len(shards))):
            raise ValueError(f"Rank {rank} does not contain its complete ordered shard")
        if correctness.shape != indices.shape or cycles.shape != indices.shape:
            raise ValueError(f"Rank {rank} has inconsistent result lengths")
        if correctness.dtype != np.bool_:
            raise ValueError("Correctness must contain Boolean values")
        if not np.issubdtype(cycles.dtype, np.integer):
            raise ValueError("Cycle counts must be integers")
        if np.any(cycles < 1) or np.any(cycles > max_cycles):
            raise ValueError("Cycle count outside requested evaluation limits")
        if stable_cycles > 0 and np.any(cycles < min(max_cycles, stable_cycles + 1)):
            raise ValueError("Stopped before enough predictions for the stability rule")
        if stable_cycles == 0 and np.any(cycles != max_cycles):
            raise ValueError("Fixed-cycle evaluation stopped before its cycle limit")
        all_indices.append(indices)
        all_correctness.append(correctness)
        all_cycles.append(cycles)
    indices = np.concatenate(all_indices)
    if not np.array_equal(np.sort(indices), np.arange(dataset_size)):
        raise ValueError("Evaluation does not cover every dataset sample exactly once")
    correctness = np.concatenate(all_correctness)
    cycles = np.concatenate(all_cycles)
    return {
        "samples": int(len(indices)),
        "correct": int(correctness.sum()),
        "exact_match": float(correctness.mean()),
        "total_cycles": int(cycles.sum()),
        "mean_cycles": float(cycles.mean()),
        "min_cycles": int(cycles.min()),
        "max_cycles": int(cycles.max()),
    }


def load_result_shards(output_dir, world_size, timeout_seconds=60):
    """Read atomically published shards, allowing bounded shared-FS visibility lag.

    The caller must first synchronize its GPU collective with the host. File
    visibility retries supplement that rendezvous; they never replace it or
    accept incomplete coverage.
    """
    if world_size < 1 or timeout_seconds < 0:
        raise ValueError("World size must be positive and visibility timeout nonnegative")
    deadline = time.monotonic() + timeout_seconds
    shards = []
    for worker in range(world_size):
        path = output_dir / f"rank_{worker:05d}.npz"
        while True:
            try:
                with np.load(path, allow_pickle=False) as saved:
                    shards.append({key: saved[key] for key in ("indices", "correctness", "cycles")})
                break
            except FileNotFoundError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Evaluation shard {path} is still missing after the worker "
                        f"rendezvous and {timeout_seconds}s visibility timeout"
                    ) from exc
                time.sleep(min(0.1, remaining))
    return shards


def atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def prepare_output_dirs(output_dir, cycle_limits):
    """Keep single-budget paths compatible and reject all existing result artifacts."""
    outputs = {limit: output_dir / f"cycles_{limit}" if len(cycle_limits) > 1 else output_dir
               for limit in cycle_limits}
    if (output_dir / "run_config.json").exists():
        raise FileExistsError(f"A frozen evaluation already exists in {output_dir}; use a fresh directory")
    existing_dirs = [output_dir, *(path for path in output_dir.glob("cycles_*") if path.is_dir())]
    for directory in existing_dirs:
        if any(directory.glob("rank_*.npz")) or (directory / "summary.json").exists():
            raise FileExistsError(
                f"Results already exist in {directory}; choose a fresh output directory"
            )
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return outputs


def parse_args(argv=None):
    from hrm.paths import repository_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", "--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--limit", type=int, help="Explicit smaller test cohort; saved test size otherwise retained")
    parser.add_argument("--data-seed", type=int, help="Override cohort shuffle seed; new limited cohorts default to the evaluation seed")
    parser.add_argument("--seed", type=int, help="Evaluation RNG; defaults to saved seed or a frozen Unix-second seed")
    parser.add_argument("--output-dir", type=Path, default=repository_root()/"exp/sudoku/records/stable_stopping/evaluation")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cycles", type=int, nargs="+", default=None,
                        help="One cycle limit or strictly increasing budgets for one continuous pass")
    parser.add_argument("--stable-cycles", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Inference batch size per GPU; defaults to saved training value")
    parser.add_argument("--expected-samples", type=int, default=None,
                        help="Assert the unchanged evaluation size (also checks saved test_size)")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--shard-visibility-timeout-seconds", type=float, default=60,
                        help="Bounded wait for shared-filesystem shard visibility after GPU synchronization")
    parser.add_argument("--timeout-seconds", type=int, default=86400,
                        help="Collective timeout, allowing slow compilation or long TTS shards")
    return parser.parse_args(argv)


def evaluate(args):
    # Use the installed package for model, dataset and shared-inference semantics.
    import torch
    import torch.distributed as dist
    from hrm.analyse.sudoku.stable16.inference import iter_inference, normalize_cycle_limits
    from hrm.training.trainer import load_module

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    if use_cuda: torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')
    initialized = False
    try:
        if "RANK" in os.environ:
            dist.init_process_group(
                backend="nccl" if use_cuda else "gloo", timeout=timedelta(seconds=args.timeout_seconds),
                **({"device_id":device} if use_cuda else {}),
            )
            initialized = True
            rank, world_size = dist.get_rank(), dist.get_world_size()

        from hrm.evaluation.config import load_config, EvalConfig
        from hrm import hub
        from hrm.training.data import puzzle_directory
        if args.config is not None and args.ckpt is None:
            raise ValueError('--config requires --checkpoint')
        explicit_checkpoint = args.ckpt is not None
        args.ckpt = hub.checkpoint('sudoku', args.ckpt)
        # No saved private dataset path is required for this public evaluation.
        dataset = args.dataset_path
        if dataset is None and explicit_checkpoint:
            from hrm.config import read_config
            from hrm.paths import checkpoint_config
            saved=read_config(args.config or checkpoint_config(args.ckpt))
            source=saved.get('data',{}).get('dataset_name')
            if source in ('sapientinc/sudoku-extreme','sapientinc/sudoku-extreme-1k'):
                dataset=source
        if dataset is None: dataset=puzzle_directory('sudoku')
        if explicit_checkpoint:
            config = load_config(args.ckpt, args.config, streaming=True, dataset=dataset)
        else:
            config = EvalConfig(arch=hub.model_config('sudoku'),
                data=dict(name='hrm.tasks.sudoku_stable16',eval_split='test_hard',dataset_name=str(dataset)),
                cycles_per_data=16,eval_stable_cycles=16,local_batch_size=32)
        if args.limit is not None and args.limit < 1:
            raise ValueError('--limit must be positive')
        split = args.split or config.data.eval_split
        if config.data.name != "hrm.tasks.sudoku_stable16" or split == "train":
            raise ValueError("This evaluator requires a non-training Sudoku split")
        cycle_limits = normalize_cycle_limits(args.max_cycles if args.max_cycles is not None else (
            config.eval_cycles or config.cycles_per_data
        ))
        multiple_budgets = len(cycle_limits) > 1
        stable_cycles = args.stable_cycles if args.stable_cycles is not None else config.eval_stable_cycles
        batch_size = args.batch_size if args.batch_size is not None else config.local_batch_size
        if args.shard_visibility_timeout_seconds < 0:
            raise ValueError("Shard visibility timeout must be nonnegative")
        if stable_cycles < 0 or batch_size < 1:
            raise ValueError("Cycles and batch size must be positive; stable cycles may be zero")

        # Broadcast preflight failure, so all ranks exit rather than waiting for
        # a worker whose output files would collide with another evaluation.
        output_dirs = {limit: args.output_dir / f"cycles_{limit}" if multiple_budgets else args.output_dir
                       for limit in cycle_limits}
        status = [None]
        if rank == 0:
            try:
                prepare_output_dirs(args.output_dir, cycle_limits)
            except Exception as exc:
                status[0] = f"{type(exc).__name__}: {exc}"
        if initialized:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])

        seed_value = [args.seed if args.seed is not None else (config.seeds[0] if config.seeds else int(time.time()))]
        if initialized: dist.broadcast_object_list(seed_value, src=0)
        seed = seed_value[0]
        config.seeds = [seed]
        torch.manual_seed(seed)
        if use_cuda: torch.cuda.manual_seed_all(seed)
        create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
        data_options = dict(config.data.__pydantic_extra__ or {})
        data_options.setdefault("seed", seed)
        data_options.setdefault("num_workers", 1 if use_cuda else 0)
        if args.limit is not None:
            data_options['test_size'] = args.limit
            data_options.setdefault('subsample_seed', seed)
        if args.data_seed is not None:
            data_options['subsample_seed'] = args.data_seed
        loader, metadata = create_dataloader(
            split, batch_size, rank=rank, world_size=world_size, **data_options,
        )
        dataset_size = len(loader.dataset)
        for label, expected in (("--expected-samples", args.expected_samples),
                                ("resolved test_size", data_options.get("test_size"))):
            if expected is not None and dataset_size != expected:
                raise ValueError(f"Evaluation size {dataset_size} differs from {label}={expected}")
        if dataset_size < 1:
            raise ValueError("Evaluation split must not be empty")
        if loader.drop_last or list(loader.sampler) != list(range(rank, dataset_size, world_size)):
            raise ValueError("Evaluation loader must use ordered, unpadded rank-strided shards")

        # Freeze run identity before loading the model or doing any inference.
        # Dataset rows are hashed in the actual selected/shuffled evaluation order.
        provenance = [None]
        status = [None]
        if rank == 0:
            try:
                digest = hashlib.sha256()
                for row in loader.dataset:
                    digest.update(json.dumps([row['question'], row['answer']], separators=(',', ':')).encode())
                    digest.update(b'\n')
                with args.ckpt.open('rb') as stream:
                    checkpoint_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                plan = dict(checkpoint=str(args.ckpt.resolve()), checkpoint_sha256=checkpoint_digest,
                    config=config.model_dump(), resolved_data_options=data_options,
                    dataset_order_sha256=digest.hexdigest(), samples=dataset_size, split=split,
                    evaluation_seed=seed, world_size=world_size, batch_size=batch_size,
                    cycles=list(cycle_limits), stable_cycles=stable_cycles, subset_limit=args.limit,
                    device_type=device.type, compiled=use_cuda and not args.no_compile,
                    public_model_revision=hub.RELEASES['sudoku'][0] if not explicit_checkpoint else None,
                    public_data_revision=({'sapientinc/sudoku-extreme':'58942f96baeb572ca3127e2a9e9c70f330783d6b','sapientinc/sudoku-extreme-1k':'48941acb15ff8cc57f6fb355dded588ef6c3c2b8'}.get(str(dataset),hub.RELEASES['sudoku'][1]) if args.dataset_path is None else None),
                    code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ['evaluate_distributed.py','inference.py']})
                with (args.output_dir/'run_config.json').open('x') as stream:
                    json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
                provenance[0] = plan
            except Exception as exc:
                status[0] = f'{type(exc).__name__}: {exc}'
        if initialized:
            dist.broadcast_object_list(status, src=0)
            dist.broadcast_object_list(provenance, src=0)
        if status[0] is not None: raise RuntimeError(status[0])

        model_cls = load_module(f"arch.{config.arch.name}")
        with torch.device(device):
            model = model_cls(config.model_config_dict(metadata))
            state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict
            if use_cuda and not args.no_compile:
                model = torch.compile(model, dynamic=False, fullgraph=True)
            model.eval()

        keys = ("indices", "correctness", "cycles", "inputs")
        if args.save_predictions:
            keys += ("predictions",)
        chunks_by_limit = {limit: {key: [] for key in keys} for limit in cycle_limits}
        print(f"rank={rank}/{world_size} samples={len(loader.sampler)} "
              f"cycle_limits={cycle_limits} stable_cycles={stable_cycles}", flush=True)
        inference_start = time.monotonic()
        with torch.inference_mode():
            for result in iter_inference(model, loader, batch_size, cycle_limits, stable_cycles):
                cycles = result.cycles.cpu().numpy()
                limit = result.cycle_limit
                if limit not in chunks_by_limit:
                    raise ValueError("Inference result contains an unrequested cycle budget")
                if np.any(cycles < 1) or np.any(cycles > limit):
                    raise ValueError("Cycle count outside requested evaluation limits")
                chunks = chunks_by_limit[limit]
                chunks["indices"].append(result.indices.cpu().numpy())
                chunks["correctness"].append(
                    (result.predictions == result.targets).flatten(1).all(dim=1).cpu().numpy()
                )
                chunks["cycles"].append(cycles)
                chunks["inputs"].append(result.inputs.cpu().numpy())
                if args.save_predictions:
                    chunks["predictions"].append(result.predictions.cpu().numpy())
        if use_cuda: torch.cuda.synchronize(device)
        inference_seconds = time.monotonic() - inference_start

        results = {}
        for max_cycles, chunks in chunks_by_limit.items():
            output_dir = output_dirs[max_cycles]
            arrays = {}
            for key, parts in chunks.items():
                if parts:
                    arrays[key] = np.concatenate(parts, axis=0)
                elif key in ("inputs", "predictions"):
                    arrays[key] = np.empty((0, metadata["seq_len"]), dtype=np.int32)
                else:
                    arrays[key] = np.empty(0, dtype=np.bool_ if key == "correctness" else np.int64)
            arrays["indices"] = restore_global_indices(arrays["indices"], rank, world_size, dataset_size)
            order = np.argsort(arrays["indices"])
            arrays = {key: values[order] for key, values in arrays.items()}
            rank_path = output_dir / f"rank_{rank:05d}.npz"
            temporary = rank_path.with_name(f".{rank_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(rank_path)

            # Persist each shard before reduction; rank 0 can then verify all files.
            cycles = arrays["cycles"]
            counts = torch.tensor([
                int(arrays["correctness"].sum()), len(cycles), int(cycles.sum()),
            ], dtype=torch.int64, device=device)
            minimum = torch.tensor(int(cycles.min()) if len(cycles) else max_cycles + 1,
                                   dtype=torch.int64, device=device)
            maximum = torch.tensor(int(cycles.max()) if len(cycles) else 0,
                                   dtype=torch.int64, device=device)
            durations = torch.tensor([time.monotonic() - started, inference_seconds],
                                     dtype=torch.float64, device=device)
            if initialized:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                dist.all_reduce(durations, op=dist.ReduceOp.MAX)
            if rank == 0:
                # NCCL reductions may return before GPU completion. Synchronize the
                # reduced counts on the host before opening other ranks' files;
                # every rank persisted its shard before entering this reduction.
                reduced = counts.cpu().tolist()
                shards = load_result_shards(
                    output_dir, world_size, args.shard_visibility_timeout_seconds,
                )
                metrics = validate_complete_results(shards, dataset_size, max_cycles, stable_cycles)
                if reduced != [metrics["correct"], metrics["samples"], metrics["total_cycles"]]:
                    raise ValueError("Persisted shard counts differ from distributed counts")
                if [minimum.item(), maximum.item()] != [metrics["min_cycles"], metrics["max_cycles"]]:
                    raise ValueError("Persisted cycle bounds differ from distributed counts")
                wall_seconds, inference_wall_seconds = durations.cpu().tolist()
                summary = {
                    "schema_version": 1,
                    "complete": True,
                    "checkpoint": str(args.ckpt.resolve()),
                    "split": split,
                    "started_at": started_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "world_size": world_size,
                    "device": str(device),
                    "compiled": use_cuda and not args.no_compile,
                    "evaluation_seed": seed,
                    "input_provenance": provenance[0],
                    "subset_limit": args.limit,
                    "resolved_data_options": data_options,
                    "batch_size_per_gpu": batch_size,
                    "requested_max_cycles": max_cycles,
                    "stable_cycles": stable_cycles,
                    "expected_samples": dataset_size,
                    "coverage": "Every evaluation sample exactly once; ordered rank-strided shards",
                    "wall_seconds": wall_seconds,
                    "inference_wall_seconds": inference_wall_seconds,
                    "predictions_saved": args.save_predictions,
                    "config": config.model_dump(),
                    **metrics,
                }
                if multiple_budgets:
                    summary["shared_inference_max_cycles"] = cycle_limits[-1]
                atomic_json(output_dir / "summary.json", summary)
                results[str(max_cycles)] = {"summary": str((output_dir / "summary.json").relative_to(args.output_dir)),
                                           **metrics}
                print(json.dumps({key: value for key, value in summary.items() if key not in ("config", "input_provenance")}), flush=True)
        if rank == 0 and multiple_budgets:
            atomic_json(args.output_dir / "summary.json", {
                "schema_version": 2, "complete": True,
                "checkpoint": str(args.ckpt.resolve()), "split": split,
                "cycle_budgets": list(cycle_limits), "stable_cycles": stable_cycles,
                "shared_inference": True, "results": results,
            })
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    evaluate(parse_args())
