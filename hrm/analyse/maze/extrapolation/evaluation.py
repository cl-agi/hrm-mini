"""Legal-shortest-path Maze evaluation; reference exact-match is auxiliary."""

import argparse
import json
import os
import numpy as np
from datetime import datetime, timezone

from hrm.analyse.maze.rt_extrapolation.maze_metrics import METRIC_NAME, METRIC_VERSION, score_predictions
from pathlib import Path

import torch
import torch.distributed as dist

from hrm.models.hrm import HRM
from hrm.analyse.maze.extrapolation.experiment_config import DEFAULT_DATA_ROOT, resolved_config, run_identity, write_json, validate_evaluation_migration
from hrm.analyse.maze.extrapolation.maze_dataset import create_eval_loader


def configure_compiler(runtime):
    # Seven sequence sizes, full/tail batches, and vector/full carries exceed
    # upstream's default limit of eight. Keep the original static full graphs.
    import torch._dynamo.config as dynamo
    dynamo.recompile_limit = runtime["dynamo_recompile_limit"]
    dynamo.accumulated_recompile_limit = runtime["dynamo_accumulated_recompile_limit"]


@torch.inference_mode()
def evaluate(model, names, config, data_root, rank, world_size, prediction_dir=None):
    """Use the compiled module without DDP forward-time buffer collectives.

    Training synchronizes the weights. Independent rank-strided evaluation can
    therefore retain every tail sample without padding or duplicate examples.
    """
    model.eval()
    results = {}
    for name in names:
        loader, metadata = create_eval_loader(
            name, batch_size=config["training"]["local_batch_size"], rank=rank,
            world_size=world_size, data_root=data_root,
            num_workers=config["runtime"]["num_workers"],
        )
        totals = torch.zeros(5, dtype=torch.long, device="cuda")
        saved = {key: [] for key in ("inputs", "targets", "predictions", "shortest_path_correct",
                                     "valid_path_correct", "canonical_correct", "format_correct")}
        for x, y in loader:
            xs, ys = x.numpy(), y.numpy()
            x = x.cuda()
            carry = model.initial_carry
            for _ in range(config["training"]["cycles_per_data"]):
                carry, logits = model(carry, x)
                prediction = logits.argmax(dim=-1)
            predictions = prediction.cpu().numpy()
            scores = score_predictions(xs, predictions, targets=ys)
            totals += torch.tensor([int(scores[key].sum()) for key in
                                    ("shortest_path_correct", "valid_path_correct", "canonical_correct", "format_correct")]
                                   + [len(ys)], dtype=torch.long, device="cuda")
            if prediction_dir is not None:
                for key, value in {"inputs": xs, "targets": ys, "predictions": predictions, **scores}.items():
                    if key in saved:
                        saved[key].append(value)
            del x, y, carry, logits, prediction
        if prediction_dir is not None:
            directory = Path(prediction_dir) / name
            directory.mkdir(parents=True, exist_ok=True)
            # A new attempt name preserves interrupted/resumed evaluation artifacts.
            attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            with (directory / f"rank_{rank}_{attempt}.npz").open("xb") as stream:
                np.savez_compressed(stream, metric_name=METRIC_NAME, metric_version=METRIC_VERSION,
                                    rank=rank, world_size=world_size, indices=np.asarray(list(loader.sampler)),
                                    **{key: np.concatenate(values, axis=0) for key, values in saved.items()})
        if dist.is_initialized():
            dist.all_reduce(totals)
        correct, valid, canonical, format_correct, total = totals.cpu().tolist()
        if total != config["evaluation"]["examples_per_dataset"]:
            raise ValueError(f"{name}: evaluated {total} examples, expected exactly 1000")
        results[name] = {"correct": correct, "total": total, "accuracy": correct / total,
                         "metric_name": METRIC_NAME, "metric_version": METRIC_VERSION,
                         "shortest_path_accuracy": correct / total, "valid_path_accuracy": valid / total,
                         "canonical_correct": canonical, "exact_match": canonical / total,
                         "format_accuracy": format_correct / total, "seq_len": metadata["seq_len"]}
        if rank == 0:
            print(f"EVAL_RESULT experiment={config['experiment']} dataset={name} "
                  f"metric={METRIC_NAME} correct={correct} total={total} "
                  f"shortest_path_accuracy={correct / total:.10f} exact_match={canonical / total:.10f}", flush=True)
        del loader
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epoch", type=int, default=19, help="Zero-based EMA checkpoint epoch")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file")
    parser.add_argument("--allow-evaluation-migration", action="store_true",
                        help="Explicitly re-evaluate archived exact-match runs using the verified metric-only migration")
    args = parser.parse_args()
    if not 0 <= args.epoch < 20:
        parser.error("epoch must be between 0 and 19")
    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())
    config = resolved_config(metadata["experiment"], metadata["seed"])
    identity = run_identity(config["experiment"], args.data_root, seed=config["seed"])
    migration = None
    if metadata["identity"] != identity:
        if not args.allow_evaluation_migration:
            raise ValueError("Run identity changed. Use --allow-evaluation-migration only for a verified metric-only migration")
        migration = validate_evaluation_migration(metadata["identity"], identity)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    checkpoint_path = args.run_dir / "checkpoints" / f"epoch_{args.epoch}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    try:
        configure_compiler(config["runtime"])
        with torch.device("cuda"):
            model = HRM(config["architecture"])
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model = torch.compile(model, dynamic=False, fullgraph=True)
        metrics = evaluate(model, config["evaluation"]["final_epoch"], config,
                           args.data_root, rank, world_size,
                           prediction_dir=args.output.parent / (args.output.stem + "_predictions"))
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, {"experiment": config["experiment"], "seed": config["seed"],
                                    "epoch": args.epoch, "weights": "EMA", "identity": identity,
                                    "checkpoint": str(checkpoint_path.resolve()), "training_identity": metadata["identity"],
                                    "evaluation_migration": migration, "metric_name": METRIC_NAME,
                                    "metric_version": METRIC_VERSION, "metrics": metrics})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
