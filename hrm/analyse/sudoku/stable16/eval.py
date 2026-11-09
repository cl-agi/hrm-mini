import argparse
import json
from pathlib import Path
import re
import yaml

import torch
from torch import nn
import tqdm
import numpy as np

from hrm.analyse.sudoku.stable16.inference import iter_inference, normalize_cycle_limits
from hrm.training.trainer import load_module
from hrm.evaluation.config import EvalConfig, load_config as read_evaluation_config


def select_checkpoint(path: Path, pick: str = "latest") -> Path:
    """Select by progress or accuracy, including historical epoch checkpoints."""
    if path.is_file():
        return path
    candidates = []
    for checkpoint in path.glob("*_acc_*.pt"):
        match = re.fullmatch(r"(?:step|epoch)_(\d+)_acc_(\d+\.\d+)\.pt", checkpoint.name)
        if match and checkpoint.is_file():
            progress, accuracy = int(match[1]), float(match[2])
            key = (accuracy, progress) if pick == "best" else (progress, accuracy)
            candidates.append((key, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No scored checkpoints found at {path}")
    return max(candidates, key=lambda item: item[0])[1]


# --- Main Eval Logic ---
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file or run directory")
    parser.add_argument("--pick", choices=("latest", "best"), default="latest",
                        help="For a directory: latest progress or highest saved eval accuracy (default: latest)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (defaults to the saved data config)")
    parser.add_argument("--max-cycles", type=int, nargs="+", default=None,
                        help="One cycle limit or increasing limits evaluated in a single pass")
    parser.add_argument("--stable-cycles", type=int, default=None, help="Unchanged-output cycles before stopping; 0 disables")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = select_checkpoint(args.ckpt, args.pick)
    print(f"Checkpoint: {checkpoint}")

    # Load config
    config = read_evaluation_config(checkpoint, streaming=True)
    split = args.split or config.data.eval_split
    try:
        limits = normalize_cycle_limits(args.max_cycles or config.eval_cycles or config.cycles_per_data)
    except ValueError as error:
        parser.error(str(error))
    stable_cycles = args.stable_cycles if args.stable_cycles is not None else config.eval_stable_cycles
    if stable_cycles < 0:
        parser.error("--stable-cycles must be nonnegative")

    # Initialize Dataloader
    create_dataloader = load_module(f"dataset.{config.data.name}@create_dataloader")
    
    # Load evaluation dataset
    eval_loader, metadata = create_dataloader(
        split, config.local_batch_size, rank=0, world_size=1, **config.data.__pydantic_extra__  # pyright: ignore[reportCallIssue]
    )

    # Initialize Model
    model_cls = load_module(f"arch.{config.arch.name}")
    with torch.device("cuda"):
        model = model_cls(config.model_config_dict(metadata))
        # Load Checkpoint
        state_dict = torch.load(checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(state_dict, assign=True)
        model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)  # pyright: ignore[reportAssignmentType]

        model.eval()

    # Each boundary shares the same trajectory; only final-budget results advance progress.
    results = {limit: {key: [] for key in ("correctness", "samples", "indices", "cycles")}
               for limit in limits}
    print(f"Starting evaluation on '{split}' split at cycles {list(limits)}...")
    with tqdm.tqdm(total=len(eval_loader.dataset)) as progress:
        for result in iter_inference(model, eval_loader, config.local_batch_size, limits, stable_cycles):
            limit = result.cycle_limit
            values = results[limit]
            values["correctness"].append(
                torch.all(result.predictions == result.targets, dim=-1).cpu().numpy()
            )
            values["samples"].append(result.inputs.cpu().numpy())
            values["indices"].append(result.indices.cpu().numpy())
            values["cycles"].append(result.cycles.cpu().numpy())
            if limit == limits[-1]:
                progress.update(len(result.indices))

    summaries = {}
    for limit, chunks in results.items():
        # Continuous batching completes out of order; restore the dataset order.
        values = {key: np.concatenate(parts) for key, parts in chunks.items()}
        order = np.argsort(values.pop("indices"))
        values = {key: value[order] for key, value in values.items()}
        filename = f"eval_result_cycles_{limit}.npz" if len(limits) > 1 else "eval_result.npz"
        np.savez(args.output_dir / filename, **values)
        summary = {"samples": len(order), "exact_match": float(values["correctness"].mean()),
                   "mean_cycles": float(values["cycles"].mean()), "result_file": filename}
        summaries[str(limit)] = summary
        print(f"Cycles {limit}: exact match {summary['exact_match']:.4f}, "
              f"mean cycles {summary['mean_cycles']:.2f}, samples {summary['samples']}")
    with (args.output_dir / "eval_summary.json").open("w") as stream:
        json.dump({"checkpoint": str(checkpoint.resolve()), "split": split,
                   "cycle_limits": limits, "stable_cycles": stable_cycles,
                   "results": summaries}, stream, indent=2)
        stream.write("\n")

if __name__ == "__main__":
    evaluate()
