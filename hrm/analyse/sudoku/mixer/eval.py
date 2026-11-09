import argparse
import json
from pathlib import Path
import re
import yaml

import torch
from torch import nn
import tqdm
import numpy as np

from hrm.analyse.sudoku.mixer.inference import iter_inference
from hrm.training.trainer import load_module
from hrm.evaluation.config import load_config


def select_checkpoint(path: Path, pick: str = "latest") -> Path:
    """Select by epoch or saved evaluation accuracy; accuracy ties favor later epochs."""
    if path.is_file():
        return path
    candidates = []
    for checkpoint in path.glob("epoch_*_acc_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)_acc_(\d+\.\d+)\.pt", checkpoint.name)
        if match and checkpoint.is_file():
            epoch, accuracy = int(match[1]), float(match[2])
            key = (accuracy, epoch) if pick == "best" else (epoch, accuracy)
            candidates.append((key, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No scored checkpoints found at {path}")
    return max(candidates, key=lambda item: item[0])[1]


# --- Main Eval Logic ---
def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint file or run directory")
    parser.add_argument("--pick", choices=("latest", "best"), default="latest",
                        help="For a directory: highest epoch or highest saved eval accuracy (default: latest)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (defaults to the saved data config)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Override the saved inference cycle limit")
    parser.add_argument("--stable-cycles", type=int, default=None, help="Unchanged-output cycles before stopping; 0 disables")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = select_checkpoint(args.ckpt, args.pick)
    print(f"Checkpoint: {checkpoint}")

    # Load config
    config = load_config(checkpoint)
    split = args.split or config.data.eval_split
    max_cycles = args.max_cycles if args.max_cycles is not None else (config.eval_cycles or config.cycles_per_data)
    stable_cycles = args.stable_cycles if args.stable_cycles is not None else config.eval_stable_cycles

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

    # Evaluation Loop
    total_correct = 0
    total_samples = 0

    correctness = []
    samples = []
    indices = []
    cycles = []

    print(f"Starting evaluation on '{split}' split...")
    with tqdm.tqdm(total=len(eval_loader.dataset)) as progress:
        for result in iter_inference(model, eval_loader, config.local_batch_size, max_cycles, stable_cycles):
            correct = torch.all(result.predictions == result.targets, dim=-1).cpu().numpy()
            correctness.append(correct)
            samples.append(result.inputs.cpu().numpy())
            indices.append(result.indices.cpu().numpy())
            cycles.append(result.cycles.cpu().numpy())
            total_correct += int(correct.sum())
            total_samples += len(correct)
            progress.update(len(correct))

    # Continuous batching completes out of order; retain the original dataset order.
    order = np.argsort(np.concat(indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             correctness=np.concat(correctness, axis=0)[order],
             samples=np.concat(samples, axis=0)[order],
             cycles=np.concat(cycles, axis=0)[order])

    args.output.with_suffix('.json').write_text(json.dumps(dict(total=total_samples,correct=total_correct,accuracy=total_correct/total_samples,max_cycles=max_cycles,stable_cycles=stable_cycles,checkpoint=str(checkpoint)),indent=2)+'\n')
    print(f"\n--- Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Match Accuracy: {total_correct / total_samples:.4f}")
    print(f"Mean Cycles: {np.concat(cycles).mean():.2f}")

if __name__ == "__main__":
    evaluate()
