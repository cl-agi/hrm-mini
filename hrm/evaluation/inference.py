"""Common recurrent inference and puzzle-level accuracy accounting."""
import torch
from hrm.training.steps import run_inference

@torch.inference_mode()
def evaluate_batch(model, x, y, cycles, stable_cycles=None):
    if cycles < 1 or (stable_cycles is not None and stable_cycles < 1):
        raise ValueError("Inference budgets must be positive")
    if getattr(model, "is_autoregressive", False):
        from hrm.training.losses import generate
        pred = generate(model, x)
        correct = int((pred == y).all(-1).sum())
        return dict(predictions=pred, inference_cycles=torch.ones(x.shape[0],dtype=torch.long,device=x.device),current_correct=correct,ever_correct=correct,total=x.shape[0],curve=[])
    carry = model.initial_carry
    previous = None
    stable = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    stopped = torch.zeros_like(stable, dtype=torch.bool)
    ever = torch.zeros_like(stopped)
    frozen = torch.zeros_like(y)
    used = torch.full_like(stable, cycles)
    curve = []
    for cycle in range(1, cycles + 1):
        carry, pred = run_inference(model, carry, x)
        if previous is not None:
            stable = torch.where((pred == previous).all(-1), stable + 1, torch.zeros_like(stable))
        if stable_cycles is not None:
            new = (~stopped) & (stable >= stable_cycles)
            frozen[new] = pred[new]
            used[new] = cycle
            stopped |= new
            pred = torch.where(stopped[:, None], frozen, pred)
        correct = (pred == y).all(-1)
        ever |= correct
        curve.append(dict(inference_cycle=cycle, current_correct=int(correct.sum()), ever_correct=int(ever.sum()), total=x.shape[0]))
        previous = pred
        if stopped.all(): break
    return dict(predictions=pred, inference_cycles=used, current_correct=int((pred == y).all(-1).sum()), ever_correct=int(ever.sum()), total=x.shape[0], curve=curve)
