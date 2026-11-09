from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from hrm.models.layers import Carry


class InferenceResult(NamedTuple):
    indices: Tensor
    inputs: Tensor
    targets: Tensor
    predictions: Tensor
    cycles: Tensor
    cycle_limit: int


def normalize_cycle_limits(max_cycles: int | Sequence[int]) -> tuple[int, ...]:
    """Validate positive, strictly increasing inference checkpoints."""
    limits = tuple(max_cycles) if isinstance(max_cycles, Sequence) else (max_cycles,)
    if (not limits or any(type(limit) is not int or limit < 1 for limit in limits)
            or any(a >= b for a, b in zip(limits, limits[1:]))):
        raise ValueError("Cycle limits must be positive, strictly increasing integers")
    return limits


@torch.inference_mode()
def run_inference(model: nn.Module, carry: Carry, x: Tensor):
    carry, logits = model(carry, x)
    return carry, logits.argmax(dim=-1)


@torch.inference_mode()
def iter_inference(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    batch_size: int,
    max_cycles: int | Sequence[int],
    stable_cycles: int = 16,
) -> Iterator[InferenceResult]:
    """Yield inference snapshots, refilling slots when their final cycle completes.

    Stability counts consecutive unchanged argmax outputs (X comparisons require
    X+1 predictions). Zero disables early stopping; max_cycles always applies.
    Results arrive in completion order; indices identify the original input order.
    Pass an eval-mode model without a DDP wrapper.

    Multiple increasing limits share one continuous pass. Each example appears
    once per cycle_limit; cycles records the actual work. Early-stopped predictions
    also serve later limits. State resets only when a completed slot is refilled.
    """
    limits = normalize_cycle_limits(max_cycles)
    max_cycles = limits[-1]
    if batch_size < 1 or stable_cycles < 0:
        raise ValueError("batch_size/max_cycles must be positive and stable_cycles nonnegative")

    source = iter(batches)
    for pending_x, pending_y in source:
        if len(pending_x):
            break
    else:
        return

    device = next(model.parameters()).device
    device_limits = torch.tensor(limits, dtype=torch.long, device=device)
    inputs = pending_x.new_zeros((batch_size, *pending_x.shape[1:]), device=device)
    targets = pending_y.new_zeros((batch_size, *pending_y.shape[1:]), device=device)
    initial = model.initial_carry
    carry = {key: value.expand(batch_size, inputs.shape[1], -1).clone() for key, value in initial.items()}
    previous = torch.full(inputs.shape, -1, dtype=torch.long, device=device)
    cycles = torch.zeros(batch_size, dtype=torch.long, device=device)
    stable = torch.zeros_like(cycles)
    indices = torch.empty_like(cycles)
    active = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pending_offset = next_index = 0
    exhausted = False

    def refill(slots: Tensor) -> int:
        nonlocal pending_x, pending_y, pending_offset, next_index, exhausted
        filled = 0
        while filled < len(slots) and not exhausted:
            if pending_offset == len(pending_x):
                try:
                    pending_x, pending_y = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending_offset = 0
                continue
            count = min(len(slots) - filled, len(pending_x) - pending_offset)
            dest = slots[filled:filled + count]
            chunk = slice(pending_offset, pending_offset + count)
            inputs[dest] = pending_x[chunk].to(device, non_blocking=True)
            targets[dest] = pending_y[chunk].to(device, non_blocking=True)
            indices[dest] = torch.arange(next_index, next_index + count, device=device)
            active[dest] = True
            pending_offset += count
            next_index += count
            filled += count
        return filled

    remaining = refill(torch.arange(batch_size, device=device))
    while remaining:
        carry, predictions = run_inference(model, carry, inputs)
        cycles += active
        unchanged = (predictions == previous).flatten(1).all(dim=1)
        stable = torch.where(active & unchanged, stable + 1, 0)
        previous = predictions
        done = active & (cycles >= max_cycles)
        if stable_cycles:
            done |= active & (stable >= stable_cycles)
        events = active & (done | torch.isin(cycles, device_limits))
        if not events.any():
            continue

        for limit in limits:
            report = active & ((cycles == limit) | (done & (cycles < limit)))
            slots = report.nonzero(as_tuple=True)[0]
            if len(slots):
                # Copies preserve every snapshot across subsequent slot refills.
                yield InferenceResult(indices[slots], inputs[slots], targets[slots],
                                      predictions[slots], cycles[slots], limit)

        slots = done.nonzero(as_tuple=True)[0]
        if not len(slots):
            continue
        active[slots] = False
        remaining -= len(slots)
        for key, value in carry.items():
            value[slots] = initial[key]
        previous[slots] = -1
        cycles[slots] = stable[slots] = 0
        remaining += refill(slots)
