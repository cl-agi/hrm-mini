"""Two fixed, frequency-only axial RoPE adaptations. No learned weights change."""

import math

import torch

TRAIN_SPAN = 29
BASE = 10000.0
AXIS_DIM = 32
BETA_FAST = 4.0
BETA_SLOW = 1.0


def axis_scale(size):
    if type(size) is not int or size < 2:
        raise ValueError("Expected an integer grid side >= 2")
    return max(1.0, (size - 1) / TRAIN_SPAN)


def frequency_plan(variant, size, device=None):
    s = axis_scale(size)
    indices = torch.arange(0, AXIS_DIM, 2, dtype=torch.float32, device=device)
    original = 1.0 / (BASE ** (indices / AXIS_DIM))
    if variant not in ("original", "ntk", "bandwise"):
        raise ValueError(f"Unknown RoPE variant: {variant}")
    if variant == "original" or size <= 30:
        return original
    if variant == "ntk":
        base = BASE * s ** (AXIS_DIM / (AXIS_DIM - 2))
        return 1.0 / (base ** (indices / AXIS_DIM))
    if variant == "bandwise":
        pairs = AXIS_DIM // 2
        fast = math.floor(pairs * math.log(TRAIN_SPAN / (2 * math.pi * BETA_FAST)) / math.log(BASE))
        slow = math.ceil(pairs * math.log(TRAIN_SPAN / (2 * math.pi * BETA_SLOW)) / math.log(BASE))
        fast, slow = max(0, fast), min(pairs - 1, slow)
        if not fast < slow:
            raise ValueError("Degenerate bandwise frequency range")
        ramp = ((indices / 2 - fast) / (slow - fast)).clamp(0, 1)
        # Preserve high frequencies, interpolate low frequencies, blend in between.
        return original * ((1 - ramp) + ramp / s)
    raise ValueError(f"Unknown RoPE variant: {variant}")


def apply_scaling(model, variant, sizes):
    """Replace only nonpersistent caches for >30; retain the original module/forward.

    Called before torch.compile, on the same device as the original caches.
    The scalar scale is fixed by the input side for all 16 recurrent iterations.
    """
    if variant not in ("original", "ntk", "bandwise"):
        raise ValueError(f"Unknown RoPE variant: {variant}")
    for size in sizes:
        axis_scale(size)
        if variant == "original" or size <= 30:
            continue
        for level in (model.core,):
            rope = level.rotary_emb
            seq_len = size * size + 1
            old_cos, old_sin = rope(seq_len)
            if old_cos.shape != (seq_len, 64) or old_sin.shape != old_cos.shape:
                raise ValueError("Expected the frozen D axial head_dim=64 caches")
            device = old_cos.device
            inv_freq = frequency_plan(variant, size, device)
            coordinates = torch.arange(size, dtype=torch.float32, device=device)
            zero = torch.zeros(1, dtype=torch.float32, device=device)
            rows = torch.cat((zero, coordinates.repeat_interleave(size)))
            cols = torch.cat((zero, coordinates.repeat(size)))
            freqs = torch.cat((torch.outer(rows, inv_freq), torch.outer(cols, inv_freq)), dim=-1)
            angles = torch.cat((freqs, freqs), dim=-1)
            rope.register_buffer(f"cos_{seq_len}", angles.cos(), persistent=False)
            rope.register_buffer(f"sin_{seq_len}", angles.sin(), persistent=False)


def describe(variant, sizes):
    original = frequency_plan("original", 30).double()
    return {str(size): {"axis_scale": axis_scale(size),
            "base": BASE * axis_scale(size) ** (32 / 30) if variant == "ntk" else BASE,
            "inverse_frequencies": frequency_plan(variant, size).tolist(),
            "frequency_ratios": (frequency_plan(variant, size).double() / original).tolist(),
            "attention_amplitude": 1.0} for size in sizes}
