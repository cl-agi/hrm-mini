"""Adjacent, independently parameterized timescales with full-segment BPTT."""
from typing import Any

import pydantic
import torch
from torch import Tensor, nn

from hrm.models.layers import CastedLinear, CastedScaledEmbedding, Transformer, TransformerConfig, trunc_normal_init_


class MultiscaleConfig(TransformerConfig):
    # No inherited scalar depth is used: each module has its explicit depth.
    num_layers: int = 1
    vocab_size: int
    block_counts: tuple[int, ...]
    update_ratios: tuple[int, ...]
    slow_cycles: int
    forward_dtype: str

    @pydantic.model_validator(mode="after")
    def validate_schedule(self):
        if len(self.block_counts) < 2 or len(self.update_ratios) != len(self.block_counts) - 1:
            raise ValueError("Expected K depths and K-1 adjacent update ratios, K >= 2")
        if min(self.block_counts) < 1 or min(self.update_ratios) < 2 or self.slow_cycles < 1:
            raise ValueError("Depths/cycles must be positive; adjacent timescales must differ")
        return self


def cycle_schedule(ratios: tuple[int, ...]) -> tuple[int, ...]:
    """Return module indices in one complete slowest-layer cycle."""
    def expand(level):
        if level == 0:
            return (0,)
        return expand(level - 1) * ratios[level - 1] + (level,)
    return expand(len(ratios))


class HRMMultiscale(nn.Module):
    def __init__(self, config_dict: dict[str, Any]):
        super().__init__()
        if any(key in config_dict for key in ("H_cycles", "L_cycles", "num_layers", "bptt", "readout")):
            raise ValueError("Use explicit block_counts/update_ratios/slow_cycles; full BPTT and slow readout are fixed")
        config = MultiscaleConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)
        self.slow_cycles = config.slow_cycles
        self.block_counts = config.block_counts
        self.schedule = cycle_schedule(config.update_ratios)
        self.cycle_blocks = sum(self.block_counts[i] for i in self.schedule)
        self.levels = nn.ModuleList([
            Transformer(config.model_copy(update={"num_layers": depth})) for depth in config.block_counts
        ])
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        for i in range(len(self.levels)):
            self.register_buffer(f"z{i + 1}_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)))

    @property
    def initial_carry(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, f"z{i + 1}_init") for i in range(len(self.levels)))

    def advance_cycle(self, carry: tuple[Tensor, ...], x: Tensor) -> tuple[Tensor, ...]:
        """Advance exactly one slow cycle, using an already embedded input."""
        states = list(carry)
        for i in self.schedule:
            if i == 0:
                incoming = states[0] + states[1] + x
            elif i == len(states) - 1:
                incoming = states[i] + states[i - 1]
            else:
                incoming = states[i] + states[i - 1] + states[i + 1]
            states[i] = self.levels[i](incoming)
        return tuple(states)

    def forward_cycle(self, carry: tuple[Tensor, ...], input_ids: Tensor):
        """Read at a real slow-layer endpoint; never used as a training segment."""
        states = self.advance_cycle(carry, self.embed(input_ids))
        return tuple(z.detach() for z in states), self.lm_head(states[-1])

    def forward(self, carry: tuple[Tensor, ...], input_ids: Tensor):
        x = self.embed(input_ids)
        for _ in range(self.slow_cycles):
            carry = self.advance_cycle(carry, x)
        return tuple(z.detach() for z in carry), self.lm_head(carry[-1])
