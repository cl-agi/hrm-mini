from typing import Any, Optional

import torch
from torch import nn
from torch import Tensor

from hrm.models.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry, trunc_normal_init_

class RecurrentTransformerConfig(TransformerConfig):
    vocab_size: int

    cycles: int
    bptt: bool = True

    forward_dtype: str

class RecurrentTransformer(nn.Module):
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = RecurrentTransformerConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.cycles = config.cycles
        self.bptt = config.bptt
        self.rope_steps = config.rope_steps

        if config.mup_base_width is not None:
            if config.mup_base_width <= 0 or config.head_dim != 64 or config.intermediate_size != 4 * config.hidden_size or config.is_mlp_mixer:
                raise ValueError("muP requires positive base width, head_dim=64, FFN=4h and attention")
        self.mup_width_mult = config.hidden_size / config.mup_base_width if config.mup_base_width else 1.0

        # Backbone Layers
        self.core = Transformer(config)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype, base_width=config.mup_base_width)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False,
            init_std=1.0 / config.mup_base_width ** 0.5 if config.mup_base_width else None,
            output_scale=1.0 / self.mup_width_mult)

        # Initial z
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

    def use_rope(self, step: Optional[int]) -> bool:
        return self.rope_steps is None or step is None or step < self.rope_steps

    def forward(self, carry: Carry, input_ids: Tensor, attn_mask: Tensor | None = None, step: Optional[int] = None) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)
        use_rope = self.use_rope(step)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z = carry["z"]
            for _i in range(self.cycles - 1):
                z = self.core(z + x, attn_mask, use_rope=use_rope)

        # 1-step grad
        z = self.core(z + x, attn_mask, use_rope=use_rope)
        return dict(z=z.detach()), self.lm_head(z)  # Ensure no gradient moves across carry

    @property
    def initial_carry(self) -> Carry:
        return dict(z=self.z_init)
