from typing import Any

import torch
from torch import nn
from torch import Tensor

from hrm.models.layers import CastedScaledEmbedding, CastedLinear, TransformerConfig, Transformer, Carry

class FeedForwardTransformerConfig(TransformerConfig):
    vocab_size: int
    forward_dtype: str

class FeedForwardTransformer(nn.Module):
    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = FeedForwardTransformerConfig(**config_dict)
        dtype = getattr(torch, config.forward_dtype)

        self.core = Transformer(config)
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        return {}, self.lm_head(self.core(self.embed(input_ids)))

    @property
    def initial_carry(self) -> Carry:
        return {}
