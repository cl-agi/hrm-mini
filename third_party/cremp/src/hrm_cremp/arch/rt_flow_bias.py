"""Recurrent Transformer with solution-directed flow bias in embedding space."""

from typing import Any

import torch
from torch import Tensor, nn

from hrm_cremp.arch.layers import Carry, CastedLinear, CastedScaledEmbedding, Transformer, TransformerConfig, trunc_normal_init_


class RecurrentTransformerFlowBiasConfig(TransformerConfig):
    vocab_size: int
    cycles: int
    bptt: bool
    forward_dtype: str
    flow_lambda: float = 0.025
    mixture_logit_std: float = 1.0
    preserve_first_zero: bool = True
    normalize_mixture_variance: bool = True


class RecurrentTransformer(nn.Module):
    """RT backbone with the same paired prior/posterior interface as HRMFlowBias."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = RecurrentTransformerFlowBiasConfig(**config_dict)
        if not 0 <= config.flow_lambda <= 1:
            raise ValueError("flow_lambda must be between zero and one")
        if config.mixture_logit_std <= 0:
            raise ValueError("mixture_logit_std must be positive")

        dtype = getattr(torch, config.forward_dtype)
        self.vocab_size = config.vocab_size
        self.cycles = config.cycles
        self.bptt = config.bptt
        self.flow_lambda = nn.Buffer(torch.tensor(config.flow_lambda, dtype=torch.float32), persistent=False)
        self.mixture_logit_std = config.mixture_logit_std
        self.preserve_first_zero = config.preserve_first_zero
        self.normalize_mixture_variance = config.normalize_mixture_variance

        self.core = Transformer(config)
        self.embed = CastedScaledEmbedding(config.vocab_size, config.hidden_size, cast_to=dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.z_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)), persistent=True)

    def placeholder_mask(self, input_ids: Tensor) -> Tensor:
        mask = input_ids.eq(0)
        if self.preserve_first_zero and mask.shape[1] > 0:
            mask = mask.clone()
            mask[:, 0] = False
        return mask

    def sample_mixture_weights(self, input_ids: Tensor) -> Tensor:
        shape = (*input_ids.shape, self.vocab_size - 1)
        logits = torch.randn(shape, device=input_ids.device, dtype=torch.float32)
        return torch.softmax(logits * self.mixture_logit_std, dim=-1)

    def _embed_mixture(self, weights: Tensor, digit_table: Tensor) -> Tensor:
        mixture = torch.matmul(weights, digit_table)
        if self.normalize_mixture_variance:
            mixture = mixture / weights.square().sum(dim=-1, keepdim=True).sqrt()
        return mixture

    def build_input_embedding(self, input_ids: Tensor, target_ids: Tensor | None, mixture_weights: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        ordinary = self.embed(input_ids)
        digit_table = self.embed.scale * self.embed.weight[1:].to(ordinary.dtype)
        prior_weights = mixture_weights.to(ordinary.dtype)
        random_placeholder = self._embed_mixture(prior_weights, digit_table)
        mask = self.placeholder_mask(input_ids)
        z_x = torch.where(mask.unsqueeze(-1), random_placeholder, ordinary)

        if target_ids is None:
            return z_x, z_x, None

        z_y = self.embed(target_ids)
        flow_lambda = self.flow_lambda.to(z_x.dtype)
        if not self.normalize_mixture_variance:
            embedded = (1 - flow_lambda) * z_x + flow_lambda * z_y
            return embedded, z_x, z_y

        target_weights = torch.nn.functional.one_hot(
            target_ids.long().clamp_min(1) - 1, num_classes=self.vocab_size - 1
        ).to(prior_weights.dtype)
        posterior_weights = (1 - flow_lambda) * prior_weights + flow_lambda * target_weights
        biased_placeholder = self._embed_mixture(posterior_weights, digit_table)
        embedded = torch.where(mask.unsqueeze(-1), biased_placeholder, ordinary)
        return embedded, z_x, z_y

    def forward(self, carry: Carry, input_ids: Tensor, target_ids: Tensor | None = None) -> tuple[Carry, Tensor]:
        mixture_weights = carry.get("mixture_weights")
        if mixture_weights is None:
            mixture_weights = self.sample_mixture_weights(input_ids)
        embedded, _z_x, _z_y = self.build_input_embedding(input_ids, target_ids, mixture_weights)
        return self.forward_embedded(carry, embedded, mixture_weights)

    def forward_embedded(self, carry: Carry, embedded_input: Tensor, mixture_weights: Tensor, detach_carry: bool = True) -> tuple[Carry, Tensor]:
        z = carry["z"]
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            for _ in range(self.cycles - 1):
                z = self.core(z + embedded_input)
        z = self.core(z + embedded_input)
        next_carry = {
            "z": z.detach() if detach_carry else z,
            "mixture_weights": mixture_weights.detach(),
        }
        return next_carry, self.lm_head(z)

    @property
    def initial_carry(self) -> Carry:
        return {"z": self.z_init}


RTFlowBias = RecurrentTransformer
