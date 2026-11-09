"""HRM with a small solution-directed interpolation in input embedding space."""

from typing import Any

import torch
from torch import Tensor, nn

from hrm_cremp.arch.layers import Carry, CastedLinear, CastedScaledEmbedding, Transformer, TransformerConfig, trunc_normal_init_


class HRMFlowBiasConfig(TransformerConfig):
    vocab_size: int
    H_cycles: int
    L_cycles: int
    bptt: bool
    forward_dtype: str
    flow_lambda: float = 0.05
    mixture_logit_std: float = 1.0
    preserve_first_zero: bool = True
    normalize_mixture_variance: bool = False


class HRMFlowBias(nn.Module):
    """Keep HRM intact and perturb only its embedded input."""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        super().__init__()
        config = HRMFlowBiasConfig(**config_dict)
        if not 0 <= config.flow_lambda <= 1:
            raise ValueError("flow_lambda must be between zero and one")
        if config.mixture_logit_std <= 0:
            raise ValueError("mixture_logit_std must be positive")

        dtype = getattr(torch, config.forward_dtype)
        self.vocab_size = config.vocab_size
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bptt = config.bptt
        # A non-persistent scalar buffer lets curriculum training update lambda
        # without recompiling the model or changing checkpoint schemas.
        self.flow_lambda = nn.Buffer(
            torch.tensor(config.flow_lambda, dtype=torch.float32),
            persistent=False,
        )
        self.mixture_logit_std = config.mixture_logit_std
        self.preserve_first_zero = config.preserve_first_zero
        self.normalize_mixture_variance = config.normalize_mixture_variance

        self.H_level = Transformer(config)
        self.L_level = Transformer(config)
        self.embed = CastedScaledEmbedding(
            config.vocab_size, config.hidden_size, cast_to=dtype
        )
        self.lm_head = CastedLinear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.zH_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)),
            persistent=True,
        )
        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=dtype)),
            persistent=True,
        )

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
            scale = weights.square().sum(dim=-1, keepdim=True).sqrt()
            mixture = mixture / scale
        return mixture

    def build_input_embedding(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None,
        mixture_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        ordinary = self.embed(input_ids)
        digit_table = self.embed.scale * self.embed.weight[1:].to(ordinary.dtype)
        inference_weights = mixture_weights.to(ordinary.dtype)
        random_placeholder = self._embed_mixture(inference_weights, digit_table)
        mask = self.placeholder_mask(input_ids)
        z_x = torch.where(mask.unsqueeze(-1), random_placeholder, ordinary)

        if target_ids is None:
            return z_x, z_x, None
        z_y = self.embed(target_ids)
        flow_lambda = self.flow_lambda.to(z_x.dtype)
        if not self.normalize_mixture_variance:
            interpolated = (1 - flow_lambda) * z_x + flow_lambda * z_y
            return interpolated, z_x, z_y

        target_weights = torch.nn.functional.one_hot(
            target_ids.long().clamp_min(1) - 1,
            num_classes=self.vocab_size - 1,
        ).to(inference_weights.dtype)
        biased_weights = (
            (1 - flow_lambda) * inference_weights
            + flow_lambda * target_weights
        )
        biased_placeholder = self._embed_mixture(biased_weights, digit_table)
        embedded = torch.where(mask.unsqueeze(-1), biased_placeholder, ordinary)
        return embedded, z_x, z_y

    def forward(
        self,
        carry: Carry,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
    ) -> tuple[Carry, Tensor]:
        mixture_weights = carry.get("mixture_weights")
        if mixture_weights is None:
            mixture_weights = self.sample_mixture_weights(input_ids)
        x, _z_x, _z_y = self.build_input_embedding(
            input_ids, target_ids, mixture_weights
        )

        return self.forward_embedded(carry, x, mixture_weights)

    def forward_embedded(
        self,
        carry: Carry,
        embedded_input: Tensor,
        mixture_weights: Tensor,
        detach_carry: bool = True,
    ) -> tuple[Carry, Tensor]:
        """Run one recurrent cycle from an already constructed input embedding."""
        x = embedded_input

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z_h, z_l = carry["z_H"], carry["z_L"]
            for index in range(self.H_cycles * self.L_cycles - 1):
                z_l = self.L_level(z_l + z_h + x)
                if (index + 1) % self.L_cycles == 0:
                    z_h = self.H_level(z_h + z_l)

        z_l = self.L_level(z_l + z_h + x)
        z_h = self.H_level(z_h + z_l)
        next_carry = {
            "z_H": z_h.detach() if detach_carry else z_h,
            "z_L": z_l.detach() if detach_carry else z_l,
            "mixture_weights": mixture_weights.detach(),
        }
        return next_carry, self.lm_head(z_h)

    @property
    def initial_carry(self) -> Carry:
        return {"z_H": self.zH_init, "z_L": self.zL_init}


HRM = HRMFlowBias
