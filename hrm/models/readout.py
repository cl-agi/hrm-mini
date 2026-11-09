from collections.abc import Callable, Collection
from typing import Any, Literal
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from hrm.models.hrm import HRM as BaseHRM, HRMConfig as BaseConfig
from hrm.models.layers import Carry, CastedLinear

class HRMConfig(BaseConfig):
    readout: Literal["h", "l", "hl"] = "h"

class HRM(BaseHRM):
    config_class = HRMConfig

    def make_head(self, config):
        return CastedLinear(config.hidden_size * (2 if config.readout == "hl" else 1), config.vocab_size, bias=False)

    def __init__(self, config):
        super().__init__(config)
        self.readout = self.config_class(**config).readout

    def readout_logits(self, z_H: Tensor, z_L: Tensor) -> Tensor:
        """Decode a pair of H/L states without changing the recurrent dynamics."""
        # At the first L update of a rollout z_H can still be the shared
        # [hidden] initial buffer, while z_L is already [batch, seq, hidden].
        # Broadcast it for intermediate trace readouts; ordinary final readouts
        # already have matching batch/sequence dimensions.
        if z_H.ndim < z_L.ndim:
            z_H = z_H.reshape((1,) * (z_L.ndim - z_H.ndim) + z_H.shape).expand_as(z_L)
        if self.readout == "h":
            readout_state = z_H
        elif self.readout == "l":
            readout_state = z_L
        else:  # self.readout == "hl"; validated by HRMConfig.
            readout_state = torch.cat((z_H, z_L), dim=-1)
        return self.lm_head(readout_state)

    def split_hl_readout_logits(self, z_H: Tensor, z_L: Tensor) -> tuple[Tensor, Tensor]:
        """Return additive H and L logit terms for an HL-readout model."""
        if self.readout != "hl":
            raise ValueError("split_hl_readout_logits is only defined for readout='hl'.")
        if z_H.ndim < z_L.ndim:
            z_H = z_H.reshape((1,) * (z_L.ndim - z_H.ndim) + z_H.shape).expand_as(z_L)
        hidden_size = z_H.shape[-1]
        weight = self.lm_head.weight.to(z_H.dtype)
        return F.linear(z_H, weight[:, :hidden_size]), F.linear(z_L, weight[:, hidden_size:])

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z_H, z_L = carry["z_H"], carry["z_L"]
            for _i in range(self.H_cycles * self.L_cycles - 1):
                z_L = self.L_level(z_L + z_H + x)
                if (_i + 1) % self.L_cycles == 0:
                    z_H = self.H_level(z_H + z_L)

        # 1-step grad
        z_L = self.L_level(z_L + z_H + x)
        z_H = self.H_level(z_H + z_L)
        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.readout_logits(z_H, z_L)  # Ensure no gradient moves across carry

    def forward_with_trace(
        self,
        carry: Carry,
        input_ids: Tensor,
        trace_callback: Callable[[Literal["l", "h"], Tensor, Tensor], None],
        events: Collection[Literal["l", "h"]] = ("l", "h"),
    ) -> tuple[Carry, Tensor]:
        """Inference-equivalent forward pass that reports every L and H state update.

        The callback receives ``("l", z_H, z_L)`` after each requested L update
        and ``("h", z_H, z_L)`` after each requested H update.  ``events`` can
        be used to request only H-boundary states during long rollouts, avoiding
        needless callback work. It is intended for no-grad trajectory analysis;
        the ordinary ``forward`` path remains unchanged.
        """
        invalid_events = set(events) - {"l", "h"}
        if invalid_events:
            raise ValueError(f"Unsupported trace events: {sorted(invalid_events)}")
        x = self.embed(input_ids)
        z_H, z_L = carry["z_H"], carry["z_L"]
        for update_index in range(self.H_cycles * self.L_cycles):
            z_L = self.L_level(z_L + z_H + x)
            if "l" in events:
                trace_callback("l", z_H, z_L)
            if (update_index + 1) % self.L_cycles == 0:
                z_H = self.H_level(z_H + z_L)
                if "h" in events:
                    trace_callback("h", z_H, z_L)

        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.readout_logits(z_H, z_L)

