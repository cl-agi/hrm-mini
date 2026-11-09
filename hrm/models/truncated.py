import torch
from torch import Tensor
from hrm.models.hrm import HRM as BaseHRM
from hrm.models.layers import Carry

class HRM(BaseHRM):
    def __init__(self, config):
        super().__init__(dict(config, bptt=True))
        self.bp_steps = config["bp_steps"]

    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)
        z_H, z_L = carry["z_H"], carry["z_L"]

        for _i in range(self.H_cycles * self.L_cycles):
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.H_cycles * self.L_cycles - _i <= self.bp_steps):
                z_L = self.L_level(z_L + z_H + x)
            if (_i + 1) % self.L_cycles == 0:
                with torch.set_grad_enabled(torch.is_grad_enabled() and self.H_cycles - _i // self.L_cycles <= self.bp_steps):
                    z_H = self.H_level(z_H + z_L)

        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.lm_head(z_H)  # Ensure no gradient moves across carry
