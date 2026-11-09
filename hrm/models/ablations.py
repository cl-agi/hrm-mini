from typing import Any
import torch
from torch import Tensor
from hrm.models.hrm import HRM as BaseHRM, HRMConfig as BaseConfig
from hrm.models.layers import Carry, Transformer, CastedLinear

class AblationConfig(BaseConfig):
    vocab_size: int

    H_cycles: int
    L_cycles: int
    bptt: bool = True

    dual_module: bool = True  # False = H-L share parameters (TRM)
    hh_link: bool = True
    sym_io: bool = False

    forward_dtype: str

class HRM(BaseHRM):
    config_class = AblationConfig

    def make_low_level(self, config):
        return Transformer(config) if config.dual_module else self.H_level

    def make_head(self, config):
        return CastedLinear(config.hidden_size * (2 if config.sym_io else 1), config.vocab_size, bias=False)

    def __init__(self, config_dict):
        super().__init__(config_dict)
        config = self.config_class(**config_dict)
        self.sym_io, self.hh_link = config.sym_io, config.hh_link


    def forward(self, carry: Carry, input_ids: Tensor) -> tuple[Carry, Tensor]:
        x = self.embed(input_ids)

        # Forward iterations
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.bptt):
            z_H, z_L = carry["z_H"], carry["z_L"]
            for _i in range(self.H_cycles * self.L_cycles - 1):
                z_L = self.L_level(z_L + z_H + x)
                if (_i + 1) % self.L_cycles == 0:
                    H_last = z_H if self.hh_link else 0
                    z_H = self.H_level(z_L + H_last + x if self.sym_io else z_L + H_last)

        # 1-step grad
        z_L = self.L_level(z_L + z_H + x)
        H_last = z_H if self.hh_link else 0
        z_H = self.H_level(z_L + H_last + x if self.sym_io else z_L + H_last)
        return dict(z_H=z_H.detach(), z_L=z_L.detach()), self.lm_head(torch.cat((z_L, z_H), dim=-1) if self.sym_io else z_H)
