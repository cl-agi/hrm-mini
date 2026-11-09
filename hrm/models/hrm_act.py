"""Native deterministic HRM with the output metadata required by the ACT loop."""
import torch
from hrm.models.hrm import HRM


class HRMAct(HRM):
    def forward(self, carry, input_ids):
        next_carry, logits = super().forward(carry, input_ids)
        output = dict(logits=logits)
        if torch.is_grad_enabled():
            output['active'] = carry['active']
        return next_carry, output
