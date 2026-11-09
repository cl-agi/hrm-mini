import torch
from hrm.models.rt import RecurrentTransformer


class RTAct(RecurrentTransformer):
    def forward(self, carry, input_ids):
        carry_out, logits = super().forward(carry, input_ids)
        output = dict(logits=logits)
        if torch.is_grad_enabled():
            output['active'] = carry['active']
        return carry_out, output
