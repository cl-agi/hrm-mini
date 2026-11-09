"""Apply the existing Single KL recipe after each recurrent RT core update."""
import math
import torch
from hrm.models.rt import RecurrentTransformer
from hrm.models.layers import CastedLinear
from hrm.models.stochastic_hrm import StochasticHRM


class StochasticRT(RecurrentTransformer):
    # Reuse the tested distributed KL reduction and native auxiliary-loss hook.
    supervised_auxiliary_loss = StochasticHRM.supervised_auxiliary_loss

    def __init__(self, config):
        super().__init__(config)
        self.kl_mode = config['kl_mode']
        assert self.kl_mode == 'single'
        self.kl_weight = config.get('kl_weight', .01)
        d = config['hidden_size']
        self.mean_head = CastedLinear(d, d, bias=True)
        self.log_std_head = CastedLinear(d, d, bias=True)
        with torch.no_grad():
            self.mean_head.weight.copy_(torch.eye(d, device=self.mean_head.weight.device))
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(math.log(config.get('initial_std', .1)))

    def forward(self, carry, input_ids):
        x = self.embed(input_ids)
        z = carry['z'].to(x.dtype)
        means, log_stds, samples = [], [], []
        for i in range(self.cycles):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.bptt or i == self.cycles-1)):
                u = self.core(z + x)
                mu = self.mean_head(u).float()
                log_std = self.log_std_head(u).float().clamp(-5, 2)
                sample = mu + log_std.exp() * torch.randn_like(mu)
                z = sample.to(x.dtype)
                means.append(mu)
                log_stds.append(log_std)
                samples.append(sample)
        output = dict(logits=self.lm_head(z))
        if torch.is_grad_enabled():
            output.update(means=torch.stack(means), log_stds=torch.stack(log_stds),
                          samples=torch.stack(samples), active=carry['active'])
        return dict(z=z.detach()), output
