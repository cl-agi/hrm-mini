"""Gaussian H extension using the native HRM carry and auxiliary-loss interface."""
import math
import torch
import torch.distributed as dist
from hrm.models.hrm import HRM
from hrm.models.layers import CastedLinear
from hrm.models.distributed_kl import distributed_gaussian_kl


class StochasticHRM(HRM):
    def __init__(self, config):
        super().__init__(config)
        d = config['hidden_size']
        self.kl_mode = config['kl_mode']
        self.kl_weight = config.get('kl_weight', .01)
        self.mean_head = CastedLinear(d, d, bias=True)
        self.log_std_head = CastedLinear(d, d, bias=True)
        with torch.no_grad():
            self.mean_head.weight.copy_(torch.eye(d, device=self.mean_head.weight.device))
            self.log_std_head.weight.zero_()
            self.log_std_head.bias.fill_(math.log(config.get('initial_std', .1)))

    def forward(self, carry, input_ids):
        x = self.embed(input_ids)
        h, l = carry['z_H'].to(x.dtype), carry['z_L'].to(x.dtype)
        means, log_stds, samples = [], [], []
        for hi in range(self.H_cycles):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (self.bptt or hi == self.H_cycles - 1)):
                for li in range(self.L_cycles):
                    with torch.set_grad_enabled(torch.is_grad_enabled() and (self.bptt or li == self.L_cycles - 1)):
                        l = self.L_level(l + h + x)
                u = self.H_level(h + l)
                mu = self.mean_head(u).float()
                log_std = self.log_std_head(u).float().clamp(-5, 2)
                sample = mu + log_std.exp() * torch.randn_like(mu)
                h = sample.to(x.dtype)
                means.append(mu)
                log_stds.append(log_std)
                samples.append(sample)
        output = dict(logits=self.lm_head(h))
        if torch.is_grad_enabled():
            output.update(means=torch.stack(means), log_stds=torch.stack(log_stds),
                          samples=torch.stack(samples), active=carry['active'])
        return dict(z_H=h.detach(), z_L=l.detach()), output

    def supervised_auxiliary_loss(self, output, y):
        # This native train.py hook runs outside the compiled forward, so the
        # existing fullgraph model compilation and DDP setup remain unchanged.
        active = output['active']
        count = active.sum()
        dist.all_reduce(count)
        kl = torch.stack([
            distributed_gaussian_kl(mu, ls, z, self.kl_mode, active, count)
            for mu, ls, z in zip(output['means'], output['log_stds'], output['samples'])
        ]).mean()
        output['kl'] = kl.detach()
        output['std'] = output['log_stds'].detach().exp().mean()
        return self.kl_weight * kl
