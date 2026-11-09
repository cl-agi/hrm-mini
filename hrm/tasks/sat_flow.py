from pathlib import Path
import torch,numpy as np
from hrm.models.hrm_flow_bias import HRM
from hrm.models.rt_flow_bias import RecurrentTransformer
from hrm.analyse.sat.rt.encoding import sat_batch,sat_satisfies

class SATTask:
    def __init__(self, config, data_root, device="cuda"):
        self.device = device
        self.config = config
        self.data_root = Path(data_root)
        self.model_class = HRM if config.get('model_family','hrm') == 'hrm' else RecurrentTransformer
    def model(self):

        torch.manual_seed(self.config['seed'])

        with torch.device(self.device):

            m = self.model_class(self.config['arch'])

        return m

    def bind(self, m, mask):

        for mod in m.modules():

            if mod.__class__.__name__ == 'Attention':

                mod.graph_mask = mask

    def arrays(self, split, n):

        z = np.load(self.data_root / 'data' / f'{split}_n{n}.npz')

        return dict(clauses=z['clauses'], answers=z['answers'])

    def batch(self, a, ids, seed, augment=True):

        n = a['answers'].shape[1]

        g = torch.Generator(device=self.device).manual_seed(seed)

        b = sat_batch(torch.as_tensor(a['clauses'][ids], device=self.device).long(), torch.as_tensor(a['answers'][ids], device=self.device).long(), g, augment, self.config['deletion_max_fraction'])

        ng = torch.Generator(device=self.device).manual_seed(seed + 810000000)

        eps = torch.randn((len(ids), n), device=self.device, generator=ng) * self.config['noise_std']

        lg = torch.Generator(device=self.device).manual_seed(seed + 850000000)

        p = torch.rand((len(ids), 1), device=self.device, generator=lg) * self.config['p_max'] if augment else torch.zeros((len(ids), 1), device=self.device)

        u = torch.rand((len(ids), n), device=self.device, generator=lg)

        bias = (p * u).clamp(max=self.config['leak_amplitude_clip']) * (2 * (b['y'][:, 1:2 * n + 1:2] - 1) - 1) if augment else torch.zeros_like(eps)

        prob = (eps + bias).sigmoid()

        w = torch.stack((1 - prob, prob), -1)

        pairs = torch.stack((w, w.flip(-1)), 2).flatten(1, 2)

        mix = torch.zeros((len(ids), b['x'].shape[1], 3), device=self.device)

        mix[:, :, 2] = 1.0

        mix[:, 1:2 * n + 1, :2] = pairs

        mix[:, 1:2 * n + 1, 2] = 0.0

        b['x'][:, 1:2 * n + 1] = 0

        b['mix'] = mix

        b['p'] = p[:, 0]

        return b

    def ce_cases(self, z, b):

        n = b['n']

        loss = torch.nn.functional.cross_entropy(z[:, :2 * n + 1].float().transpose(1, 2), b['y'][:, :2 * n + 1], reduction='none')

        return (loss * b['loss_mask']).sum(1) / b['loss_mask'].sum(1)
