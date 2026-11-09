from hrm.paths import analysis_directory
from pathlib import Path
import sys, json, torch, numpy as np
P = analysis_directory(__file__)
pass
from hrm.models.hrm_flow_bias import HRM
from hrm.training.optimizer import AdamATan2
from hrm.analyse.sat.rt.encoding import sat_batch, sat_satisfies
CFG = json.loads((P / 'config.json').read_text())

def model():
    torch.manual_seed(CFG['seed'])
    with torch.device('cuda'):
        m = HRM(CFG['arch'])
    return m

def bind(m, mask):
    for mod in m.modules():
        if mod.__class__.__name__ == 'Attention':
            mod.graph_mask = mask

def arrays(split, n):
    z = np.load(P / 'data' / f'{split}_n{n}.npz')
    return dict(clauses=z['clauses'], answers=z['answers'])

def batch(a, ids, seed, augment=True):
    n = a['answers'].shape[1]
    g = torch.Generator(device='cuda').manual_seed(seed)
    b = sat_batch(torch.as_tensor(a['clauses'][ids], device='cuda').long(), torch.as_tensor(a['answers'][ids], device='cuda').long(), g, augment, CFG['deletion_max_fraction'])
    ng = torch.Generator(device='cuda').manual_seed(seed + 810000000)
    eps = torch.randn((len(ids), n), device='cuda', generator=ng) * CFG['noise_std']
    lg = torch.Generator(device='cuda').manual_seed(seed + 850000000)
    p = torch.rand((len(ids), 1), device='cuda', generator=lg) * CFG['p_max'] if augment else torch.zeros((len(ids), 1), device='cuda')
    u = torch.rand((len(ids), n), device='cuda', generator=lg)
    bias = p * u * (2 * (b['y'][:, 1:2 * n + 1:2] - 1) - 1) if augment else torch.zeros_like(eps)
    prob = (eps + bias).sigmoid()
    w = torch.stack((1 - prob, prob), -1)
    pairs = torch.stack((w, w.flip(-1)), 2).flatten(1, 2)
    mix = torch.zeros((len(ids), b['x'].shape[1], 3), device='cuda')
    mix[:, :, 2] = 1.0
    mix[:, 1:2 * n + 1, :2] = pairs
    mix[:, 1:2 * n + 1, 2] = 0.0
    b['x'][:, 1:2 * n + 1] = 0
    b['mix'] = mix
    b['p'] = p[:, 0]
    return b

def ce_cases(z, b):
    n = b['n']
    loss = torch.nn.functional.cross_entropy(z[:, :2 * n + 1].float().transpose(1, 2), b['y'][:, :2 * n + 1], reduction='none')
    return (loss * b['loss_mask']).sum(1) / b['loss_mask'].sum(1)

def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
