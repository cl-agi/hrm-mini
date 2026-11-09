from hrm.paths import work_path
from hrm.paths import repo_path, work_path
from pathlib import Path
from hrm.paths import repo_path
from pathlib import Path
import sys, json, argparse
import numpy as np
import torch
R = work_path('constraint_intervention')
pass
from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
from hrm.analyse.dynamics.constraint_intervention.population import count
pass
from hrm.analyse.dynamics.perturbation_controls.verify_test_time_scaling import load_metadata

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(json.loads((repo_path('data/dynamics/constraint_intervention/landscape_config.json')).read_text())['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    parts = [np.load(p) for p in sorted((R / 'population').glob('batch_*.npz'))]
    f = np.concatenate([p['features'] for p in parts])
    y = np.concatenate([p['target'] for p in parts])
    assert f.shape == (1056, 96, 1024)
    fits = {}
    pred = {}
    for name, sl in [('L', slice(0, 512)), ('HL', slice(None))]:
        x = torch.tensor(f[..., sl].reshape(-1, f[..., sl].shape[-1]), dtype=torch.float64)
        t = torch.tensor(y.ravel(), dtype=torch.float64)
        mu = x.mean(0)
        ym = t.mean()
        w = torch.linalg.lstsq(x - mu, (t - ym)[:, None], driver='gelsd').solution[:, 0]
        b = ym - mu @ w
        fits[name] = (w, b)
        pred[name] = (x @ w + b).numpy().reshape(y.shape)
    delta = np.concatenate([p['delta'] for p in parts])
    norm = np.concatenate([p['norm'] for p in parts])
    freeze = np.concatenate([p['freeze_count'] for p in parts])
    np.savez_compressed(R / 'population_counts.npz', target=y, predictions=pred['HL'], L_predictions=pred['L'], delta=delta, norm=norm, freeze_count=freeze)
    out = R / 'hrm'
    out.mkdir(parents=True, exist_ok=True)
    torch.save(fits, out / 'probes.pt')
    source = Path(str(work_path('case6566_global_lift/results')))
    fm = load_global(source / 'kernel_model.pt', device='cuda')
    d = np.load(source / 'prepared.npz')
    base, B, V = [torch.tensor(d[k], dtype=torch.float32, device='cuda') for k in ['L_base', 'B', 'L_V']]
    w, b = [t.cuda() for t in fits['L']]
    audit = {'fit': 'OLS on1056 ordered test puzzles at96 post-L events; descriptive in-sample', 'target': 'raw argmax duplicate excess, no clue clamping', 'checkpoint': '3a535185e4c0b4fcd03ba5c7241198047ed19b5bb5825af09f416ebc49816115', 'grids': {}}
    for name in ['global_padded', 'H1_context', 'H2', 'H3', 'H4']:
        g = np.load(source / 'existing_L_probe' / f'existing_L_probe_{name}.npz')
        q = np.c_[g['x'].ravel(), g['y'].ravel()]
        v = []
        for j in range(0, len(q), 256):
            z = q[j:j + 256]
            c = torch.tensor(fm.predict(z), device='cuda')
            l = base + 41984 ** 0.5 * (torch.tensor(z, dtype=torch.float32, device='cuda') @ B.T + c[:, :V.shape[1]] @ V.T)
            v.append((l.reshape(-1, 82, 512).bfloat16().float().mean(1).double() @ w + b).cpu().numpy())
        v = np.concatenate(v).reshape(g['x'].shape)
        np.savez_compressed(out / f'grid_{name}.npz', x=g['x'], y=g['y'], probe=v)
        audit['grids'][name] = {'min': float(v.min()), 'max': float(v.max())}
        print('HRM', name, flush=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    source = Path(str(work_path('rt_landscape/case35')))
    fm = load_global(source / 'kernel_model.pt', device='cuda')
    d = np.load(source / 'prepared.npz')
    base, B, V = [torch.tensor(d[k], dtype=torch.float32, device='cuda') for k in ['base', 'B', 'normal']]
    m, identity = load_metadata(argparse.Namespace(checkpoint=str(repo_path('checkpoint/sudoku/checkpoints/h3_formal_homogeneous_param_flop/seed_3/epoch_5.pt')), config=None, architecture=None, device='cuda'))
    m.eval().requires_grad_(False)
    out = R / 'rt'
    out.mkdir(parents=True, exist_ok=True)
    for name in ['global', 'local1', 'local2', 'local3', 'local4']:
        g = np.load(source / f'grid_{name}.npz')
        q = np.c_[g['x'].ravel(), g['y'].ravel()]
        v = []
        for j in range(0, len(q), 256):
            z = q[j:j + 256]
            c = torch.tensor(fm.predict(z), device='cuda')
            l = base + 41984 ** 0.5 * (torch.tensor(z, dtype=torch.float32, device='cuda') @ B.T + c @ V.T)
            v.append(count(m.lm_head(l.reshape(-1, 82, 512).bfloat16())).cpu().numpy())
        v = np.concatenate(v).reshape(g['x'].shape)
        np.savez_compressed(out / f'grid_{name}.npz', x=g['x'], y=g['y'], constraint_violation=v)
        print('RT', name, flush=True)
    (out / 'identity.json').write_text(json.dumps(identity, indent=2))
main()
