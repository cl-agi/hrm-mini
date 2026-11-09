from hrm.paths import repo_path, work_path
from pathlib import Path
import csv, hashlib, importlib, json, sys
import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from hrm.paths import repo_path, work_path, checkpoint_config
BASE = work_path('answer_error')
ROOT = BASE.parents[2]
RUNTIME = ROOT / 'exp/00_shared_runtime'
pass
NAMES = ['hrm', 'tied', 'rt', 'stage_shared', 'untied_wide', 'untied_param']
FOLDERS = ['baseline_upstream_8gpu_tuned_hrm_trainpy', 'h3_formal_tied_same_width', 'h3_formal_homogeneous_param_flop', 'stage_factor_shared_deep', 'stage_factor_untied_deep', 'stage_factor_untied_parammatch_deep']
SEED = 2026090732

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def js(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def csvout(p, rows):
    if not rows:
        return
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with Path(p).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def prepare():
    out = BASE / 'inputs'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        return
    data = repo_path('data/sudoku/sudoku-extreme-1k')
    rows = list(csv.DictReader((data / 'test_hard.csv').open()))
    train = list(csv.DictReader((data / 'train.csv').open()))
    assert len(rows) == 20000 and len(train) == 1000
    ids = np.random.default_rng(SEED).permutation(len(rows))[:1024]
    assert not {rows[i]['question'] for i in ids} & {r['question'] for r in train}
    assert len({rows[i]['question'] for i in ids}) == 1024
    x = torch.tensor([[0] + [int(c) for c in rows[i]['question'].replace('.', '0')] for i in ids])
    y = torch.tensor([[0] + [int(c) for c in rows[i]['answer']] for i in ids])
    assert x.shape == y.shape == (1024, 82) and torch.all((x == 0) | (x == y))
    for board in y[:, 1:].reshape(-1, 9, 9):
        for unit in list(board) + list(board.T) + [board[r:r + 3, c:c + 3].flatten() for r in (0, 3, 6) for c in (0, 3, 6)]:
            assert torch.equal(unit.sort().values, torch.arange(1, 10))
    split = torch.cat([torch.zeros(512, dtype=torch.long), torch.ones(256, dtype=torch.long), torch.full((256,), 2)])
    torch.save({'x': x, 'y': y, 'ids': torch.tensor(ids), 'split': split}, out / 'data.pt')
    meta = {'protocol': 'fixed original seed, 1024 unique puzzles, 512/256/256 split', 'data_sha256': sha(out / 'data.pt'), 'csv_sha256': sha(data / 'test_hard.csv'), 'rng_seed': SEED, 'models': {}, 'split_sizes': [512, 256, 256], 'evaluation_is_existing_development_data': True}
    for name, folder in zip(NAMES, FOLDERS):
        for s in [1, 2, 3]:
            f = folder + ('_seeds23' if name == 'hrm' and s > 1 else '')
            matches = sorted(repo_path('checkpoint/sudoku').glob(f'**/{f}/seed_{s}/epoch_19.pt'))
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError(f'Ambiguous selected checkpoint: {f} seed {s}')
            cp = matches[0]
            cfg = checkpoint_config(cp)
            meta['models'][f'{name}_s{s}'] = {'checkpoint': str(cp), 'config': str(cfg), 'checkpoint_sha256': sha(cp), 'config_sha256': sha(cfg), 'config_content': yaml.safe_load(cfg.read_text())}
    js(out / 'manifest.json', meta)
    csvout(out / 'split.csv', [{'ordinal': i, 'row_id': int(idx), 'split': ['train', 'calibration', 'evaluation'][int(split[i])]} for i, idx in enumerate(ids)])

def model_load(name, seed, dtype='float32', device='cuda:0'):
    prepare()
    meta = json.loads((BASE / 'inputs/manifest.json').read_text())
    info = meta['models'][f'{name}_s{seed}']
    assert sha(info['checkpoint']) == info['checkpoint_sha256'] and sha(info['config']) == info['config_sha256']
    cfg = info['config_content']['arch'] | {'forward_dtype': dtype, 'vocab_size': 10, 'seq_len': 82, 'is_causal': False}
    mod, cls = cfg['name'].split('@')
    with torch.device(device):
        m = getattr(importlib.import_module(__import__('hrm.registry',fromlist=['resolve_module']).resolve_module('arch.' + mod)), cls)(cfg)
    m.load_state_dict(torch.load(info['checkpoint'], map_location=device, weights_only=True), strict=True)
    m.eval()
    m.requires_grad_(False)
    return (m, info)

def is_stage(m):
    return hasattr(m, 'stages')

def streams(m):
    return ['H', 'L'] if hasattr(m, 'H_level') else ['z']

def schedule(m):
    return ([1, 3, 5, 7], 3, 7) if is_stage(m) else ([2, 4, 8, 16], 4, 16)

def initial(m, b):
    return {k: v.expand(b, 82, -1).clone() for k, v in m.initial_carry.items()}

def key(s):
    return 'z_' + s if s in ['H', 'L'] else 'z'

def read(m, c):
    return m.lm_head(c['z_H'] if 'z_H' in c else c['z'])

def step(m, c, x, t):
    emb = m.embed(x)
    if hasattr(m, 'H_level'):
        h, l = (c['z_H'], c['z_L'])
        for _ in range(m.H_cycles):
            for _ in range(m.L_cycles):
                l = m.L_level(l + h + emb)
            h = m.H_level(h + l)
        return {'z_H': h, 'z_L': l}
    z = c['z']
    if is_stage(m):
        z = m._core_at(t - 1)(z + emb)
    else:
        for _ in range(m.cycles):
            z = m.core(z + emb)
    return {'z': z}

def baseline_features(logits, x):
    p = logits.softmax(-1)
    sort = p.sort(-1, descending=True).values
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1, keepdim=True)
    margin = sort[..., :1] - sort[..., 1:2]
    pred = torch.nn.functional.one_hot(p.argmax(-1), 10).float()
    pos = torch.arange(82).float()[None, :, None].expand(len(x), -1, -1)
    row = (pos - 1).clamp_min(0) // 9 / 8
    col = (pos - 1).clamp_min(0) % 9 / 8
    clues = x[:, 1:].ne(0).float().sum(-1)[:, None, None].expand(-1, 82, 1) / 81
    return torch.cat([sort, entropy, margin, pred, clues, row, col], -1)

def ridge(X, y, mode='hidden'):
    X = X.double()
    y = y.double()
    mu = X.mean(0)
    centered = X - mu
    scale = (centered.square().mean(0).sqrt() if mode == 'baseline' else centered.square().mean().sqrt()).clamp_min(1e-06)
    z = centered / scale
    c = z.T @ z / len(z)
    a = c + torch.eye(c.shape[0], dtype=c.dtype) * 0.01
    w = torch.linalg.solve(a, z.T @ (y - y.mean()) / len(z)) / scale
    return {'w': w, 'b': y.mean() - mu @ w, 'mean': mu}

def score(fit, X):
    return (X.double() @ fit['w'] + fit['b']).numpy()

def calibrate(scores, y):
    scores = np.asarray(scores)
    y = np.asarray(y, dtype=float)
    mu = float(scores.mean())
    sd = max(float(scores.std()), 1e-06)
    z = (scores - mu) / sd
    if y.min() == y.max():
        return {'a': 0.0, 'b': float(np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))), 'mu': mu, 'sd': sd, 'success': True}

    def fn(ab):
        v = ab[0] * z + ab[1]
        p = expit(v)
        return (np.mean(np.logaddexp(0, v) - y * v) + 0.0001 * np.sum(ab ** 2), np.array([np.mean((p - y) * z), np.mean(p - y)]) + 0.0002 * ab)
    r = minimize(fn, [1.0, 0.0], jac=True, method='L-BFGS-B', bounds=[(0, None), (None, None)])
    return {'a': float(r.x[0]), 'b': float(r.x[1]), 'mu': mu, 'sd': sd, 'success': bool(r.success)}

def probability(cal, s):
    return expit(cal['a'] * (s - cal['mu']) / cal['sd'] + cal['b'])

def auc(y, s):
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def metrics(y, s, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(p, 1e-08, 1 - 1e-08)
    order = np.argsort(-np.asarray(s), kind='stable')
    ys = y[order]
    ss = np.asarray(s)[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / y.sum()) if y.sum() else None
    ece = 0.0
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p >= a) & (p < b)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {'n': len(y), 'positive': int(y.sum()), 'auroc': auc(y, s), 'ap': ap, 'brier': float(np.mean((p - y) ** 2)), 'nll': float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))), 'ece': float(ece)}

def terminal(logits, x, y):
    blank = x.eq(0)
    blank[:, 0] = False
    bad = logits.argmax(-1).ne(y)
    ce = -logits.log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)
    return {'exact': ~(bad & blank).any(-1), 'accuracy': 1 - (bad & blank).sum(-1) / blank.sum(-1), 'nll': (ce * blank).sum(-1) / blank.sum(-1), 'pred': logits.argmax(-1)}

def rotate(z, direction, Q, mask, theta):
    row = z @ Q @ Q.T
    u = z - row
    radius = u.norm(dim=-1, keepdim=True)
    direction = direction - direction @ Q @ Q.T
    tangent = direction - (u * direction).sum(-1, keepdim=True) * u / radius.square().clamp_min(1e-20)
    tn = tangent.norm(dim=-1, keepdim=True)
    v = tangent / tn.clamp_min(1e-20)
    rotated = row + u * np.cos(theta) + radius * v * np.sin(theta)
    return torch.where(mask[..., None] & (tn > 1e-10) & (radius > 1e-10), rotated, z)
if __name__ == '__main__':
    prepare()
