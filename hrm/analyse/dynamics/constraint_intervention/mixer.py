from hrm.paths import work_path
from hrm.paths import repo_path, work_path
from pathlib import Path
from hrm.paths import repo_path
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
R = work_path('constraint_intervention') / 'mixer'
R.mkdir(parents=True, exist_ok=True)
OLD = Path(str(work_path('hrm/analyse/dynamics/constraint_landscape')))
S = Path(str(work_path('mixer_mechanism')))
cfg = json.loads((repo_path('data/dynamics/constraint_intervention/mixer_config.json')).read_text())
torch.manual_seed(cfg['seed'])
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
pass
from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
pass
from hrm.analyse.dynamics.mixer.common import load
from hrm.analyse.dynamics.constraint_intervention.population import count

def decode(x):
    return (x.astype(np.uint32) << 16).view(np.float32)

def excess(p):
    p = p.reshape(-1, 9, 9, 9)
    return (p.sum(-2) - 1).clamp_min(0).sum((-2, -1)) + (p.sum(-3) - 1).clamp_min(0).sum((-2, -1)) + (p.reshape(-1, 3, 3, 3, 3, 9).sum((-4, -2)) - 1).clamp_min(0).sum((-3, -2, -1))

def metric(t, p):
    t = np.asarray(t)
    p = np.asarray(p)
    return dict(n=len(t), pearson=float(np.corrcoef(t, p)[0, 1]), rmse=float(np.sqrt(np.mean((t - p) ** 2))), mae=float(np.mean(np.abs(t - p))), r2=float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def main():
    m, identity = load(2)
    cohort = np.load(S / 'beads_seed2/cohort.npz')
    ci = int(np.flatnonzero(cohort['ids'] == 11486)[0])
    fs = []
    ts = []
    for h in range(1, 7):
        raw = np.load(S / f'selected_seed2/H{h:03d}_case11486_bits.npy', mmap_mode='r')
        ctx = torch.tensor(decode(np.load(S / f'beads_seed2/H{h:03d}_boundary.npz')['H_before_bits'][ci]), device='cuda').bfloat16()
        for j in range(0, 4097, 128):
            l = torch.tensor(decode(raw[j:j + 128]), device='cuda').bfloat16()
            fs.append(l.float().mean(1).cpu().numpy())
            ts.append(count(m.lm_head(m.H_level(ctx + l))).cpu().numpy())
        print('TARGETS', h, flush=True)
    del m
    f = np.concatenate(fs).astype(np.float64)
    t = np.concatenate(ts).astype(np.float64)
    phase = np.repeat(np.arange(1, 7), 4097)
    step = np.tile(np.arange(4097), 6)
    block = np.minimum(step // 128, 31)
    split = np.where(block % 5 == 3, 1, np.where(block % 5 == 4, 2, 0))
    tr = split == 0
    x = torch.from_numpy(f[tr])
    y = torch.from_numpy(t[tr])
    mean = x.mean(0)
    ym = y.mean()
    sol = torch.linalg.lstsq(x - mean, (y - ym)[:, None], driver='gelsd')
    w = sol.solution[:, 0].numpy()
    bias = float(ym - mean @ sol.solution[:, 0])
    pred = f @ w + bias
    old = np.load(str(repo_path('checkpoint/dynamics/mixer/probe.npz')))
    op = f @ old['weight'] + old['bias']
    metrics = {name: dict(new=metric(t[split == i], pred[split == i]), old=metric(t[split == i], op[split == i])) for i, name in enumerate(['train', 'validation', 'test'])}
    per_phase = {str(h): dict(new=metric(t[(phase == h) & (split == 2)], pred[(phase == h) & (split == 2)]), old=metric(t[(phase == h) & (split == 2)], op[(phase == h) & (split == 2)])) for h in range(1, 7)}
    np.savez_compressed(R / 'probe.npz', weight=w, bias=bias, features=f, target=t, predicted=pred, old_predicted=op, phase=phase, step=step, split=split)
    print('FIT', json.dumps(metrics), flush=True)
    d = np.load(OLD / 'prepared.npz')
    model = load_global(OLD / 'kernel_model.pt')
    B = torch.tensor(d['B'], device='cuda')
    V = torch.tensor(d['L_V'], device='cuda')
    base = torch.tensor(d['L_base'], device='cuda')
    wt = torch.tensor(w, device='cuda')
    bt = torch.tensor(bias, device='cuda', dtype=torch.float64)
    SQ = 41984 ** 0.5

    def score(q):
        c = torch.tensor(model.predict(q), device='cuda')
        z = base + SQ * (torch.tensor(q, dtype=torch.float32, device='cuda') @ B.T + c[:, :V.shape[1]] @ V.T)
        return (z.reshape(-1, 82, 512).bfloat16().float().mean(1).double() @ wt + bt).cpu().numpy()
    regions = {}
    for name in ['global'] + [f'H{i}' for i in range(1, 7)]:
        g = np.load(OLD / f'grid_{name}.npz')
        q = np.c_[g['x'].ravel(), g['y'].ravel()]
        v = np.concatenate([score(q[j:j + 256]) for j in range(0, len(q), 256)]).reshape(g['x'].shape)
        assert np.isfinite(v).all()
        np.savez_compressed(R / f'grid_{name}.npz', x=g['x'], y=g['y'], probe=v)
        regions[name] = dict(min=float(v.min()), max=float(v.max()), negative_count=int((v < 0).sum()))
        print('GRID', name, regions[name], flush=True)
    q = d['q']
    lift_pred = np.concatenate([score(q[j:j + 256]) for j in range(0, len(q), 256)])
    np.savez_compressed(R / 'trajectory_audit.npz', q=q, phase=phase, step=step, target=t, native_probe=pred, lift_probe=lift_pred, split=split)
    report = dict(status='complete', case=11486, seed=cfg['seed'], checkpoint=identity, probe_fit='Unregularized float64 OLS on token-mean L (82 tokens incl BOS); training blocks only, NO full-data refit', split='Each phase: 128-step blocks; block modulo5=3 validation, =4 test, others train. Step4096 assigned block31. Temporal holdout is within the same correlated trajectory, not independent puzzle generalization.', target='raw argmax duplicate excess across27 units and digits1..9, no clue clamping; native BF16 H-lookahead', metrics=metrics, per_phase_test=per_phase, rank=int(sol.rank), singular_values_range=[float(sol.singular_values.min()), float(sol.singular_values.max())], native_to_lift_probe=metric(pred, lift_pred), lift_to_target_test=metric(t[split == 2], lift_pred[split == 2]), regions=regions, unchanged={'kernel_model_sha256': sha(OLD / 'kernel_model.pt'), 'prepared_sha256': sha(OLD / 'prepared.npz'), 'old_probe_sha256': sha(Path(str(repo_path('checkpoint/dynamics/mixer/probe.npz'))))}, limitations='In-trajectory supervised probe landscape. Kernel lift uses all trajectory states but no score labels; off-trajectory scores remain unconstrained. Phase H is used only for target generation, not probe features.')
    (R / 'audit.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', json.dumps(report['native_to_lift_probe']), flush=True)
main()
