from hrm.paths import work_path
from hrm.paths import repo_path
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np, torch
R = Path(str(work_path('hrm/analyse/sudoku/core')))
pass
os.chdir(R)
from hrm.analyse.dynamics.perturbation_controls.verify_test_time_scaling import load_metadata

@torch.inference_mode()
def run():
    alpha = float(sys.argv[1])
    out = Path(sys.argv[2])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    a = argparse.Namespace(**json.loads((R / 'experiment/hrm_step_geometry_cases_20260908_v3/summary.json').read_text())['args'])
    a.device = 'cuda:0'
    m, meta = load_metadata(a)
    m.requires_grad_(False)
    p = R / 'experiment/L4096_case_search_20260928'
    d = np.load(p / 'cohort.npz')
    s = np.load(p / 'H004_context.npz')
    i = list(d['ids']).index(6566)

    def tensor(b):
        return torch.from_numpy(b.copy()).view(torch.bfloat16).cuda()
    old = tensor(s['H_before_bits'])
    new = tensor(s['H_after_bits'])
    l = tensor(s['L_after_bits'])
    h = (old.float() + alpha * (new.float() - old.float())).to(old.dtype)
    x = m.embed(torch.tensor(d['inputs'], device='cuda'))
    tail = tensor(np.load(p / 'H004_tail_bits.npy')[:, i]).float().flatten(1)
    pred = lambda z: m.lm_head(z).argmax(-1)[i, 1:]
    before = pred(old)
    after = pred(h)
    ys = d['targets'][i, 1:]
    boards = []
    steps = []
    nearest = []
    states = []
    last = l[i].float()
    rms = []
    for t in range(4097):
        if t <= 128 or t % 32 == 0:
            boards.append(pred(m.H_level(h + l)).cpu().numpy())
            steps.append(t)
            z = l[i].float().flatten()
            nearest.append(float((tail - z).square().mean(1).min().sqrt()))
            rms.append(float((l[i].float() - last).square().mean().sqrt()))
            states.append(l[i].float().cpu().numpy())
        last = l[i].float().clone()
        if t < 4096:
            l = m.L_level(l + h + x)
    h = m.H_level(h + l)
    follow = [int((pred(h).cpu().numpy() != ys).sum())]
    for t in range(32):
        for k in range(6):
            l = m.L_level(l + h + x)
        h = m.H_level(h + l)
        follow.append(int((pred(h).cpu().numpy() != ys).sum()))
    np.savez_compressed(out / f'detail_{alpha:g}.npz', steps=steps, boards=boards, nearest=nearest, step_rms=rms, states=states, before=before.cpu().numpy(), after=after.cpu().numpy(), targets=ys, inputs=d['inputs'][i, 1:], follow_errors=follow)
    err = (np.array(boards) != ys).sum(1)
    hit = np.flatnonzero(err == 0)
    print(json.dumps(dict(alpha=alpha, immediate_changes=int((before != after).sum()), relative_H_rms=float(((old[i].float() + alpha * (new[i].float() - old[i].float())).to(old.dtype).float() - old[i].float()).square().mean().sqrt() / old[i].float().square().mean().sqrt()), first_correct_probe=int(steps[hit[0]]) if len(hit) else None, tail_errors_range=[int(err[-32:].min()), int(err[-32:].max())], follow_errors=follow)), flush=True)
run()
