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
def main():
    c = int(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    a = argparse.Namespace(**json.loads((R / 'experiment/hrm_step_geometry_cases_20260908_v3/summary.json').read_text())['args'])
    a.device = 'cuda:0'
    m, meta = load_metadata(a)
    m.requires_grad_(False)
    p = R / 'experiment/L4096_case_search_20260928'
    d = np.load(p / 'cohort.npz')
    s = np.load(p / f'H{c:03d}_context.npz')

    def tensor(b):
        return torch.from_numpy(b.copy()).view(torch.bfloat16).cuda()
    old = tensor(s['H_before_bits'])
    new = tensor(s['H_after_bits'])
    l0 = tensor(s['L_after_bits'])
    x = m.embed(torch.tensor(d['inputs'], device='cuda'))
    assert torch.equal(m.H_level(old + l0), new)
    pred = lambda h: m.lm_head(h).argmax(-1)[:, 1:]
    b0 = pred(old)
    b1 = pred(new)
    ys = d['targets'][:, 1:]
    result = {'cycle': c, 'model': meta, 'ids': d['ids'].tolist(), 'immediate_changed': (b0 != b1).sum(1).tolist(), 'h_relative_rms': ((new.float() - old.float()).square().mean((1, 2)).sqrt() / old.float().square().mean((1, 2)).sqrt()).tolist(), 'before': b0.tolist(), 'after': b1.tolist()}
    arrays = {'inputs': d['inputs'][:, 1:], 'targets': ys, 'ids': d['ids'], 'before': b0.cpu().numpy(), 'after': b1.cpu().numpy()}
    for name, h in [('old', old), ('new', new)]:
        l = l0.clone()
        boards = []
        steps = []
        dist = []
        for t in range(4097):
            if t <= 32 or t % 32 == 0:
                boards.append(pred(m.H_level(h + l)).cpu().numpy())
                steps.append(t)
                dist.append((l.float() - l0.float()).square().mean((1, 2)).sqrt().cpu().numpy())
            if t < 4096:
                l = m.L_level(l + h + x)
        arrays[name + '_boards'] = np.stack(boards)
        arrays[name + '_distance'] = np.stack(dist)
        if name == 'new':
            nxt = np.load(p / f'H{c + 1:03d}_context.npz')
            result['next_L_bitwise_equal'] = np.array_equal(l.view(torch.uint16).cpu().numpy(), nxt['L_after_bits'])
            result['next_H_bitwise_equal'] = np.array_equal(m.H_level(h + l).view(torch.uint16).cpu().numpy(), nxt['H_after_bits'])
        print(c, name, 'done', flush=True)
    arrays['steps'] = np.array(steps)
    np.savez_compressed(out / f'cycle{c}.npz', **arrays)
    for name in ['old', 'new']:
        err = (arrays[name + '_boards'] != ys).sum(-1)
        result[name + '_final_errors'] = err[-1].tolist()
        result[name + '_tail_min_errors'] = err[-32:].min(0).tolist()
        result[name + '_tail_max_errors'] = err[-32:].max(0).tolist()
    (out / f'cycle{c}.json').write_text(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
