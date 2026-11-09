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
    boards = []
    for t in range(4097):
        boards.append(m.lm_head(m.H_level(h + l)).argmax(-1)[i, 1:])
        if t < 4096:
            l = m.L_level(l + h + x)
    boards = torch.stack(boards).cpu().numpy()
    sparse = np.load(out / f'compact_{alpha:g}.npz')
    assert np.array_equal(boards[sparse['steps']], sparse['boards'])
    np.savez_compressed(out / f'dense_{alpha:g}.npz', boards=boards, errors=(boards != d['targets'][i, 1:]).sum(1))
    print(alpha, 'dense replay equal', flush=True)
run()
