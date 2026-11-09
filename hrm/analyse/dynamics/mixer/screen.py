"""Screen a reproducible Mixer cohort for long-L trajectories (E25)."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm.paths import checkpoint_config, repository_root
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture, load_config, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def rank_candidates(summary, rows):
    candidates = []
    for value in summary:
        first = value['first_correct_H']
        if first is None or not 3 <= first <= 6 or not value['stable_after_first']:
            continue
        pre = [r for r in rows if r['case'] == value['case'] and 2 <= r['H_cycle'] < first]
        candidates.append(dict(value,
            clear_pre_solution_cycles=sum(r['tail_pc123'] >= .85 and r['tail_scatter_rms'] >= .02 for r in pre),
            pre_solution_mean_pc123=float(np.mean([r['tail_pc123'] for r in pre]))))
    return sorted(candidates, key=lambda c: (-c['clear_pre_solution_cycles'], -c['first_correct_H'],
                                           -c['pre_solution_mean_pc123'], c['case']))


@torch.inference_mode()
def screen(model, x, y, ids, output, l_steps=4096, h_steps=8, tail_steps=128):
    if not 1 <= tail_steps <= l_steps or h_steps < 1:
        raise ValueError('Require 1 <= tail steps <= L steps and positive H steps')
    emb = model.embed(x)
    h = model.zH_init.expand_as(emb).clone(); l = model.zL_init.expand_as(emb).clone()
    native, _ = model(dict(z_H=h.clone(), z_L=l.clone()), x)
    hh, ll = h.clone(), l.clone()
    for _ in range(2):
        for _ in range(6):
            ll = model.L_level(ll + hh + emb)
        hh = model.H_level(hh + ll)
    if not torch.equal(hh, native['z_H']) or not torch.equal(ll, native['z_L']):
        raise RuntimeError('Explicit recurrence does not match native 2H6L forward')
    rows = []
    for hc in range(1, h_steps+1):
        before = h.clone(); tail = []
        for step in range(1, l_steps+1):
            l = model.L_level(l+h+emb)
            if step > l_steps-tail_steps:
                tail.append(l.clone())
        h = model.H_level(h+l)
        pred = model.lm_head(h).argmax(-1)
        errors = (pred[:, 1:] != y[:, 1:]).sum(1).cpu().tolist()
        tails = torch.stack(tail)
        torch.save(dict(H_before=before.cpu(), H_after=h.cpu(), L_after=l.cpu(),
                        tail=tails.cpu(), predictions=pred.cpu()), output/f'H{hc:03d}.pt')
        for bi, case in enumerate(ids):
            centered = tails[:, bi].reshape(tail_steps, -1).double()
            centered -= centered.mean(0)
            ev = torch.linalg.eigvalsh(centered@centered.T).clamp_min(0)
            rows.append(dict(case=int(case), H_cycle=hc, errors=errors[bi],
                             tail_pc123=float(ev[-3:].sum()/ev.sum().clamp_min(1e-30)),
                             tail_scatter_rms=float(centered.square().mean().sqrt())))
    summary = []
    for case in ids:
        errors = [r['errors'] for r in rows if r['case'] == case]
        first = next((i+1 for i, e in enumerate(errors) if e == 0), None)
        summary.append(dict(case=int(case), errors=errors, first_correct_H=first,
                            stable_after_first=bool(first and all(e == 0 for e in errors[first-1:]))))
    return rows, summary, rank_candidates(summary, rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--config', type=Path)
    p.add_argument('--dataset', type=Path); p.add_argument('--output-dir', type=Path)
    p.add_argument('--seed', type=int); p.add_argument('--boards', type=int, default=32)
    p.add_argument('--board-ids', type=int, nargs='+')
    p.add_argument('--l-steps', type=int, default=4096); p.add_argument('--h-steps', type=int, default=8)
    p.add_argument('--tail-steps', type=int, default=128)
    p.add_argument('--device', default='cuda'); p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(argv)
    if min(a.boards, a.h_steps, a.tail_steps) < 1 or a.tail_steps > a.l_steps:
        p.error('Invalid population or update counts')
    output = a.output_dir or Path(os.environ.get('HRM_ANALYSIS_RECORDS', repository_root()/'exp/dynamics/records/mixer'))/'screen'
    plan_path = output/'run_config.json'
    old = json.loads(plan_path.read_text()) if plan_path.exists() else None
    seed = a.seed if a.seed is not None else (old['seed'] if old else int(time.time()))
    config = load_config(a.config or checkpoint_config(a.checkpoint))
    arch = analysis_architecture(config)
    if arch.pop('name', '') not in ('hrm@HRM', 'hrm.models.hrm@HRM') or not arch.get('is_mlp_mixer'):
        p.error('A Mixer HRM checkpoint/configuration is required')
    if arch.get('H_cycles') != 2 or arch.get('L_cycles') != 6 or arch.get('seq_len') != 82 or arch.get('vocab_size') != 10:
        p.error('Expected Sudoku Mixer with native 2H6L schedule')
    if arch.get('rope_steps') is not None:
        p.error('Stationary positional encoding is required')
    records = read_test_records(a.dataset if a.dataset is not None else puzzle_directory('sudoku'))
    ids = select_board_ids(len(records), boards=a.boards, offset=0, selection='random',
                           selection_seed=seed, explicit=a.board_ids)
    plan = dict(seed=seed, board_ids=list(ids), dataset_sha256=dataset_sha256(records),
                checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint), config=config,
                l_steps=a.l_steps, h_steps=a.h_steps, tail_steps=a.tail_steps, device=a.device,
                code_sha256=sha(Path(__file__)))
    if old is not None and old != plan:
        raise ValueError('Inputs/settings changed; choose a new output directory')
    if old is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2))
    if a.prepare_only:
        return 0
    with (output/'started.json').open('x') as f:
        json.dump({'started_unix': time.time()}, f)
    torch.set_num_threads(2)
    model = HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True)), strict=True)
    x, y, _ = next(iter(build_loader(records, ids, len(ids))))
    np.savez_compressed(output/'cohort.npz', ids=ids, inputs=x.numpy(), targets=y.numpy())
    rows, summary, candidates = screen(model, x.to(a.device), y.to(a.device), ids, output,
                                       a.l_steps, a.h_steps, a.tail_steps)
    for name, value in [('screen', rows), ('summary', summary), ('ranked_candidates', candidates)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False))
    manifest = dict(plan, status='complete', native_forward_parity='bitwise H and L',
                    precision=str(model.embed(x[:1].to(a.device)).dtype),
                    cohort_sha256=sha(output/'cohort.npz'),
                    record_sha256={f'{name}.json': sha(output/f'{name}.json') for name in
                                   ('screen', 'summary', 'ranked_candidates')},
                    boundary_sha256={f'H{i:03d}.pt': sha(output/f'H{i:03d}.pt') for i in range(1, a.h_steps+1)})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
