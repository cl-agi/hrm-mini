"""Stage-limited SAT evaluation; synchronous outer steps and per-case carries."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import numpy as np
import torch
P = Path(sys.argv[1]).resolve()
rank = int(sys.argv[2])
cfg = json.loads((P / 'config.json').read_text())
stage = json.loads((P.parent / 'stage.json').read_text())['outer']
pass
from hrm.analyse.sat.hrm import common as cm
torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest() == cfg['checkpoint_sha256']
for name, sha in cfg['source_sha256'].items():
    assert hashlib.sha256((Path(cfg['source_run']) / name).read_bytes()).hexdigest() == sha
data = {}
for density, sha in cfg['data_hashes'].items():
    path = P / f'data_{density}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    data[density] = dict(np.load(path))
    assert not data[density]['answers'].any()
m = cm.model()
ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
m.load_state_dict(ck['model'])
del ck
m.eval()
run = torch.compile(m, dynamic=False, fullgraph=True)
db = sqlite3.connect(P / 'queue.sqlite', timeout=120)
out = P / f'worker{rank}'
out.mkdir(exist_ok=True)
(P / 'results').mkdir(exist_ok=True)
(P / 'states').mkdir(exist_ok=True)

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value.to(device) if torch.is_tensor(value) else value

def pause():
    return (P / 'PAUSE').exists() or (P.parent / 'PAUSE').exists() or shutil.disk_usage('/').free < 15 * 1024 ** 3
while not pause():
    db.execute('BEGIN IMMEDIATE')
    row = db.execute("SELECT id,density,case_id FROM tasks WHERE status='running' AND worker=?", (rank,)).fetchone()
    if row is None:
        row = db.execute("SELECT id,density,case_id FROM tasks WHERE status IN ('pending','stage_done') AND covered_outer<? ORDER BY covered_outer,id LIMIT 1", (stage,)).fetchone()
    if row is None:
        db.commit()
        write(out / 'status.json', dict(state='stage_complete', stage=stage, time=time.time()))
        break
    task, density, case = row
    db.execute("UPDATE tasks SET status='running',worker=? WHERE id=?", (rank, task))
    db.commit()
    result_path = P / f'results/task{task}.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['first_valid_outer'] > 0 or previous.get('assessed_outer', 0) >= stage:
            db.execute('UPDATE tasks SET status=?,covered_outer=? WHERE id=?', ('done' if previous['first_valid_outer'] > 0 else 'stage_done', previous.get('assessed_outer', cfg['outer']), task))
            db.commit()
            continue
    a = data[density]
    official_id = int(a['official_ids'][case])
    seeds = [cfg['seed'] + list(data).index(density) * 100000000 + official_id * 64 + s for s in range(64)]
    batches = []
    for lo in range(0, 64, cfg['microbatch']):
        rows = [cm.batch(a, [case], s, False) for s in seeds[lo:lo + cfg['microbatch']]]
        b = {k: torch.cat([r[k] for r in rows]) for k in ('x', 'mask', 'mix')}
        batches.append(dict(b=b, carry=m.initial_carry | {'mixture_weights': b['mix']}, outer=0))
    del rows, b
    first = np.zeros(64, np.int32)
    witness = np.zeros((64, cfg['n_vars']), np.uint8)
    forwards = 0
    elapsed = 0.0
    state_path = P / f'states/task{task}.pt'
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        assert saved['task'] == task
        first, witness = (saved['first'], saved['witness'])
        forwards, elapsed = (saved['forwards'], saved['elapsed'])
        for bb, saved_bb in zip(batches, saved['batches']):
            bb['carry'] = move(saved_bb['carry'], 'cuda')
            bb['outer'] = saved_bb['outer']
        del saved
    clause = a['clauses'][case].astype(np.int64)
    cg = torch.tensor(np.broadcast_to(clause, (cfg['microbatch'],) + clause.shape).copy(), device='cuda')
    begin = time.time()
    write(out / 'status.json', dict(state='running', task=task, density=density, official_id=official_id, outer=min((bb['outer'] for bb in batches)), time=time.time()))
    start = min((bb['outer'] for bb in batches))
    assert start < stage and (not first.any())
    targets = sorted((t for t in set(cfg['budgets'] + list(range(256, stage + 1, 256)) + [stage]) if start < t <= stage))
    for target in targets:
        best = 0
        for it in range(min((bb['outer'] for bb in batches)) + 1, target + 1):
            for bi, bb in enumerate(batches):
                if bb['outer'] >= it:
                    continue
                b = bb['b']
                cm.bind(m, b['mask'])
                bb['carry'], z = run(bb['carry'], b['x'])
                pred = z[:, 1:2 * cfg['n_vars'] + 1:2].argmax(-1)
                good = (cm.sat_satisfies(cg, pred == 2) & ((pred == 1) | (pred == 2)).all(-1)).cpu().numpy()
                forwards += cfg['microbatch']
                bb['outer'] = it
                new = good & (first[bi * cfg['microbatch']:(bi + 1) * cfg['microbatch']] == 0)
                if new.any():
                    pp = pred.to(torch.uint8).cpu().numpy()
                    sat = np.zeros((len(pp), len(clause)), bool)
                    for lit in range(3):
                        sat |= (pp[:, abs(clause[:, lit]) - 1] == 2) == (clause[:, lit] > 0)
                    assert np.array_equal(good, sat.all(1) & np.isin(pp, [1, 2]).all(1))
                    positions = np.flatnonzero(new) + bi * cfg['microbatch']
                    first[positions] = it
                    witness[positions] = pp[new]
                    best = it
                    break
            if best:
                break
        finished = best > 0
        covered = min((bb['outer'] for bb in batches))
        stage_done = not finished and covered >= stage
        duration = elapsed + time.time() - begin
        if not finished:
            tmp = state_path.with_suffix('.tmp')
            torch.save(dict(task=task, first=first, witness=witness, forwards=forwards, elapsed=duration, batches=[dict(carry=move(bb['carry'], 'cpu'), outer=bb['outer']) for bb in batches]), tmp)
            tmp.replace(state_path)
        npz_tmp = P / f'results/task{task}.tmp.npz'
        np.savez_compressed(npz_tmp, first_valid_outer=first, witness=witness, evaluated_outer=np.array([bb['outer'] for bb in batches]), seeds=np.array(seeds))
        npz_tmp.replace(result_path.with_suffix('.npz'))
        write(result_path, dict(task=task, density=density, case_id=case, official_id=official_id, seed=cfg['seed'], first_valid_outer=best, assessed_outer=covered, starts=64, sample_outer_forwards=forwards, elapsed_seconds=duration, trajectory_first_hits='censored at first formula hit or assessed_outer', complete=finished or covered >= cfg['outer'], time=time.time()))
        db.execute('UPDATE tasks SET status=?,covered_outer=? WHERE id=?', ('done' if finished else 'stage_done' if stage_done else 'running', covered, task))
        db.commit()
        status = dict(state='task_complete' if finished else 'stage_task_complete' if stage_done else 'paused' if pause() else 'running', task=task, density=density, official_id=official_id, outer=min((bb['outer'] for bb in batches)), stage=stage, elapsed_seconds=duration, sample_outer_forwards=forwards, time=time.time())
        write(out / 'status.json', status)
        print(json.dumps(status), flush=True)
        if finished or stage_done or pause():
            break
    del batches, b, bb, cg, z, pred
