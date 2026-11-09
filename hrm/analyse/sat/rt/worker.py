from hrm.paths import analysis_directory
from pathlib import Path
import os, sys, time, json, sqlite3, traceback
import torch, numpy as np, hashlib
P = analysis_directory(__file__)
cfg = json.loads((P / 'config.json').read_text())
D = Path(cfg['source_run'])
pass
import hrm.analyse.sat.rt.common as cm
cfg = json.loads((P / 'config.json').read_text())
assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest() == cfg['checkpoint_sha256']
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
db = sqlite3.connect(P / 'queue.sqlite', timeout=120)
m = cm.model()
ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
m.load_state_dict(ck['model'])
del ck
m.eval()
run = torch.compile(m, dynamic=False, fullgraph=True)
datasets = {d: dict(np.load(P / f'data_{d}.npz')) for d in ['4.2', '4.3']}
assert all((not a['answers'].any() for a in datasets.values()))
out = P / f'rank{rank}'
out.mkdir(exist_ok=True)
for tag, sha in cfg['data_hashes'].items():
    assert hashlib.sha256((P / f'data_{tag}.npz').read_bytes()).hexdigest() == sha
while True:
    if (P / 'PAUSE').exists():
        break
    db.execute('BEGIN IMMEDIATE')
    row = db.execute("SELECT id,density,case_id,lo,hi FROM tasks WHERE status='pending' ORDER BY lo,id LIMIT 1").fetchone()
    if row is None:
        db.commit()
        break
    task, density, i, lo, hi = row
    a = datasets[density.split('_o')[0]]
    outer = int(density.split('_o')[1])
    db.execute("UPDATE tasks SET status='running',gpu=? WHERE id=?", (rank, task))
    db.commit()
    t = time.time()
    hit = 0
    end = lo
    sample_forwards = 0
    try:
        for cursor in range(lo, hi, 16):
            best = db.execute('SELECT MIN(hit) FROM tasks WHERE case_id=? AND density=? AND hit>0', (i, density)).fetchone()[0]
            if best is not None and cursor >= best:
                break
            ss = np.arange(cursor, min(cursor + 16, hi))
            ps = np.r_[ss, np.repeat(ss[0], 16 - len(ss))]
            rows = [cm.batch(a, [i], 801000 + int(a['official_ids'][i]) + 100000 * int(s), False) for s in ps]
            b = {k: torch.cat([r[k] for r in rows]) for k in ['x', 'mask', 'mix']}
            del rows
            cm.bind(m, b['mask'])
            carry = m.initial_carry | {'mixture_weights': b['mix']}
            c = np.broadcast_to(a['clauses'][i], (len(ss),) + a['clauses'][i].shape).astype(np.int64)
            cg = torch.tensor(c, device='cuda')
            first_outer = np.zeros(len(ss), np.int32)
            witness = np.zeros((len(ss), cfg['n_vars']), np.uint8)
            for it in range(1, outer + 1):
                carry, z = run(carry, b['x'])
                pred = z[:len(ss), 1:2 * cfg['n_vars'] + 1:2].argmax(-1)
                good = (cm.sat_satisfies(cg, pred == 2) & ((pred == 1) | (pred == 2)).all(-1)).cpu().numpy()
                new = good & (first_outer == 0)
                if new.any():
                    candidate = pred.to(torch.uint8).cpu().numpy()
                    sat = np.zeros(c.shape[:2], bool)
                    for j in range(3):
                        sat |= np.take_along_axis(candidate == 2, abs(c[:, :, j]) - 1, 1) == (c[:, :, j] > 0)
                    assert np.array_equal(good, sat.all(1) & np.isin(candidate, [1, 2]).all(1))
                    first_outer[new] = it
                    witness[new] = candidate[new]
            sample_forwards += 16 * outer
            pp = z[:len(ss), 1:2 * cfg['n_vars'] + 1:2].argmax(-1).to(torch.uint8).cpu().numpy()
            pp[first_outer > 0] = witness[first_outer > 0]
            sat = np.zeros(c.shape[:2], bool)
            for j in range(3):
                sat |= np.take_along_axis(pp == 2, abs(c[:, :, j]) - 1, 1) == (c[:, :, j] > 0)
            valid = first_outer > 0
            assert np.array_equal(valid, sat.all(1) & np.isin(pp, [1, 2]).all(1))
            np.savez_compressed(out / f'task{task}_block{cursor}.npz', ids=np.full(len(ss), i), official_id=int(a['official_ids'][i]), starts=ss, pred=pp, valid=valid, first_valid_outer=first_outer, violated=(~sat).sum(1))
            end = int(ss[-1]) + 1
            if valid.any():
                hit = int(ss[np.flatnonzero(valid)[0]]) + 1
                break
        db.execute("UPDATE tasks SET status='done',hit=?,checked_end=?,seconds=?,sample_forwards=? WHERE id=?", (hit, end, time.time() - t, sample_forwards, task))
        if hit:
            db.execute("UPDATE tasks SET status='skipped' WHERE status='pending' AND case_id=? AND density=? AND lo>=?", (i, density, hit))
        db.commit()
        print(json.dumps(dict(gpu=rank, density=density, task=task, case=i, lo=lo, end=end, hit=hit, seconds=time.time() - t)), flush=True)
    except Exception:
        db.execute("UPDATE tasks SET status='failed' WHERE id=?", (task,))
        db.commit()
        raise
(P / f'rank{rank}_complete.json').write_text(json.dumps(dict(complete=True, time=time.time())))
