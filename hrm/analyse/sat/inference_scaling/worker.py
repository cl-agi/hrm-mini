from hrm.paths import analysis_directory
from pathlib import Path
import os, sys, json, time, hashlib
import numpy as np
import torch
P = analysis_directory(__file__)
cfg = json.loads((P / 'config.json').read_text())
pass
from hrm.analyse.sat.hrm import common as cm
rank = int(os.environ['LOCAL_RANK'])
world = int(os.environ.get('LOCAL_WORLD_SIZE', 4))
torch.cuda.set_device(rank)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest() == cfg['checkpoint_sha256']
assert hashlib.sha256((P / 'data_test.npz').read_bytes()).hexdigest() == cfg['data_sha256']
a = dict(np.load(P / 'data_test.npz'))
assert not a['answers'].any()
m = cm.model()
ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
m.load_state_dict(ck['model'])
del ck
m.eval()
run = torch.compile(m, dynamic=False, fullgraph=True)
ids = np.arange(len(a['clauses']))[rank::world]
n = len(ids)
limit = cfg['outer']
out = P / f'rank{rank}'
out.mkdir(exist_ok=True)

def write(name, obj):
    t = out / (name + '.tmp')
    t.write_text(json.dumps(obj, indent=2))
    t.replace(out / name)

def cpu(v):
    return {k: cpu(x) for k, x in v.items()} if isinstance(v, dict) else v.detach().cpu() if torch.is_tensor(v) else v

def gpu(v):
    return {k: gpu(x) for k, x in v.items()} if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v
batches = []
for j in range(0, n, 2):
    ii = ids[j:j + 2]
    padded = np.resize(ii, 2)
    rows = [cm.batch(a, [int(i)], 801000 + int(i), False) for i in padded]
    b = {k: torch.cat([r[k] for r in rows]) for k in ['x', 'mask', 'mix']}
    batches.append(dict(b=b, carry=m.initial_carry | {'mixture_weights': b['mix']}, j=j, count=len(ii), clauses=torch.tensor(a['clauses'][ii], device='cuda').long()))
first = np.zeros(n, np.int32)
witness = np.zeros((n, 1024), np.uint8)
solved = np.zeros((limit, n), bool)
violated = np.zeros((limit, n), np.uint16)
start = 0
elapsed = 0.0
state = out / 'resume.pt'
if state.exists():
    s = torch.load(state, map_location='cpu', weights_only=False)
    start = s['outer']
    elapsed = s['elapsed']
    first = s['first']
    witness = s['witness']
    solved[:start] = s['solved']
    violated[:start] = s['violated']
    for b, c in zip(batches, s['carries']):
        b['carry'] = gpu(c)
    del s
beg = time.time()
write('status.json', dict(status='running', outer=start, cases=n, ids=ids.tolist(), started=time.time()))
for lo in range(start, limit, 64):
    hi = min(lo + 64, limit)
    final = np.zeros((n, 1024), np.uint8)
    for bb in batches:
        b = bb['b']
        cm.bind(m, b['mask'])
        j = bb['j']
        k = bb['count']
        c = bb['clauses']
        for it in range(lo, hi):
            bb['carry'], z = run(bb['carry'], b['x'])
            pred = z[:k, 1:2049:2].argmax(-1)
            sat = torch.zeros(c.shape[:2], dtype=torch.bool, device='cuda')
            for lit in range(3):
                sat |= torch.gather(pred == 2, 1, c[:, :, lit].abs() - 1) == (c[:, :, lit] > 0)
            v = (~sat).sum(1).cpu().numpy()
            good = (sat.all(1) & ((pred == 1) | (pred == 2)).all(1)).cpu().numpy()
            solved[it, j:j + k] = good
            violated[it, j:j + k] = v
            new = good & (first[j:j + k] == 0)
            if new.any():
                pp = pred.to(torch.uint8).cpu().numpy()
                first[j:j + k][new] = it + 1
                witness[j:j + k][new] = pp[new]
            if it == hi - 1:
                final[j:j + k] = pred.to(torch.uint8).cpu().numpy()
    c = a['clauses'][ids]
    sat = np.zeros(c.shape[:2], bool)
    for lit in range(3):
        sat |= np.take_along_axis(final == 2, abs(c[:, :, lit]) - 1, 1) == (c[:, :, lit] > 0)
    assert np.array_equal((~sat).sum(1), violated[hi - 1])
    assert np.array_equal(sat.all(1) & np.isin(final, [1, 2]).all(1), solved[hi - 1])
    paused = (P / 'PAUSE').exists() or (P.parent / 'PAUSE').exists()
    if hi % 256 == 0 or hi == limit or paused:
        tmp = out / 'resume.tmp'
        torch.save(dict(outer=hi, elapsed=elapsed + time.time() - beg, carries=[cpu(bb['carry']) for bb in batches], first=first, witness=witness, solved=solved[:hi], violated=violated[:hi]), tmp)
        tmp.replace(state)
    if hi & hi - 1 == 0 or hi == limit:
        np.savez_compressed(out / f'outer{hi}.npz', ids=ids, outer=hi, final_pred=final, final_valid=solved[hi - 1], final_violated=violated[hi - 1], first_valid_outer=first, witness=witness)
    np.savez_compressed(out / 'trajectory.tmp.npz', ids=ids, solved=solved[:hi], violated=violated[:hi], first_valid_outer=first)
    (out / 'trajectory.tmp.npz').replace(out / 'trajectory.npz')
    status = dict(status='paused' if paused else 'complete' if hi == limit else 'running', outer=hi, cases=n, final_solved=int(solved[hi - 1].sum()), any_solved=int((first > 0).sum()), mean_violated=float(violated[hi - 1].mean()), elapsed_seconds=elapsed + time.time() - beg, time=time.time())
    write('status.json', status)
    print(json.dumps(dict(rank=rank, **status)), flush=True)
    if paused:
        break
