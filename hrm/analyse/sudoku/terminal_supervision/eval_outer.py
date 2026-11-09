from hrm.paths import repo_path, work_path
from pathlib import Path
import os, sys, json, time, hashlib
from pathlib import Path
import torch
import numpy as np
root = Path(str(work_path('terminal_supervision_seed1_500ep')))
pass
from hrm.models.hrm import HRM
from hrm.tasks.sudoku import create_dataloader
out = Path(str(work_path('terminal_outer32_probe')))
out.mkdir(exist_ok=True)
checkpoint = root / 'checkpoints/terminal_hrm_seed1_500ep_20261023/seed_1/epoch_0340_full.pt'
c = torch.load(checkpoint, map_location='cpu', weights_only=False)
config = c['config']
data = dict(config['data'])
data.pop('name')
data['dataset_name'] = str((root / data['dataset_name']).resolve())
loader, metadata = create_dataloader('test_hard', 96, rank=0, world_size=1, seed=1, **data)
model = HRM(config['arch'] | metadata)
model.load_state_dict(c['model'])
ids = [i for g in c['optimizer']['param_groups'] for i in g['params']]
assert len(ids) == len(list(model.parameters()))
with torch.no_grad():
    for param, i in zip(model.parameters(), ids):
        param.copy_(c['optimizer']['state'][i]['param_ema'])
del c
model = model.cuda().eval()
model = torch.compile(model, dynamic=False, fullgraph=True)
correct = {i: [] for i in range(1, 33)}
token_correct = {i: 0 for i in range(1, 33)}
n = 0
start = time.time()
with torch.inference_mode():
    for batch, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        carry = model.initial_carry
        for outer in range(1, 33):
            carry, logits = model(carry, x)
            pred = logits.argmax(-1)
            correct[outer].append((pred == y).all(-1).cpu().numpy())
            token_correct[outer] += int((pred[:, 1:] == y[:, 1:]).sum())
        n += len(y)
        if (batch + 1) % 25 == 0:
            print(json.dumps(dict(batches=batch + 1, n=n, seconds=time.time() - start)), flush=True)
assert n == 19968
arrays = {f'outer_{i}': np.concatenate(v) for i, v in correct.items()}
np.savez(out / 'per_puzzle_correctness.npz', **arrays)
result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), epoch=340, weights='ema', n=n, seconds=time.time() - start, results={str(i): dict(exact_correct=int(arrays[f'outer_{i}'].sum()), exact_match=float(arrays[f'outer_{i}'].mean()), per_position_accuracy=token_correct[i] / (n * 81)) for i in correct})
result['paired_8_16'] = dict(both_correct=int((arrays['outer_8'] & arrays['outer_16']).sum()), only8=int((arrays['outer_8'] & ~arrays['outer_16']).sum()), only16=int((~arrays['outer_8'] & arrays['outer_16']).sum()))
result['paired_16_32'] = dict(both_correct=int((arrays['outer_16'] & arrays['outer_32']).sum()), only16=int((arrays['outer_16'] & ~arrays['outer_32']).sum()), only32=int((~arrays['outer_16'] & arrays['outer_32']).sum()))
(out / 'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
