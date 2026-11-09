from hrm.paths import analysis_directory
from pathlib import Path
import sys, json, time, hashlib, statistics, os
import numpy as np
import torch
P = analysis_directory(__file__)
cfg = json.loads((P / 'config.json').read_text())
pass
from hrm.analyse.sat.hrm import common as cm
torch.set_num_threads(2)
torch.set_grad_enabled(False)
torch.cuda.set_device(0)
m = cm.model()
ck = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
m.load_state_dict(ck['model'])
del ck
m.eval()
torch.manual_seed(cfg['seed'])
torch.cuda.manual_seed_all(cfg['seed'])
run = torch.compile(m, dynamic=False, fullgraph=True)
records = []
summaries = []
for nv, path in cfg['datasets'].items():
    nv = int(nv)
    z = np.load(path)
    a = {k: z[k] for k in ['clauses', 'answers']}
    a['answers'] = np.zeros_like(a['answers'])
    for budget in cfg['inference_cycles']:
        for j in range(cfg['warmups'] + cfg['cases']):
            idx = max(0, j - cfg['warmups'])
            b = cm.batch(a, [idx], cfg['seed'] + idx, False)
            cm.bind(m, b['mask'])
            carry = m.initial_carry | {'mixture_weights': b['mix']}
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(budget):
                carry, logits = run(carry, b['x'])
            pred = logits[:, 1:2 * nv + 1:2].argmax(-1)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            records.append(dict(n_variables=nv, inference_cycles=budget, case_index=idx, warmup=j < cfg['warmups'], ms=ms, sequence_length=b['x'].shape[1]))
        values = [r['ms'] for r in records if r['n_variables'] == nv and r['inference_cycles'] == budget and (not r['warmup'])]
        summary = dict(n_variables=nv, clauses=a['clauses'].shape[1], inference_cycles=budget, random_starts=1, batch_size=1, n=len(values), mean_ms=statistics.mean(values), sd_ms=statistics.stdev(values), median_ms=statistics.median(values))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        (P / 'results.json').write_text(json.dumps(dict(summaries=summaries, records=records, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), checkpoint_sha256=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest(), timing='CUDA-synchronized perf_counter; compiled model recurrence and final argmax; excludes encoding, transfer, loading and warmup'), indent=2))
