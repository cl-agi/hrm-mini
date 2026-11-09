"""Post-hoc energy ranking against labels, never used for selecting predictions."""
import json
import numpy as np
import torch
from hrm.analyse.baselines.ired.run import ROOT, model
from hrm_ired.sat_dataset import SudokuRRNDataset

torch.set_num_threads(4)
ckpt = torch.load(ROOT/'run_seed42/latest.pt', map_location='cpu', weights_only=False)
m = model().cuda().eval()
m.load_state_dict({k.removeprefix('ema_model.'):v for k,v in ckpt['ema'].items() if k.startswith('ema_model.')})
ds = SudokuRRNDataset('sudoku-rrn','test')
indices = np.random.default_rng(20261005).permutation(len(ds))[:1024]
pred = np.load(ROOT/'run_seed42/rrn.predictions.npy')
rows=[]
with torch.no_grad():
    for start in range(0,len(indices),64):
        ids=indices[start:start+64]
        x,y,mask=[torch.stack([ds[int(i)][j] for i in ids]).float().cuda() for j in range(3)]
        p=torch.nn.functional.one_hot(torch.tensor(pred[ids].astype('int64')-1,device='cuda'),9).float().reshape(-1,729)*2-1
        t=torch.zeros(len(ids),device='cuda',dtype=torch.long)
        scale=m.sqrt_alphas_cumprod[0]
        ep=m.model(x,p*scale,t,return_energy=True).flatten()
        ey=m.model(x,y*scale,t,return_energy=True).flatten()
        correct=(p==y).all(1)
        for i,c,a,b in zip(ids,correct.cpu().tolist(),ep.cpu().tolist(),ey.cpu().tolist()):
            rows.append(dict(index=int(i),correct=c,pred_energy=a,target_energy=b))
failed=[r for r in rows if not r['correct']]
out=dict(n=len(rows),failed=len(failed),wrong_lower_energy_than_target=sum(r['pred_energy']<r['target_energy'] for r in failed),
         note='Discrete predicted and true boards at t=0 scaled by sqrt(alpha_bar[0]); diagnostic only, not inference-time target use.',rows=rows)
(ROOT/'diagnostics/energy_ranking.json').write_text(json.dumps(out,indent=2))
print({k:v for k,v in out.items() if k!='rows'},flush=True)
