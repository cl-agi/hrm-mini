"""Eight initializations: energy selection and diagnostic pass@8, no label selection."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from hrm.analyse.baselines.ired.run import ROOT, model
from hrm_ired.sat_dataset import SudokuRRNDataset

torch.set_num_threads(4)
ckpt=torch.load(ROOT/'run_seed42/latest.pt',map_location='cpu',weights_only=False)
m=model().cuda().eval()
m.load_state_dict({k.removeprefix('ema_model.'):v for k,v in ckpt['ema'].items() if k.startswith('ema_model.')})
ds=SudokuRRNDataset('sudoku-rrn','test')
indices=np.random.default_rng(20261005).permutation(len(ds))[:256]
all_correct=[];all_energies=[]
for k in range(8):
    torch.manual_seed(42+k)
    correct=[];energies=[]
    for x,y,mask in DataLoader(Subset(ds,indices.tolist()),batch_size=64):
        x,y,mask=[v.float().cuda() for v in (x,y,mask)]
        out=m.sample(x,None,mask,batch_size=len(x))
        with torch.no_grad():
            e=m.model(x,out,torch.zeros(len(x),device='cuda',dtype=torch.long),return_energy=True).flatten()
            c=(out.reshape(-1,81,9).argmax(-1)==y.reshape(-1,81,9).argmax(-1)).all(1)
        correct.extend(c.cpu().tolist());energies.extend(e.cpu().tolist())
    all_correct.append(correct);all_energies.append(energies)
c=np.array(all_correct);e=np.array(all_energies)
selected=e.argmin(0)
result=dict(n=256,k=8,single_run_exact=c[0].mean(),energy_selected_exact=c[selected,np.arange(256)].mean(),
            diagnostic_pass_at_8=c.any(0).mean(),per_restart=c.mean(1).tolist(),
            note='pass@8 uses labels only for diagnostic scoring and is not deployable selection. Lowest continuous t=0 energy selects predictions without labels.')
(ROOT/'diagnostics/restart_probe.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
