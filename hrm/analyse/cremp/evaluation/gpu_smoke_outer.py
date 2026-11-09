"""Bounded engineering overfit/throughput check; not an experimental result."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import hrm.analyse.cremp.evaluation.bootstrap as bootstrap
import torch
from torch.utils.data import DataLoader
from hrm_cremp.angle_data import AngleDataset, sha
from hrm_cremp.angle_diffusion import noised_dataset, noise_loss, generate
from hrm_cremp.angle_model import configuration
from hrm.analyse.cremp.evaluation.outer_model import OuterDenoiser, budget

p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=64)
a=p.parse_args();torch.set_num_threads(1)
if a.output.exists():raise ValueError('Refusing to overwrite smoke')
a.output.mkdir(parents=True)
ds=AngleDataset(a.train);ds.means=ds.training_means()
torch.manual_seed(2001)
b=next(iter(DataLoader(noised_dataset(ds),batch_size=64,shuffle=True,
                      generator=torch.Generator().manual_seed(1001))))
noise_sha=hashlib.sha256(b['known_noise'].numpy().tobytes()+b['t'].numpy().tobytes()).hexdigest()
b={k:v.to('cuda') for k,v in b.items()};reports=[]
for family in ('hrm','rt'):
    for mode in ('retain','reset'):
        torch.manual_seed(1)
        c=configuration(family,384,512,12);c.update(outer_rounds=2,carry_mode=mode)
        model=OuterDenoiser(c).cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.)
        model.train();carry=None;losses=[];torch.cuda.synchronize();start=time.perf_counter()
        for step in range(1,a.updates+1):
            if step%2==1:carry=None
            for group in opt.param_groups:group['lr']=5e-4*min(step/500,(20000-step)/19500)
            opt.zero_grad(set_to_none=True)
            carry,pred=model.transition(b['corrupted'],b['t'],b['attn_mask'],b['atom_features'],carry)
            loss=noise_loss(pred,b);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        model.eval()
        with torch.no_grad():
            carry=None;prefix_losses=[]
            for r in (1,2):
                carry,pred=model.transition(b['corrupted'],b['t'],b['attn_mask'],b['atom_features'],carry)
                prefix_losses.append(float(noise_loss(pred,b)))
            sampled=generate(model,ds,[0],2,327681,'cuda')
            assert all(torch.isfinite(torch.as_tensor(s)).all() for s in sampled)
        reports.append(dict(family=family,carry_mode=mode,updates=a.updates,
            episodes=a.updates//2,loss_first=losses[0],loss_last=losses[-1],
            prefix_losses_after_smoke=prefix_losses,training_seconds=elapsed,
            updates_per_second=a.updates/elapsed,extrapolated_20k_training_only_seconds=elapsed*20000/a.updates,
            generated_candidates=len(sampled),budget=budget(model),noise_sha256=noise_sha))
        del model,opt,carry,pred;torch.cuda.empty_cache()
report=dict(engineering_only=True,device='cuda',train_sha256=sha(a.train),models=reports,
            note='All arms deliberately reuse one real batch/noise for a tiny overfit smoke. Production redraws every two updates. Extrapolation excludes validation/checkpoint/startup cost.')
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
