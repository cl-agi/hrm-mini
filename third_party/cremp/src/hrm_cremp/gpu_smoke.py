"""Short fixed-noise overfit/throughput engineering probe, never a validation result."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from hrm_cremp.angle_data import AngleDataset, sha
from hrm_cremp.angle_diffusion import noised_dataset, noise_loss, generate
from hrm_cremp.angle_model import configuration, make_model, budget


def now():
    torch.cuda.synchronize();return time.perf_counter()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds-per-family',type=float,default=18)
    p.add_argument('--families',nargs='+',default=['hrm','rt','ringer'])
    p.add_argument('--batch',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ds=AngleDataset(a.data);ds.means=ds.training_means()
    batch=next(iter(DataLoader(noised_dataset(ds),batch_size=a.batch)))
    batch={k:v.cuda() for k,v in batch.items()};results={}
    for family in a.families:
        torch.manual_seed(1);torch.cuda.reset_peak_memory_stats()
        model=make_model(configuration(family,384,512,12,batch['atom_features'].shape[-1])).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.)
        model.train();rows=[];start=now()
        while now()-start<a.seconds_per_family:
            tick=now();opt.zero_grad(set_to_none=True)
            y=model(batch['corrupted'],batch['t'],attention_mask=batch['attn_mask'],atom_features=batch['atom_features'])
            loss=noise_loss(y,batch);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
            rows.append(dict(loss=float(loss.detach()),grad_norm=float(norm),seconds=now()-tick))
        elapsed=now()-start;model.eval();gs=now();samples=generate(model,ds,[0],2,314,'cuda')
        padded=np.full((2,18,12),np.nan,np.float32)
        for i,s in enumerate(samples):padded[i,:len(s)]=s
        np.savez(out/f'{family}_generated.npz',angles=padded,ids=np.repeat(ds.ids[:1],2),
                 graph_smiles=np.repeat(ds.arrays['graph_smiles'][:1],2),lengths=np.repeat(ds.lengths[:1],2),
                 train_cache_sha256=np.asarray(sha(a.data)))
        results[family]=dict(budget=budget(model),fixed_noise_overfit=True,updates=len(rows),
                             batch_examples=len(batch['corrupted']),
                             train_seconds=elapsed,generation_k2_seconds=now()-gs,
                             mean_update_seconds=elapsed/len(rows),first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],
                             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),trace=rows)
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({k:v for k,v in results[family].items() if k!='trace'}),flush=True)
        del model,opt;torch.cuda.empty_cache()


if __name__=='__main__':main()
