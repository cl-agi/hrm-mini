"""Fixed-budget 1k Sudoku training with rule-preserving online augmentation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ema_pytorch import EMA
from hrm import hub
from hrm.training.data import freeze_training_seed
from hrm.training.ired_core import model

def augment_pair(x,y,rng):
    digits=np.concatenate([[0],rng.permutation(np.arange(1,10))])
    rows=np.concatenate([b*3+rng.permutation(3) for b in rng.permutation(3)])
    cols=np.concatenate([b*3+rng.permutation(3) for b in rng.permutation(3)])
    if rng.integers(2):x,y=x.T,y.T
    return digits[x[np.ix_(rows,cols)]],digits[y[np.ix_(rows,cols)]]

class Train1k(Dataset):
    def __init__(self,augment,seed,path=None):
        self.path=hub.puzzle_data("sudoku",path,split="train")
        with self.path.open() as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==1000
        self.x=np.array([[int(c) for c in r['question'].replace('.','0')] for r in rows],dtype=np.int64).reshape(-1,9,9)
        self.y=np.array([[int(c) for c in r['answer']] for r in rows],dtype=np.int64).reshape(-1,9,9)
        self.augment=augment;self.rng=np.random.default_rng(seed)
    def __len__(self):return len(self.x)
    def __getitem__(self,i):
        x,y=self.x[i],self.y[i]
        if self.augment:x,y=augment_pair(x,y,self.rng)
        x=torch.from_numpy(x.copy()).flatten();y=torch.from_numpy(y.copy()).flatten()
        inp=torch.nn.functional.one_hot(x,10)[:,1:].float()*2-1
        target=torch.nn.functional.one_hot(y-1,9).float()*2-1
        mask=(x!=0)[:,None].expand(-1,9).float()
        return inp.flatten(),target.flatten(),mask.flatten()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--initialization',choices=['pretrained','scratch'],default='scratch')
    ap.add_argument('--augment',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--steps',type=int,default=50000)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--seed',type=int)
    ap.add_argument('--data-path',type=Path)
    ap.add_argument('--initial-checkpoint',type=Path)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skip-evaluation',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.steps<=0 or a.lr<=0:ap.error('steps and lr must be positive')
    if a.initialization=='pretrained' and a.initial_checkpoint is None:ap.error('--initial-checkpoint is required for pretrained initialization')
    if a.initialization=='scratch' and a.initial_checkpoint is not None:ap.error('--initial-checkpoint requires pretrained initialization')
    if a.seed is None:a.seed=freeze_training_seed(a.output)
    INITIAL=hub.local_file(a.initial_checkpoint) if a.initial_checkpoint else None
    ds=Train1k(a.augment,a.seed,a.data_path)
    TRAIN=ds.path
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    config=dict(initialization=a.initialization,augmentation=a.augment,steps=a.steps,lr=a.lr,
                seed=a.seed,train_n=len(ds),batch_size=64,drop_last=True,torch=torch.__version__,
                train_source=str(TRAIN),train_sha256=hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
                initial_checkpoint=str(INITIAL) if a.initialization=='pretrained' else None,
                initial_sha256=hashlib.sha256(INITIAL.read_bytes()).hexdigest() if a.initialization=='pretrained' else None,
                optimizer='fresh Adam, betas=(0.9,0.99)',ema='fresh EMA 0.995, update_every=10',
                augmentation_operations=['digit permutation','band/stack permutation','within-band/stack row/column permutation','transpose'] if a.augment else [],
                test_policy=f'Fixed {a.steps}-step endpoint; no test-based checkpoint selection. No clue revealing, solver, or test augmentation.')
    config_path=a.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Prepared configuration changed; use a new run directory')
    hub._write(config_path,lambda p:p.write_text(json.dumps(config,indent=2)))
    print(json.dumps(config),flush=True)
    if a.prepare_only:return
    with (a.output/'training_started.json').open('x') as f:json.dump({'seed':a.seed},f)
    m=model().to(a.device)
    if a.initialization=='pretrained':
        c=torch.load(INITIAL,map_location='cpu',weights_only=False)
        m.load_state_dict({k.removeprefix('ema_model.'):v for k,v in c['ema'].items() if k.startswith('ema_model.')},strict=True)
        del c
    opt=torch.optim.Adam(m.parameters(),lr=a.lr,betas=(.9,.99))
    ema=EMA(m,beta=.995,update_every=10).to(a.device)
    # Drop 40-sample tail so every gradient step has the declared 64 puzzles.
    loader=DataLoader(ds,batch_size=64,shuffle=True,drop_last=True,num_workers=0)
    start=time.monotonic();step=0;loss_sum=0.;last_t=start
    while step<a.steps:
        for batch in loader:
            m.train();x,y,mask=[v.to(a.device) for v in batch]
            opt.zero_grad(set_to_none=True)
            loss,_=m(x,y,mask)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step();ema.update();step+=1;loss_sum+=loss.item()
            if step%100==0:
                now=time.monotonic()
                status=dict(step=step,loss_mean_100=loss_sum/100,seconds=now-start,seconds_per_step=(now-last_t)/100)
                tmp=a.output/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.output/'status.json')
                print(json.dumps(status),flush=True);loss_sum=0.;last_t=now
            if step%10000==0 or step==a.steps:
                state=dict(step=step,model=m.state_dict(),ema=ema.state_dict(),opt=opt.state_dict(),
                           rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else [],augmentation_rng=ds.rng.bit_generator.state)
                tmp=a.output/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(a.output/'latest.pt')
            if step>=a.steps:break
    (a.output/'training_complete.json').write_text(json.dumps(dict(step=step,seconds=time.monotonic()-start)))
    if a.skip_evaluation:return
    # Release model memory before evaluation subprocesses use this same GPU.
    del state,m,ema,opt,x,y,mask,loss
    torch.cuda.empty_cache()
    desc=f"extreme-1k train1000, {a.steps} steps, initialization={a.initialization}, augmentation={a.augment}"
    base=[__import__('sys').executable,'-u','-m','hrm.analyse.baselines.ired.eval_extreme',
          '--checkpoint',str(a.output/'latest.pt'),'--output',str(a.output/'evaluation'),
          '--device',a.device,'--expected-step',str(a.steps),'--training-description',desc]
    for shard in range(8):
        subprocess.run(base+['--shard',str(shard)],check=True)
    subprocess.run(base+['--merge'],check=True)
    (a.output/'complete.json').write_text(json.dumps(dict(state='complete',total_seconds=time.monotonic()-start)))

if __name__=='__main__':main()
