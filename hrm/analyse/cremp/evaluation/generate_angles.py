"""Read-only checkpoint sampling. Source checkpoints and training logs are never edited."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from hrm_cremp.angle_data import AngleDataset, sha
from hrm_cremp.angle_model import make_model, budget
from hrm_cremp.angle_diffusion import generate

p=argparse.ArgumentParser();p.add_argument('--checkpoint','--ckpt','--checkpoint-dir','--ckpt-dir');p.add_argument('--data','--dataset-path')
p.add_argument('--split',default='validation')
p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=16);p.add_argument('--limit',type=int)
p.add_argument('--batch-molecules',type=int,default=16);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=327681)
a=p.parse_args()
from hrm import hub
a.checkpoint=str(hub.checkpoint('cremp',a.checkpoint))
a.data=str(hub.cremp_data(a.data,split=a.split))
torch.set_num_threads(1);out=Path(a.output)
if out.exists():raise ValueError('Refusing to overwrite inference evidence')
out.mkdir(parents=True)
original_sha=sha(a.checkpoint);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=make_model(state['config']).to(a.device);model.load_state_dict(state['model']);model.eval()
ds=AngleDataset(a.data,state['means']);n=min(a.limit or len(ds.ids),len(ds.ids));samples=[]
if str(a.device).startswith('cuda'):torch.cuda.synchronize()
start=time.perf_counter()
for first in range(0,n,a.batch_molecules):
    samples.extend(generate(model,ds,list(range(first,min(first+a.batch_molecules,n))),a.k,a.seed+first,a.device))
if str(a.device).startswith('cuda'):torch.cuda.synchronize()
elapsed=time.perf_counter()-start
padded=np.full((len(samples),18,12),np.nan,np.float32)
for i,sample in enumerate(samples):padded[i,:len(sample)]=sample
np.savez(out/'generated_angles.npz',angles=padded,ids=np.repeat(ds.ids[:n],a.k),
         graph_smiles=np.repeat(ds.arrays['graph_smiles'][:n],a.k),lengths=np.repeat(ds.lengths[:n],a.k),
         train_cache_sha256=np.asarray(state['train_cache_sha256']),checkpoint_sha256=np.asarray(original_sha),
         candidate_index=np.tile(np.arange(a.k),n),
         rng_batch_seed=np.repeat(a.seed+(np.arange(n)//a.batch_molecules)*a.batch_molecules,a.k))
if state.get('geometry') is not None:
    (out/'generated_angles.geometry.json').write_text(json.dumps(state['geometry'],indent=2)+'\n')
assert sha(a.checkpoint)==original_sha
manifest=dict(args=vars(a),checkpoint_sha256=original_sha,data_sha256=sha(a.data),
              checkpoint_step=state['step'],budget=budget(model),inference_seconds=elapsed,
              candidates=len(samples),molecules=n,k=a.k,reference_angles_used=False,
              denoiser_candidate_calls=20*len(samples),
              actual_batched_denoiser_calls=20*((n+a.batch_molecules-1)//a.batch_molecules),
              rng='Fixed CPU initial noise and CUDA ancestral noise seed per batch, same batch/K shape across arms')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
