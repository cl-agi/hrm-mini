"""Read-only checkpoint sampling. Source checkpoints and training logs are never edited."""
import hrm.analyse.cremp.evaluation.bootstrap as bootstrap
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from hrm_cremp.angle_data import AngleDataset, sha
from hrm.analyse.cremp.evaluation.outer_model import OuterDenoiser, budget
from hrm_cremp.angle_diffusion import generate

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=16);p.add_argument('--limit',type=int)
p.add_argument('--batch-molecules',type=int,default=16);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=327681)
p.add_argument('--inference-rounds',type=int,choices=[1,2])
a=p.parse_args();torch.set_num_threads(1);out=Path(a.output)
if out.exists():raise ValueError('Refusing to overwrite inference evidence')
out.mkdir(parents=True)
original_sha=sha(a.checkpoint);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
trained_rounds=int(state['config'].get('outer_rounds',1));eval_config=dict(state['config'])
if a.inference_rounds is not None:eval_config['outer_rounds']=a.inference_rounds
model=OuterDenoiser(eval_config).to(a.device);model.load_state_dict(state['model']);model.eval()
transition_count=0; original_transition=model.transition
def counted_transition(*args,**kwargs):
    global transition_count
    transition_count+=1
    return original_transition(*args,**kwargs)
model.transition=counted_transition
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
assert sha(a.checkpoint)==original_sha
manifest=dict(args=vars(a),checkpoint_sha256=original_sha,data_sha256=sha(a.data),
              trained_outer_rounds=trained_rounds,evaluation_outer_rounds=model.outer_rounds,
              checkpoint_step=state['step'],budget=budget(model),inference_seconds=elapsed,
              candidates=len(samples),molecules=n,k=a.k,reference_angles_used=False,
              denoiser_candidate_calls=20*model.outer_rounds*len(samples),
              ddpm_wrapper_candidate_calls=20*len(samples),
              actual_batched_denoiser_calls=transition_count,
              actual_batched_ddpm_wrapper_calls=20*((n+a.batch_molecules-1)//a.batch_molecules),
              rng='Fixed CPU initial noise and CUDA ancestral noise seed per batch, same batch/K shape across arms')
assert transition_count==20*model.outer_rounds*((n+a.batch_molecules-1)//a.batch_molecules)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
