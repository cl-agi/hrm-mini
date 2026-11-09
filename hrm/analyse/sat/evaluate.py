"""Evaluate frozen SAT weights with formula-specific random starts.

Write aggregate solved counts only. The supplied official IDs determine RNG seeds;
changing the batching does not change the per-start seed assignment.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hrm.paths import repo_path
from hrm import hub
from hrm.tasks.sat_flow import SATTask,sat_satisfies

def load_checkpoint(checkpoint, training_config=None, *, weight_state='raw', device='cpu'):
    """Use saved local training configuration and select weights explicitly."""
    checkpoint=Path(checkpoint)
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    source='explicit' if training_config is not None else None
    if training_config is None and 'config' in state:
        training_config=state['config'];source='checkpoint.config'
    if training_config is None:
        for directory in (checkpoint.parent,checkpoint.parent.parent if checkpoint.parent.name=='weights' else checkpoint.parent):
            for filename in ('resolved_config.json','model_config.json'):
                path=directory/filename
                if path.is_file():
                    training_config=json.loads(path.read_text());source=str(path);break
            if training_config is not None:break
    if training_config is None:
        training_config=hub.sat_training_config();source='published_default'
    if 'seed' not in training_config:
        raise ValueError('SAT evaluation requires a frozen seed; use the prepared resolved_config.json rather than the training template')
    model=SATTask(training_config,checkpoint.parent,device=device).model()
    weights=state.get('model',state)
    if weight_state=='ema':
        if 'ema' not in state:raise ValueError('EMA requested but checkpoint contains no EMA parameter state')
        if set(state['ema'])!=set(dict(model.named_parameters())):
            raise ValueError('EMA parameter keys differ from model configuration')
        weights=dict(weights,**state['ema'])
    elif weight_state!='raw':raise ValueError('weight_state must be raw or ema')
    model.load_state_dict(weights,strict=True)
    return model,training_config,source


@torch.inference_mode()
def evaluate(config, training_config=None, data_root=None, *, cycles, starts=64, limit=None, checkpoint=None, device="cuda", weight_state="raw"):
    if config.get("model_family", "hrm") != "hrm" and checkpoint is None and not config.get("checkpoint"):
        raise ValueError("Unpublished model families require an explicit checkpoint")
    data_root = hub.sat_public_data(data_root, n_vars=config.get("n_vars", 1024))
    checkpoint = checkpoint or config.get("checkpoint")
    if checkpoint is not None and not Path(checkpoint).exists() and str(checkpoint).startswith("checkpoint/"):
        checkpoint = repo_path(checkpoint)
    checkpoint = hub.checkpoint("sat", checkpoint)
    model,training_config,config_source=load_checkpoint(checkpoint,training_config,weight_state=weight_state,device=device)
    task=SATTask(training_config,data_root,device=device)
    model.eval()
    datasets={d:dict(np.load(Path(data_root)/f'data_{d}.npz')) for d in config['data_hashes']}
    results={}
    for density_index,(density,a) in enumerate(datasets.items()):
        if a['answers'].any():raise ValueError('Public evaluation must not supply answer cues')
        total=len(a['clauses']) if limit is None else min(limit,len(a['clauses']));solved=0
        for case in range(total):
            official_id=int(a['official_ids'][case]);found=False
            for lo in range(0,starts,config['microbatch']):
                seeds=[config['seed']+density_index*100000000+official_id*64+s for s in range(lo,min(starts,lo+config['microbatch']))]
                rows=[task.batch(a,[case],seed,False) for seed in seeds]
                b={k:torch.cat([row[k] for row in rows]) for k in ('x','mask','mix')}
                task.bind(model,b['mask']);carry=model.initial_carry|{'mixture_weights':b['mix']}
                clauses=torch.tensor(a['clauses'][case],device=device).expand(len(rows),-1,-1)
                n=a['answers'].shape[1]
                for cycle in range(cycles):
                    carry,logits=model(carry,b['x']);pred=logits[:,1:2*n+1:2].argmax(-1)
                    valid=sat_satisfies(clauses,pred==2)&((pred==1)|(pred==2)).all(-1)
                    if valid.any():found=True;break
                if found:break
            solved+=found
        results[density]=dict(solved=solved,total=total,accuracy=solved/total)
    return dict(inference_cycles=cycles,starts=starts,model=training_config.get('model_family','hrm'),subsets=results,
                checkpoint=str(checkpoint),weight_state=weight_state,training_config_source=config_source,training_seed=training_config['seed'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path);p.add_argument('--training-config',type=Path)
    p.add_argument('--checkpoint','--ckpt','--checkpoint-dir','--ckpt-dir',type=Path)
    p.add_argument('--data-root','--data','--dataset-path',type=Path)
    p.add_argument('--n-vars',type=int,choices=[512,1024],default=1024)
    p.add_argument('--device',default='cuda')
    p.add_argument('--weight-state',choices=['raw','ema'],default='raw')
    p.add_argument('--cycles',type=int,default=1024);p.add_argument('--starts',type=int,default=64)
    p.add_argument('--limit',type=int);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 0<a.starts<=64 or a.cycles<1 or (a.limit is not None and a.limit<1):raise ValueError('Invalid inference budget')
    config=json.loads(a.config.read_text()) if a.config else hub.sat_eval_config(a.n_vars)
    training=json.loads(a.training_config.read_text()) if a.training_config else None
    result=evaluate(config,training,a.data_root,cycles=a.cycles,starts=a.starts,limit=a.limit,checkpoint=a.checkpoint,device=a.device,weight_state=a.weight_state)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
