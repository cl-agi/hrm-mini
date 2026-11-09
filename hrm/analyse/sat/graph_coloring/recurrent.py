"""Evaluate recurrent SAT models on public graph-coloring inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root
from hrm.training.data import freeze_training_seed
from hrm.analyse.sat.evaluate import load_checkpoint
from hrm.tasks.sat_flow import SATTask
from .data import load_data
from .metrics import check_predictions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path,help='Saved model training configuration')
    p.add_argument('--dataset-path',type=Path);p.add_argument('--colors',type=int,choices=(3,4),default=3)
    p.add_argument('--cycles',type=int,default=1024);p.add_argument('--starts',type=int,default=64)
    p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,default=repository_root()/'exp/sat/records/graph_coloring/hrm')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if min(a.cycles,a.starts)<1 or (a.limit is not None and a.limit<1):p.error('Positive budgets required')
    d=load_data(a.colors,a.dataset_path);n=min(len(d['ids']),a.limit or len(d['ids']))
    cp=hub.checkpoint('sat',a.checkpoint)
    cfg=json.loads(a.config.read_text()) if a.config else None
    model,cfg,source=load_checkpoint(cp,cfg,device=a.device);model.eval()
    a.output.mkdir(parents=True,exist_ok=True);seed=freeze_training_seed(a.output)
    plan=dict(seed=seed,checkpoint=str(cp),model_config=cfg,config_source=source,colors=a.colors,
              inference_cycles=a.cycles,starts=a.starts,case_ids=d['ids'][:n],ordered_data_sha256=d['ordered_sha256'])
    frozen=a.output/'config.json'
    if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('Use a new output directory for changed settings')
    frozen.write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return
    with (a.output/'started.json').open('x') as f:json.dump({'seed':seed},f)
    task=SATTask(cfg,a.output,device=a.device)
    data=dict(clauses=d['clauses'],answers=np.zeros((len(d['ids']),d['variables']),np.int8))
    solved=np.zeros(n,dtype=bool)
    with torch.inference_mode():
        for i in range(n):
            for start in range(a.starts):
                b=task.batch(data,[i],seed+start*100000+i,False)
                task.bind(model,b['mask']);carry=model.initial_carry|{'mixture_weights':b['mix']}
                for _ in range(a.cycles):
                    carry,logits=model(carry,b['x'])
                    pred=logits[:,1:2*d['variables']+1:2].argmax(-1).cpu().numpy()[None]
                    good,_=check_predictions(pred,d['clauses'][i:i+1],d['edges'][i:i+1],a.colors)
                    if good[0,0,0]:solved[i]=True;break
                if solved[i]:break
    result=dict(configuration=plan,total=n,available=len(d['ids']),solved=int(solved.sum()),accuracy=float(solved.mean()))
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
