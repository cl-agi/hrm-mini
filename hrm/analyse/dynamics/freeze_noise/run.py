"""Frozen-state noise experiment with the original loop, seeds and cohort IDs."""
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import yaml
from hrm.paths import repo_path, work_path, checkpoint_config, dataset_path
from hrm.models.hrm import HRM
from hrm.tasks.sudoku import collate_fn
from .random_kicks import evaluate_random_kicks

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',choices=['original','noise_repeats','cohort_repeats'],default='original')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cfg=json.loads(repo_path(f'data/dynamics/freeze_noise/{args.protocol}.json').read_text())
    out=args.output or work_path('freeze_noise/'+args.protocol);out.mkdir(parents=True,exist_ok=True)
    ckpt=repo_path(cfg['checkpoint']);mc=yaml.safe_load(checkpoint_config(ckpt).read_text())
    torch.set_num_threads(4)
    with torch.device('cuda'):
        model=HRM(mc['arch'] | dict(vocab_size=10,seq_len=82,is_causal=False))
        model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True),assign=True)
        model=torch.compile(model,dynamic=False,fullgraph=True);model.eval()
    data=dataset_path(mc['data']['dataset_name'])/'test_hard.csv'
    all_rows=list(csv.DictReader(data.open()))
    runs=cfg.get('runs') or [dict(seed=s,row_ids=list(range(cfg['n']))) for s in cfg['seeds']]
    for run in runs:
        rows=[all_rows[i] for i in run['row_ids']]
        loader=[collate_fn(rows[i:i+cfg['batch_size']],False) for i in range(0,len(rows),cfg['batch_size'])]
        seed=run['seed'];torch.manual_seed(seed);np.random.seed(seed % 2**32)
        started=time.time()
        result=evaluate_random_kicks(model,loader,H_warmup=cfg['H_warmup'],H_cycles=cfg['H_cycles'],noise_scales=[float(x) for x in cfg['noise_scales']],output=out/f'per_puzzle_{seed}.npz')
        result.update(seed=seed,seconds=time.time()-started)
        (out/f'result_{seed}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    protocol=dict(config=cfg,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),torch=torch.__version__)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

if __name__=='__main__':main()
