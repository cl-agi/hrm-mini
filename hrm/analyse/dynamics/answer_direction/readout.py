"""Prepare, fit and evaluate the E22 frozen-backbone shared H+L readout."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hrm import hub
from hrm.paths import checkpoint_config, repository_root
from hrm.training.data import puzzle_directory, local_directory
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .train_secondary_hl_readout import digits, model_hash, gather, fit, evaluate_batch, CYCLES


def split_data(train, test, seed, train_count=800, val_count=200, test_count=256, excluded=()):
    if min(train_count,val_count,test_count)<1 or train_count+val_count>len(train):
        raise ValueError('Invalid split sizes')
    normalize=lambda frame:set(frame.question.str.replace('.','0',regex=False))
    if normalize(train)&normalize(test) or set(train.answer)&set(test.answer):
        raise ValueError('Training/test puzzles or solutions overlap')
    rng=np.random.default_rng(seed);ids=rng.permutation(len(train))
    tr,va=ids[:train_count],ids[train_count:train_count+val_count]
    if set(train.iloc[tr].answer)&set(train.iloc[va].answer):
        raise ValueError('Readout train/validation solutions overlap')
    excluded=set(excluded)
    candidates=[i for i,q in enumerate(test.question) if q.replace('.','0') not in excluded]
    if len(candidates)<test_count:raise ValueError('Not enough untouched test puzzles')
    heldout=rng.choice(candidates,test_count,replace=False)
    return dict(train_rows=tr.tolist(),validation_rows=va.tolist(),test_rows=heldout.tolist(),
                excluded_test_rows=[i for i,q in enumerate(test.question) if q.replace('.','0') in excluded],seed=seed)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--exclude-cohort',type=Path,help='Exploratory NPZ cohort with inputs/input, including BOS')
    p.add_argument('--output-dir',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/answer_direction'))/'readout')
    p.add_argument('--seed',type=int);p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--train-count',type=int,default=800);p.add_argument('--val-count',type=int,default=200)
    p.add_argument('--test-count',type=int,default=256);p.add_argument('--horizon',type=int,default=64)
    p.add_argument('--cells-per-board',type=int,default=24);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--half-batch',type=int,default=2048);p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if min(a.steps,a.horizon,a.cells_per_board,a.batch_size,a.half_batch)<1 or a.cells_per_board>81 or a.horizon>64:p.error('Invalid training/cache sizes')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    out=a.output_dir;old=json.loads((out/'run_config.json').read_text()) if (out/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else old['seed'] if old else int(time.time())
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config)
    if arch.pop('name','') not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False):p.error('Standard HRM required')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6) or arch.get('rope_steps') is not None:p.error('Stationary Sudoku 2H6L required')
    data=local_directory(a.dataset) if a.dataset else puzzle_directory('sudoku')
    train_file=data/'train.csv';test_file=data/'test_hard.csv'
    if not test_file.exists():test_file=data/'test.csv'
    train=pd.read_csv(train_file,dtype=str);test=pd.read_csv(test_file,dtype=str)
    # Validate schema/digits before tensor construction, preserving leading zeros.
    from hrm.analyse.dynamics.jacobian.data import read_test_records
    read_test_records(train_file);read_test_records(test_file)
    excluded=[]
    if a.exclude_cohort:
        with np.load(a.exclude_cohort,allow_pickle=False) as f:
            inputs=f['inputs'] if 'inputs' in f.files else f['input']
            if inputs.ndim!=2 or inputs.shape[1]!=82 or not np.isin(inputs[:,1:],np.arange(10)).all():raise ValueError('Expected encoded Sudoku cohort [N,82]')
            excluded=[''.join(map(str,row[1:].astype(int))) for row in inputs]
    splits=split_data(train,test,seed,a.train_count,a.val_count,a.test_count,excluded)
    plan=dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),config=config,
        train_sha256=sha(train_file),test_sha256=sha(test_file),splits=splits,
        exploratory_cohort_sha256=sha(a.exclude_cohort) if a.exclude_cohort else None,
        steps=a.steps,horizon=a.horizon,cells_per_board=a.cells_per_board,batch_size=a.batch_size,
        half_batch=a.half_batch,device=a.device,feature_cycles=[c for c in CYCLES if c<=a.horizon],
        code_sha256={f:sha(Path(__file__).parent/f) for f in ('readout.py','train_secondary_hl_readout.py')})
    if old is not None and old!=plan:raise ValueError('Changed inputs/settings; use new output')
    if old is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True);(out/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (out/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    before=model_hash(model);tx,ty=digits(train);xx,yy=digits(test)
    rng=np.random.default_rng(seed);caches={}
    for name,key in [('train','train_rows'),('validation','validation_rows')]:
        caches[name]=gather(model,tx,ty,np.array(splits[key]),rng,name,a.device,a.horizon,a.cells_per_board,a.batch_size)
        torch.save(caches[name],out/f'{name}_features.pt')
    saved,selected=fit(caches['train'],caches['validation'],a.steps,out,seed,a.device,a.half_batch)
    del caches
    weights={n:h['weight'].to(a.device) for n,h in saved.items()};heldout=np.array(splits['test_rows']);chunks=[]
    for start in range(0,len(heldout),a.batch_size):
        ii=heldout[start:start+a.batch_size];chunks.append(evaluate_batch(model,weights,xx[ii],yy[ii],a.horizon,a.device))
    payload=dict(ids=heldout,input=xx[heldout],target=yy[heldout],steps=chunks[0]['steps'],rms=np.concatenate([c['rms'] for c in chunks]))
    for n in ['native',*weights]:
        payload[f'{n}_pred']=np.concatenate([c['pred'][n] for c in chunks]);payload[f'{n}_ce']=np.concatenate([c['ce'][n] for c in chunks])
    np.savez_compressed(out/'heldout_trajectories.npz',**payload)
    if model_hash(model)!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise RuntimeError('Backbone changed')
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    report=dict(plan,status='complete',selected=selected,unchanged_backbone_sha256=before,
        artifact_sha256={f.name:sha(f) for f in out.iterdir() if f.suffix in ('.pt','.npz')},
        limitation='New frozen-backbone readout experiment; heldout counts/directions need not equal the historical 253 puzzles/3360 eligible states')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
