"""Prepare puzzle-disjoint E30 cohorts and extract frozen old-H probe features."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.training.data import puzzle_directory, local_directory
from hrm.models.ablations import HRM
from hrm.analyse.dynamics.jacobian.models import load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records, build_loader, dataset_sha256
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from hrm.analyse.dynamics.answer_direction.train_secondary_hl_readout import model_hash
from hrm.analyse.sudoku.core.old_h_recovery import extract
from hrm.analyse.sudoku.core.no_hh_mediation import initial_state, low_update, commit


def select_split(train, test, seed, counts, tokens):
    rng=np.random.default_rng(seed);seen=set();ids=[]
    for i in rng.permutation(len(test)):
        question=test[i].question
        if question not in seen:seen.add(question);ids.append(int(i))
        if len(ids)==sum(counts):break
    if len(ids)!=sum(counts):raise ValueError('Not enough distinct test puzzles')
    trained={r.question for r in train}
    if any(test[i].question in trained for i in ids):raise ValueError('Probe cohort overlaps backbone training puzzles')
    positions=np.array([rng.choice(np.arange(1,82),tokens,replace=False) for _ in ids])
    return np.array(ids),positions


@torch.no_grad()
def check_forward(model, x):
    emb=model.embed(x);h,l=initial_state(model,x)
    native,_=model(dict(z_H=h.clone(),z_L=l.clone()),x)
    for _ in range(model.H_cycles):l=low_update(model,h,l,emb);h=commit(model,h,l)
    if not torch.equal(h,native['z_H']) or not torch.equal(l,native['z_L']):raise RuntimeError('Manual recurrence differs from native forward')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/memory/records/old_h'))/'features')
    p.add_argument('--hh-link',choices=['native','retained','deleted'],default='native')
    p.add_argument('--seed',type=int);p.add_argument('--train-count',type=int,default=1024)
    p.add_argument('--val-count',type=int,default=256);p.add_argument('--test-count',type=int,default=512)
    p.add_argument('--tokens',type=int,default=16);p.add_argument('--steps',type=int,nargs='+',default=[1,8,24])
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    counts=[a.train_count,a.val_count,a.test_count]
    if min(*counts,a.tokens,a.batch_size,*a.steps)<1 or a.tokens>81 or a.steps!=sorted(set(a.steps)):p.error('Invalid counts, cells or update steps')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    out=a.output_dir;old=json.loads((out/'run_config.json').read_text()) if (out/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else old['seed'] if old else int(time.time())
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config);name=arch.pop('name','')
    if name not in ('hrm@HRM','hrm.models.hrm@HRM','hrm_ablations@HRM','hrm.models.ablations@HRM'):p.error('HRM checkpoint required')
    if arch.get('sym_io',False) or not arch.get('dual_module',True) or arch.get('is_mlp_mixer',False) or arch.get('rope_steps') is not None:p.error('E30 requires standard dual-module attention with asymmetric I/O and stationary RoPE')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6):p.error('Sudoku 2H6L required')
    trained_hh=arch.get('hh_link',True)
    if a.hh_link!='native':arch['hh_link']=a.hh_link=='retained'
    data=local_directory(a.dataset) if a.dataset else puzzle_directory('sudoku')
    train=read_test_records(data/'train.csv');test=read_test_records(data)
    ids,positions=select_split(train,test,seed,counts,a.tokens)
    plan=dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),config=config,evaluation_arch=arch,
        trained_hh_link=trained_hh,evaluation_hh_link=arch.get('hh_link',True),intervention=a.hh_link,
        train_dataset_sha256=dataset_sha256(train),test_dataset_sha256=dataset_sha256(test),
        counts=counts,tokens=a.tokens,steps=a.steps,batch_size=a.batch_size,device=a.device,
        board_ids=ids.tolist(),positions=positions.tolist(),code_sha256={str(f):sha(Path(f)) for f in [__file__,extract.__code__.co_filename]})
    if old is not None and old!=plan:raise ValueError('Changed inputs/settings; use new output directory')
    if old is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True);(out/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (out/'started.json').open('x') as f:json.dump(dict(started_unix=time.time()),f)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    before=model_hash(model)
    inputs,_,_=next(iter(build_loader(test,range(len(test)),len(test))))
    check_forward(model,inputs[ids[:a.batch_size]].to(a.device))
    np.savez_compressed(out/'split.npz',indices=ids,positions=positions,split=np.repeat(['train','validation','test'],counts))
    saved=extract(model,inputs.numpy(),ids,positions,out,a.steps,a.device,a.batch_size)
    if not model.hh_link:
        for step in a.steps:
            if not torch.equal(saved[f'{step}/h'],saved[f'{step}/bare_old_h']) or not torch.equal(saved[f'{step}/next_h'],saved[f'{step}/bare_next_h']):raise RuntimeError('No-HH identity failed')
    if model_hash(model)!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise RuntimeError('Frozen backbone changed')
    report=dict(plan,status='complete',native_forward_parity=True,no_hh_identity_checked=not model.hh_link,
        unchanged_backbone_sha256=before,artifacts={n:sha(out/n) for n in ('split.npz','features.pt')},
        note='Features only. Readout train/validation/test are disjoint puzzles; each step is a completed H phase (6 L updates), not a full 2H6L inference cycle. Link deletion is an inference intervention, not a separately trained no-HH model.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
