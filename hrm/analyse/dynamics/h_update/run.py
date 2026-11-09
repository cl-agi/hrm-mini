"""E21: retain versus update H from the identical long-L boundary."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm import hub
from hrm.paths import checkpoint_config, repository_root
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


@torch.inference_mode()
def boundary(model, x, phase=4, l_steps=4096, tail_steps=128):
    if phase < 1 or not 1 <= tail_steps <= l_steps:
        raise ValueError('Invalid boundary schedule')
    emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
    native,_=model(dict(z_H=h.clone(),z_L=l.clone()),x)
    hh,ll=h.clone(),l.clone()
    for _ in range(2):
        for _ in range(6):ll=model.L_level(ll+hh+emb)
        hh=model.H_level(hh+ll)
    if not torch.equal(hh,native['z_H']) or not torch.equal(ll,native['z_L']):
        raise RuntimeError('Native 2H6L forward parity failed')
    for hc in range(1,phase+2):
        before=h.clone();tail=[]
        for step in range(1,l_steps+1):
            l=model.L_level(l+h+emb)
            if hc==phase and step>l_steps-tail_steps:tail.append(l.clone())
        h=model.H_level(h+l)
        if hc==phase:
            result=dict(old=before,new=h.clone(),l0=l.clone(),tail=torch.stack(tail))
    result.update(next_H=h,next_L=l)
    return result


@torch.inference_mode()
def branch(model,x,y,state,index,alpha,l_steps=4096,follow_h=32):
    if alpha not in (0,1):raise ValueError('E21 compares alpha=0 and alpha=1')
    old,new,l0=state['old'],state['new'],state['l0']
    h=(old.float()+alpha*(new.float()-old.float())).to(old.dtype)
    if alpha==1 and not torch.equal(h,new):
        raise RuntimeError('Interpolated update differs from the native H state')
    l=l0.clone();emb=model.embed(x);tail=state['tail'][:,index].float().flatten(1)
    before=model.lm_head(old).argmax(-1)[index,1:]
    after=model.lm_head(h).argmax(-1)[index,1:]
    boards=[];steps=[];nearest=[];rms=[];states=[];last=l[index].float().clone()
    for t in range(l_steps+1):
        boards.append(model.lm_head(model.H_level(h+l)).argmax(-1)[index,1:].cpu().numpy())
        if t<=128 or t%32==0 or t==l_steps:
            steps.append(t);z=l[index].float()
            nearest.append(float((tail-z.flatten()).square().mean(1).min().sqrt()))
            rms.append(float((z-last).square().mean().sqrt()))
            states.append(z.cpu().numpy())
        last=l[index].float().clone()
        if t<l_steps:l=model.L_level(l+h+emb)
    h=model.H_level(h+l)
    if alpha==1 and (not torch.equal(l,state['next_L']) or not torch.equal(h,state['next_H'])):
        raise RuntimeError('Updated branch differs from independent next native long-L phase')
    follow=[int((model.lm_head(h).argmax(-1)[index,1:]!=y[index,1:]).sum())]
    for _ in range(follow_h):
        for _ in range(6):l=model.L_level(l+h+emb)
        h=model.H_level(h+l)
        follow.append(int((model.lm_head(h).argmax(-1)[index,1:]!=y[index,1:]).sum()))
    boards=np.asarray(boards);target=y[index,1:].cpu().numpy();errors=(boards!=target).sum(1)
    hit=np.flatnonzero(errors==0)
    relative=float((after!=before).sum())
    arrays=dict(boards=boards,errors=errors,steps=np.asarray(steps),nearest=np.asarray(nearest),
                step_rms=np.asarray(rms),states=np.asarray(states),before=before.cpu().numpy(),
                after=after.cpu().numpy(),targets=target,inputs=x[index,1:].cpu().numpy(),follow_errors=np.asarray(follow))
    summary=dict(alpha=alpha,immediate_changes=int(relative),
        relative_H_rms=float(((old[index].float()+alpha*(new[index].float()-old[index].float())).to(old.dtype).float()-old[index].float()).square().mean().sqrt()/old[index].float().square().mean().sqrt().clamp_min(1e-30)),
        first_correct_dense_step=int(hit[0]) if len(hit) else None,
        tail_errors_range=[int(errors[-min(1024,len(errors)):].min()),int(errors[-min(1024,len(errors)):].max())],
        follow_errors=follow,independent_next_boundary_bitwise_equal=True if alpha else None)
    return arrays,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/h_update')))
    p.add_argument('--seed',type=int);p.add_argument('--boards',type=int,default=32);p.add_argument('--board-ids',type=int,nargs='+')
    p.add_argument('--case',type=int,help='Selected dataset row ID; must be in the cohort. Default: first sampled row')
    p.add_argument('--phase',type=int,default=4);p.add_argument('--l-steps',type=int,default=4096)
    p.add_argument('--tail-steps',type=int,default=128);p.add_argument('--follow-h',type=int,default=32)
    p.add_argument('--device',default='cuda');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    if min(a.boards,a.phase,a.tail_steps)<1 or a.tail_steps>a.l_steps or a.follow_h<0:p.error('Invalid schedule/counts')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    output=a.output_dir;old=json.loads((output/'run_config.json').read_text()) if (output/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else (old['seed'] if old else int(time.time()))
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config)
    if arch.pop('name','') not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False):p.error('Standard attention HRM required')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6) or arch.get('rope_steps') is not None:p.error('Stationary Sudoku HRM with native 2H6L required')
    records=read_test_records(a.dataset if a.dataset else puzzle_directory('sudoku'))
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=a.board_ids)
    case=a.case if a.case is not None else ids[0]
    if case not in ids:p.error('--case must be a member of --board-ids or the sampled cohort')
    plan=dict(seed=seed,board_ids=list(ids),case=int(case),selection='explicit case' if a.case is not None else 'first sampled case, no outcome selection',
              checkpoint=str(cp),checkpoint_sha256=sha(cp),config=config,dataset_sha256=dataset_sha256(records),
              phase=a.phase,l_steps=a.l_steps,tail_steps=a.tail_steps,follow_h=a.follow_h,device=a.device,code_sha256=sha(Path(__file__)))
    if old is not None and old!=plan:raise ValueError('Changed settings/input; use a new output directory')
    if old is None:
        if output.exists() and any(output.iterdir()):raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True);(output/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (output/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    x,y,_=next(iter(build_loader(records,ids,len(ids))));x=x.to(a.device);y=y.to(a.device)
    state=boundary(model,x,a.phase,a.l_steps,a.tail_steps)
    torch.save({k:v.cpu() for k,v in state.items()},output/'boundary.pt')
    np.savez_compressed(output/'cohort.npz',ids=ids,inputs=x.cpu().numpy(),targets=y.cpu().numpy())
    results={}
    for alpha in (0,1):
        arrays,summary=branch(model,x,y,state,ids.index(case),alpha,a.l_steps,a.follow_h)
        np.savez_compressed(output/f'branch_{alpha}.npz',**arrays);results[str(alpha)]=summary
    report=dict(plan,status='complete',native_forward_parity='bitwise H/L',same_L_start=True,branches=results,
        artifact_sha256={f.name:sha(f) for f in output.iterdir() if f.suffix in ('.pt','.npz')},
        limitation='Resampled or explicitly selected case; not guaranteed to show the historical case6566 behavior')
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
