"""lowest-index never-solved cases, matched native/fixed-H branches."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root
from hrm.analyse.dynamics.jacobian.models import HRM,unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def read_screen(path,count):
    path=Path(path);meta=json.loads(path.read_text())
    raw_path=path.with_suffix('.npz');carry_path=path.with_suffix('.carry.pt')
    with np.load(raw_path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    saved=torch.load(carry_path,map_location='cpu',weights_only=True)
    ids=raw['indices'];n=len(ids);cycles=meta['max_outer_cycles']
    if len(set(ids.tolist()))!=n or raw['inputs'].shape!=(n,82) or raw['targets'].shape!=(n,82):raise ValueError('Invalid screen cohort')
    if raw['exact_board81'].shape!=(n,cycles) or raw['valid'].shape!=(n,cycles):raise ValueError('Incomplete screen horizon')
    if meta['examples']!=n or saved['completed_outer_cycles']!=cycles or saved['checkpoint_sha256']!=meta['checkpoint_sha256']:raise ValueError('Screen/carry metadata mismatch')
    if not np.array_equal(ids,saved['indices'].numpy()) or not np.array_equal(raw['inputs'],saved['input_ids'].numpy()):raise ValueError('Screen/carry inputs or ordering mismatch')
    if meta['data'].get('shard_count',1)!=1:raise ValueError('Use one complete unsharded screen for global lowest-index selection')
    if set(saved['carry'])!={'z_H','z_L'}:raise ValueError('Standard HRM carry required')
    for state in saved['carry'].values():
        if state.ndim!=3 or tuple(state.shape[:2])!=(n,82) or not torch.isfinite(state).all():raise ValueError('Invalid final carry')
    terminal_exact=(raw['final_prediction'][:,1:]==raw['targets'][:,1:]).all(1)
    if not np.array_equal(terminal_exact,raw['exact_board81'][:,-1]):raise ValueError('Terminal predictions disagree with correctness trace; regenerate the screen')
    if raw['exact_board81'].dtype!=np.bool_ or raw['valid'].dtype!=np.bool_:raise ValueError('Boolean correctness traces required')
    protocol=meta.get('protocol',{})
    if not protocol.get('carry_continuous') or protocol.get('early_stopping') is not False or protocol.get('restarts')!=0:raise ValueError('Continuous, no-restart screen required')
    never=~raw['exact_board81'].any(1)
    eligible=sorted(np.flatnonzero(never),key=lambda i:int(ids[i]))
    chosen=eligible[:count]
    return meta,raw,saved,chosen,len(eligible),{p.name:sha(p) for p in (path,raw_path,carry_path)}


def recurrence(states,reference=256,gap=32):
    z=np.asarray(states,dtype=np.float64).reshape(len(states),-1)
    if not 1<reference<reference+gap<len(z) or gap<0 or not np.isfinite(z).all():raise ValueError('Invalid recurrence window/states')
    # Same full-dimensional RMS geometry as the archived analysis. Translating
    # by the first state reduces cancellation without changing distances.
    z=z-z[0];ref=z[:reference];future=z[reference+gap:]
    radius=float(np.sqrt(np.mean((ref-ref.mean(0))**2)))
    q=(ref*ref).mean(1);nearest=[]
    for block in np.array_split(future,max(1,(len(future)+31)//32)):
        d=np.maximum((block*block).mean(1)[:,None]+q[None]-2*block@ref.T/z.shape[1],0)
        nearest.extend(np.sqrt(d.min(1)).tolist())
    q95=float(np.quantile(nearest,.95))
    return dict(reference_radius_RMS=radius,heldout_nearest_q95_RMS=q95,
        nearest_q95_over_radius=q95/radius if radius else None,zero_reference_radius=radius==0,
        reference_size=reference,gap=gap,validation_size=len(future)),np.asarray(nearest)


@torch.inference_mode()
def continue_case(model,x,y,carry,steps=4096,tail=513):
    if steps<2 or steps%2 or not 1<tail<=steps+1:raise ValueError('Even continuation steps and valid tail required')
    emb=model.embed(x);h=carry['z_H'].clone();l=carry['z_L'].clone();fixed_h=h.clone();fixed_l=l.clone()
    start=steps-tail+1;hs=[];ls=[];fs=[];errors=[];updates=[None];parity=0
    for t in range(steps+1):
        pred=model.lm_head(h).argmax(-1)[:,1:]
        errors.append(int((pred!=y[:,1:]).sum()))
        if t>=start:
            hs.append(h.float().cpu().numpy()[0]);ls.append(l.float().cpu().numpy()[0]);fs.append(fixed_l.float().cpu().numpy()[0])
        if t==steps:break
        if t%2==0:native,_=model(dict(z_H=h.clone(),z_L=l.clone()),x)
        old=h.clone()
        for _ in range(6):l=model.L_level(l+h+emb)
        h=model.H_level(h+l)
        fixed_l=model.L_level(fixed_l+fixed_h+emb)
        updates.append(float((h.float()-old.float()).square().mean().sqrt()))
        if t%2==1:
            if not torch.equal(h,native['z_H']) or not torch.equal(l,native['z_L']):raise RuntimeError('Native continuation parity failed')
            parity+=1
    hit=next((i for i,e in enumerate(errors) if i>0 and e==0),None)
    under=np.asarray([False]+[v<.01 for v in updates[1:]])
    starts=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(under,128).all(1)) if len(under)>=128 else []
    arrays=dict(steps=np.arange(start,steps+1),H=np.asarray(hs),L=np.asarray(ls),fixed_H_L=np.asarray(fs),
        errors=np.asarray(errors),H_update_RMS=np.asarray([np.nan if v is None else v for v in updates]),inputs=x.cpu().numpy(),targets=y.cpu().numpy())
    return arrays,dict(first_correct_additional_H=hit,final_errors=errors[-1],native_solved=hit is not None,
        first_128_updates_below_001_start=int(starts[0]) if len(starts) else None,native_cycle_bitwise_checks=parity)


def main(argv=None):
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/recurrence'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=root/'screen/continuous.json')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output-dir',type=Path,default=root/'continuation')
    p.add_argument('--cases',type=int,default=12);p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--tail',type=int,default=513);p.add_argument('--reference',type=int,default=256);p.add_argument('--gap',type=int,default=32)
    p.add_argument('--device',default='cuda');a=p.parse_args(argv)
    if a.cases<1 or a.steps<2 or a.steps%2 or not 1<a.reference<a.reference+a.gap<a.tail<=a.steps+1:p.error('Invalid counts or matched window')
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    meta,raw,saved,chosen,eligible,sources=read_screen(a.records,a.cases)
    cp=hub.checkpoint('sudoku',a.checkpoint)
    if sha(cp)!=meta['checkpoint_sha256']:raise ValueError('Checkpoint differs from screen')
    arch=dict(meta['arch']);name=arch.pop('name','')
    if name not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False) or arch.get('rope_steps') is not None or (arch.get('H_cycles'),arch.get('L_cycles'))!=(2,6):raise ValueError('Standard stationary 2H6L HRM required')
    seed=meta['data']['seed'];os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    plan=dict(screen=str(a.records.resolve()),source_sha256=sources,checkpoint_sha256=sha(cp),seed=seed,
        selected_ids=[int(raw['indices'][i]) for i in chosen],eligible_cases=eligible,requested_cases=a.cases,
        selection='lowest dataset row IDs never exact-board-correct at any saved outer-cycle boundary',
        starting_H=2*meta['max_outer_cycles'],steps=a.steps,tail=a.tail,reference=a.reference,gap=a.gap,
        device=a.device,deterministic_algorithms=True,code_sha256=sha(Path(__file__)))
    a.output_dir.mkdir(parents=True);(a.output_dir/'run_config.json').write_text(json.dumps(plan,indent=2))
    reports=[];artifacts={}
    for i in chosen:
        row=int(raw['indices'][i]);x=torch.tensor(raw['inputs'][i:i+1],device=a.device);y=torch.tensor(raw['targets'][i:i+1],device=a.device)
        carry={k:v[i:i+1].to(a.device) for k,v in saved['carry'].items()}
        with torch.inference_mode():pred=model.lm_head(carry['z_H']).argmax(-1).cpu().numpy()[0]
        if not np.array_equal(pred,raw['final_prediction'][i]):raise ValueError('Saved carry does not reproduce final screen prediction')
        arrays,report=continue_case(model,x,y,carry,a.steps,a.tail)
        if arrays['errors'][0]==0:raise ValueError('Selected never-solved case is already correct at branch start')
        metrics={}
        for name,states in [('native_H',arrays['H']),('native_joint',np.concatenate([arrays['H'],arrays['L']],axis=-1)),('fixed_H_L',arrays['fixed_H_L'])]:
            metrics[name],arrays[name+'_nearest']=recurrence(states,a.reference,a.gap)
        filename=f'case_{row}.npz';np.savez_compressed(a.output_dir/filename,**arrays);artifacts[filename]=sha(a.output_dir/filename)
        reports.append(dict(case=row,**report,metrics=metrics));print(f'case {row} complete',flush=True)
    result=dict(plan,status='complete',cases=reports,artifact_sha256=artifacts,
        recurrence_cohort=[r['case'] for r in reports if not r['native_solved']],
        reference_schedule=meta['examples']==4096 and meta['max_outer_cycles']==16384 and a.cases==12 and (a.steps,a.tail,a.reference,a.gap)==(4096,513,256,32),
        limitation='Resampled cohorts may have fewer than twelve eligible cases or no newly solved case. No attractor/convergence claim follows from finite-window recurrence. Native H and fixed-H L use different computational budgets per update.')
    (a.output_dir/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
