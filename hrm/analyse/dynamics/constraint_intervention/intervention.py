"""Fit a count readout and apply matched fractional-score L interventions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.training.data import puzzle_directory, freeze_training_seed
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader
from hrm.analyse.dynamics.constraint_readout.run import fit_linear, finite
from .population import count
from .count_target_dose import target_rotation, score_fixed_control, short_run, arm_list
from hrm.analyse.dynamics.answer_error import intervene_internal_critic as geometry


@torch.inference_mode()
def fit(model,x,cycles):
    feature_batches=[];target_batches=[]
    for x in x.split(16):
        emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
        features=[];targets=[]
        for _ in range(cycles):
            for _ in range(model.H_cycles):
                for _ in range(model.L_cycles):
                    l=model.L_level(l+h+emb)
                    features.append(l.mean(1).cpu())
                    targets.append(count(model.lm_head(model.H_level(h+l))).cpu())
                h=model.H_level(h+l)
        feature_batches.append(torch.stack(features,1).double());target_batches.append(torch.stack(targets,1))
    features=torch.cat(feature_batches);targets=torch.cat(target_batches)
    w,b=fit_linear(features.flatten(0,1),targets.flatten())
    centered=features.flatten(0,1)-features.flatten(0,1).mean(0)
    covariance=centered.T@centered/len(centered)
    return w,b,covariance


@torch.inference_mode()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path,default=repository_root()/'exp/error_readout/records/sudoku_constraints')
    p.add_argument('--train-boards',type=int,default=256);p.add_argument('--test-boards',type=int,default=128)
    p.add_argument('--fit-cycles',type=int,default=8);p.add_argument('--device',default='cpu')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if min(a.train_boards,a.test_boards,a.fit_cycles)<1:p.error('Positive budgets required')
    a.output.mkdir(parents=True,exist_ok=True);seed=freeze_training_seed(a.output)
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config);name=arch.pop('name','hrm@HRM');arch['forward_dtype']='float32'
    if name not in ('hrm@HRM','hrm.models.hrm@HRM') or (arch['H_cycles'],arch['L_cycles'])!=(2,6):p.error('A 2H6L attention HRM is required')
    records=read_test_records(a.dataset or puzzle_directory())
    ids=select_board_ids(len(records),boards=a.train_boards+a.test_boards,offset=0,selection='random',selection_seed=seed,explicit=None)
    plan=dict(seed=seed,checkpoint=str(cp),config=config,train_ids=ids[:a.train_boards],test_ids=ids[a.train_boards:],fit_cycles=a.fit_cycles,
              fork_H_updates=4,threshold=1.0,module='L',arms=arm_list(('L',)),target='raw argmax duplicate excess; nonnegative-clipped readout',
              geometry='training covariance preconditioned tangent; matched token norms, chord and probe-preserving controls')
    path=a.output/'config.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Settings changed; use a new output')
    path.write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return
    with (a.output/'started.json').open('x') as f:json.dump({'seed':seed},f)
    torch.manual_seed(seed);torch.set_num_threads(2);geometry.SEED=seed
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    # Accumulate the training population on CPU; infer one puzzle at a time.
    trainx,_,_=next(iter(build_loader(records,ids[:a.train_boards],a.train_boards)))
    # Pooling sufficient statistics across puzzle batches preserves the OLS objective.
    # The training batch is explicit to retain one covariance over all observations.
    w,b,cov=fit(model,trainx.to(a.device),a.fit_cycles)
    torch.save(dict(weight=w,bias=b,covariance=cov),a.output/'probe.pt')
    fullw=torch.cat([w,torch.zeros_like(w)]);rows=[];excluded=[]
    for x,y,rid in build_loader(records,ids[a.train_boards:],1):
        x=x.to(a.device);y=y.to(a.device);emb=model.embed(x)
        h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
        for _ in range(4):
            for _ in range(6):l=model.L_level(l+h+emb)
            h=model.H_level(h+l)
        # Geometry uses FP64 on the state device; the recurrence remains FP32.
        ww=w.to(a.device);bb=b.to(a.device)
        z=l
        baseline=(z.mean(1).double()@ww+bb).clamp_min(0)
        if baseline.item()<1:excluded.append(int(rid[0]));continue
        vectors=geometry.directions(z,ww,cov.to(a.device))
        for arm in plan['arms']:
            altered=z
            if arm['module']!='native':
                target,theta=target_rotation(z,vectors['critic'],ww,baseline*arm['target_fraction'])
                altered=target if arm['kind']=='critic' else score_fixed_control(z,vectors[f"random{arm['seed']}"],ww,theta,arm['sign'])
                chord=(altered.double()-z.double()).flatten(1).norm(dim=-1)/z.double().flatten(1).norm(dim=-1)
                reference=(target.double()-z.double()).flatten(1).norm(dim=-1)/z.double().flatten(1).norm(dim=-1)
                assert (chord-reference).abs().max()<2e-7
                assert ((altered.double().norm(dim=-1)-z.double().norm(dim=-1)).abs()/z.double().norm(dim=-1)).max()<2e-6
                expected=baseline*(1+arm['target_fraction']) if arm['kind']=='critic' else baseline
                assert ((altered.mean(1).double()@ww+bb).clamp_min(0)-expected).abs().max()<2e-4
            result=short_run(model,h,altered,x,y,fullw.to(a.device),bb)
            rows.append(dict(row_id=int(rid[0]),arm=arm,step_norm=result['step_norm'][0].tolist(),exact=bool(result['exact'][0]),baseline=float(baseline)))
    output=dict(config=plan,eligible=a.test_boards-len(excluded),excluded_ids=excluded,observations=rows)
    (a.output/'results.json').write_text(json.dumps(finite(output),indent=2,allow_nan=False))
    print(json.dumps({'eligible':output['eligible'],'excluded':len(excluded),'observations':len(rows)}))

if __name__=='__main__':main()
