"""Reload E30 decoders and independently reconstruct their held-out metrics."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hrm.paths import repository_root
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .fit import load_features
from hrm.analyse.sudoku.core.old_h_recovery import ridge, shuffle_within_position

ATOL, RTOL = 2e-5, 5e-5


def close(actual, expected, label):
    if isinstance(actual,torch.Tensor):actual=actual.cpu().numpy()
    if isinstance(expected,torch.Tensor):expected=expected.cpu().numpy()
    if actual is None or expected is None:
        if actual is not None or expected is not None:raise ValueError(f'Undefined metric mismatch: {label}')
        return 0.
    aa,bb=np.asarray(actual),np.asarray(expected)
    if aa.shape!=bb.shape or not np.allclose(aa,bb,atol=ATOL,rtol=RTOL,equal_nan=False):raise ValueError(f'Numerical mismatch: {label}')
    return float(np.max(np.abs(aa-bb))) if aa.size else 0.


def mean_by_position(values, positions):
    result=torch.zeros(81,values.shape[-1],device=values.device)
    for cell in range(81):
        selected=values[positions==cell]
        if len(selected):result[cell]=selected.mean(0)
    return result


def statistics(pred, target, mean, blank, puzzles, tokens, seed, repeats):
    pred,target,mean=[x.cpu().numpy().astype(np.float64) for x in (pred,target,mean)]
    error=((pred-target)**2).mean(-1);baseline=((target-mean)**2).mean(-1);energy=(target**2).mean(-1)
    def cosine(a,b):return (a*b).sum(-1)/(np.maximum(np.linalg.norm(a,axis=-1),1e-8)*np.maximum(np.linalg.norm(b,axis=-1),1e-8))
    absolute_error=np.abs(pred-target).mean(-1);target_abs=np.abs(target).mean(-1)
    cos=cosine(pred,target);centered=cosine(pred-mean,target-mean)
    ratio=lambda e,d:float(e/d) if d>0 else None
    r2=lambda e,d:1-float(e/d) if d>0 else None
    values={k:v.reshape(puzzles,tokens).mean(-1) for k,v in dict(mse=error,baseline_mse=baseline,target_energy=energy,mae=absolute_error,target_abs=target_abs,cosine=cos,centered_cosine=centered).items()}
    result=dict(mse=float(error.mean()),baseline_mse=float(baseline.mean()),r2=r2(error.sum(),baseline.sum()),relative_energy_error=ratio(error.sum(),energy.sum()),cosine=float(cos.mean()),centered_cosine=float(centered.mean()))
    result.update(mae=float(absolute_error.mean()),target_abs=float(target_abs.mean()),relative_l1=ratio(absolute_error.sum(),target_abs.sum()),relative_l2=float(np.sqrt(error.sum()/energy.sum())) if energy.sum()>0 else None)
    for label,mask in [('blank',blank.cpu().numpy()),('given',~blank.cpu().numpy())]:
        result[label+'_r2']=r2(error[mask].sum(),baseline[mask].sum());result[label+'_cells']=int(mask.sum())
    ix=np.random.default_rng(seed).integers(puzzles,size=(repeats,puzzles))
    denominator=values['baseline_mse'][ix].mean(1);valid=denominator>0
    estimates=1-values['mse'][ix].mean(1)[valid]/denominator[valid]
    result['conditional_puzzle_r2_ci95']=np.quantile(estimates,[.025,.975]).tolist() if len(estimates) else None
    result['defined_bootstrap_replicates']=int(valid.sum())
    return result,values


@torch.no_grad()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/memory/records/old_h'))
    p.add_argument('--records',type=Path,default=root/'decoders');p.add_argument('--feature-records',type=Path,default=root/'features')
    p.add_argument('--output-dir',type=Path,default=root/'audit');p.add_argument('--device',help='Default: fitting device; same backend required for seeded shuffle reconstruction');a=p.parse_args(argv)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    source,features=load_features(a.feature_records)
    plan=json.loads((a.records/'run_config.json').read_text());manifest=json.loads((a.records/'manifest.json').read_text())
    if manifest.get('status')!='complete' or any(manifest.get(k)!=v for k,v in plan.items()):raise ValueError('Incomplete or inconsistent decoder records')
    if plan['feature_manifest_sha256']!=sha(a.feature_records/'manifest.json'):raise ValueError('Decoder/feature source mismatch')
    if any(plan[k]!=source[k] for k in ('counts','tokens','steps')):raise ValueError('Decoder split/settings mismatch')
    device=a.device or plan['device']
    if torch.device(device).type!=torch.device(plan['device']).type:raise ValueError('Use the fitting device backend for seeded shuffle reconstruction')
    features={k:v.to(device) for k,v in features.items()}
    names=('next_l','old_l','x','old_l_x')
    required={'results.json','per_puzzle.npz'}|{f'probe_{step}_{name}.pt' for step in plan['steps'] for name in (*names,'next_l_shuffled')}
    if set(manifest['artifacts'])!=required:raise ValueError('Missing or unexpected decoder artifacts')
    for name in required:
        if sha(a.records/name)!=manifest['artifacts'][name]:raise ValueError(f'Decoder artifact changed: {name}')
    rows=json.loads((a.records/'results.json').read_text());lookup={(r['step'],r['probe']):r for r in rows}
    labels=['position_mean',*['raw_'+k for k in ('next_h','next_l','bare_next_h','bare_old_h')],*[name+suffix for name in names for suffix in ('_linear','_mlp')],'next_l_shuffled_linear']
    if len(lookup)!=len(rows) or set(lookup)!={(step,label) for step in plan['steps'] for label in labels} or len(rows)!=manifest['rows']:raise ValueError('Missing or duplicate reported results')
    with np.load(a.records/'per_puzzle.npz',allow_pickle=False) as f:per_puzzle={k:f[k] for k in f.files}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    ntrain,nval,ntest=plan['counts'];tokens=plan['tokens'];ntrain*=tokens;nval*=tokens
    test=slice(ntrain+nval,None);val=slice(ntrain,ntrain+nval);pos=features['positions'].flatten()
    blank=features['is_blank'].flatten()[test];embedding=features['x'].flatten(0,1).float()
    checked_keys=set();max_error=0.;checks=[]
    def check_result(step,label,pred,target,mu):
        nonlocal max_error
        result,arrays=statistics(pred,target,mu,blank,ntest,tokens,plan['seed'],plan['bootstrap_repeats'])
        for key,value in result.items():max_error=max(max_error,close(value,lookup[step,label][key],f'{step}/{label}/{key}'))
        for key,value in arrays.items():
            path=f'{step}/{label}/{key}';checked_keys.add(path)
            max_error=max(max_error,close(value,per_puzzle[path],path))
        checks.append(dict(step=step,probe=label,recomputed_r2=result['r2']))
    for si,step in enumerate(plan['steps']):
        target=features[f'{step}/h'].flatten(0,1).float();ymean=mean_by_position(target[:ntrain],pos[:ntrain]);mu=ymean[pos];y=target-mu
        check_result(step,'position_mean',mu[test],target[test],mu[test])
        for key in ('next_h','next_l','bare_next_h','bare_old_h'):
            check_result(step,'raw_'+key,features[f'{step}/{key}'].flatten(0,1).float()[test],target[test],mu[test])
        if not source['evaluation_hh_link']:
            if not torch.equal(features[f'{step}/h'],features[f'{step}/bare_old_h']) or not torch.equal(features[f'{step}/next_h'],features[f'{step}/bare_next_h']):raise ValueError('No-HH feature identity failed')
        for j,name in enumerate(names):
            raw=embedding if name=='x' else features[f'{step}/'+('l' if name in ('old_l','old_l_x') else 'next_l')].flatten(0,1).float()
            if name=='old_l_x':raw=torch.cat((raw,embedding),-1)
            state=torch.load(a.records/f'probe_{step}_{name}.pt',map_location=device,weights_only=True)
            xm=mean_by_position(raw[:ntrain],pos[:ntrain]);scale=(raw[:ntrain]-xm[pos[:ntrain]]).square().mean(0).sqrt().clamp_min(1e-3)
            for label,actual,expected in [('target means',state['ymean'],ymean),('input means',state['xmean'],xm),('input scales',state['xscale'],scale)]:close(actual,expected,label)
            # Use saved transforms to reconstruct predictions, independently checked above.
            xx=(raw-state['xmean'][pos])/state['xscale'];saved_mu=state['ymean'][pos];centered=target-saved_mu
            w,selection=ridge(xx,centered,ntrain,nval)
            close(state['ridge'],w,'selected ridge')
            close(state['linear_selection']['validation_mse'],selection['validation_mse'],'ridge validation')
            if state['linear_selection']['alpha']!=selection['alpha']:raise ValueError('Ridge selection differs')
            check_result(step,name+'_linear',xx[test]@state['ridge']+saved_mu[test],target[test],saved_mu[test])
            expected_seed=plan['seed']+100+si*10+j
            if state['mlp_seed']!=expected_seed or state['sampling_seed']!=expected_seed+1:raise ValueError('Decoder seed differs')
            net=nn.Sequential(nn.Linear(xx.shape[-1],256),nn.GELU(),nn.Linear(256,target.shape[-1])).to(device);net.load_state_dict(state['mlp']);net.eval()
            history=state['mlp_selection']['history']
            if [r['epoch'] for r in history]!=list(range(plan['epochs']+1)):raise ValueError('Missing validation epochs')
            best=min(history,key=lambda r:r['validation_mse'])
            if best['epoch']!=state['mlp_selection']['epoch']:raise ValueError('MLP selection differs')
            validation=float((xx[val]@state['ridge']+net(xx[val])-centered[val]).square().mean())
            close(validation,best['validation_mse'],'MLP validation');close(validation,state['mlp_selection']['validation_mse'],'selected MLP validation')
            check_result(step,name+'_mlp',xx[test]@state['ridge']+net(xx[test])+saved_mu[test],target[test],saved_mu[test])
            if name=='next_l':
                sh=torch.load(a.records/f'probe_{step}_next_l_shuffled.pt',map_location=device,weights_only=True)
                for key in ('xmean','xscale','ymean'):close(sh[key],state[key],'shuffled transforms')
                if sh['shuffle_train_seed']!=plan['seed']+10000 or sh['shuffle_validation_seed']!=plan['seed']+10001:raise ValueError('Shuffle seeds differ')
                yy=centered.clone();yy[:ntrain]=shuffle_within_position(yy[:ntrain],pos[:ntrain],sh['shuffle_train_seed']);yy[val]=shuffle_within_position(yy[val],pos[val],sh['shuffle_validation_seed'])
                sw,info=ridge(xx,yy,ntrain,nval);close(sh['ridge'],sw,'shuffled ridge');close(sh['linear_selection']['validation_mse'],info['validation_mse'],'shuffled validation')
                check_result(step,'next_l_shuffled_linear',xx[test]@sh['ridge']+saved_mu[test],target[test],saved_mu[test])
    if checked_keys!=set(per_puzzle):raise ValueError('Unexpected per-puzzle metrics')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(status='complete',checks=checks,maximum_metric_absolute_error=max_error,absolute_tolerance=ATOL,relative_tolerance=RTOL,
        feature_manifest_sha256=sha(a.feature_records/'manifest.json'),decoder_manifest_sha256=sha(a.records/'manifest.json'),
        code_sha256=sha(Path(__file__)),device=device,no_hh_identity_checked=not source['evaluation_hh_link'],
        limitation='Saved-device reload and numerical reconstruction, not model retraining. Validation history minimum and selected weights are checked; unselected MLP epoch weights are not retained.')
    (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(f'Checked {len(checks)} reported rows and {len(checked_keys)} per-puzzle arrays');return 0


if __name__=='__main__':raise SystemExit(main())
