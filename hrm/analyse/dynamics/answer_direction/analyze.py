"""Answer-direction audit of recorded paired H/L single-step branches; no model fitting."""
import argparse, hashlib, json, os, time
from pathlib import Path
from hrm.paths import repository_root
from .probe_hl_direction_intervention import load_records, BRANCHES
import numpy as np
import pandas as pd

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def prob(z):
 z=z.astype(np.float64);z-=z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)

def paired_summary(vals, seed, repeats):
 if not len(vals):
  return dict(puzzles=0, **{key:None for key in ('H_mean','L_mean','H_minus_L','ci_low','ci_high','H_ci_low','H_ci_high','L_ci_low','L_ci_high')})
 rng=np.random.default_rng(seed);boot=vals[rng.integers(len(vals),size=(repeats,len(vals)))].mean(1)
 ci=np.quantile(boot[:,0]-boot[:,1],[.025,.975])
 return dict(puzzles=len(vals),H_mean=vals[:,0].mean(),L_mean=vals[:,1].mean(),H_minus_L=(vals[:,0]-vals[:,1]).mean(),ci_low=ci[0],ci_high=ci[1],H_ci_low=np.quantile(boot[:,0],.025),H_ci_high=np.quantile(boot[:,0],.975),L_ci_low=np.quantile(boot[:,1],.025),L_ci_high=np.quantile(boot[:,1],.975))


def paired_records(records, readout_records):
 meta=load_records(readout_records)
 branch=json.loads((records/'manifest.json').read_text())
 if branch.get('status')!='complete' or branch['readout_manifest_sha256']!=sha(readout_records/'manifest.json'):
  raise ValueError('Incomplete or mismatched intervention records')
 if branch['selected']!=meta['selected'] or branch['horizon']!=meta['horizon'] or branch['checkpoint_sha256']!=meta['checkpoint_sha256']:
  raise ValueError('Intervention protocol differs from readout')
 for name,digest in branch['artifact_sha256'].items():
  if sha(records/name)!=digest:raise ValueError(f'Intervention artifact changed: {name}')
 files=[records/'branch_logits.npz',readout_records/'heldout_trajectories.npz']
 with np.load(files[0],allow_pickle=False) as f:z={k:f[k] for k in f.files}
 with np.load(files[1],allow_pickle=False) as f:t={k:f[k] for k in f.files}
 expected={(int(rid),cycle) for i,rid in enumerate(t['ids']) for cycle in range(2,meta['horizon']) if (t['native_pred'][i,(cycle-1)*7]!=t['target'][i,1:]).any()}
 actual=list(zip(z['ids'].tolist(),z['cycles'].tolist()))
 if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('Duplicate, missing or ineligible paired states')
 if len(actual)!=branch['states']:raise ValueError('State count differs from manifest')
 for key in BRANCHES:
  if z[key].shape!=(len(actual),81,10) or not np.isfinite(z[key]).all():raise ValueError(f'Invalid branch logits: {key}')
 if t['ids'].tolist()!=meta['splits']['test_rows']:raise ValueError('Heldout IDs differ from frozen split')
 lookup={int(x):i for i,x in enumerate(t['ids'])};ii=np.array([lookup[int(i)] for i in z['ids']],dtype=int)
 y=t['target'][ii,1:];blank=t['input'][ii,1:]==0
 if len(set(t['ids']))!=len(t['ids']) or not ((y>=1)&(y<=9)).all():raise ValueError('Invalid heldout IDs/labels')
 if (blank.sum(1)==0).any():raise ValueError('Eligible puzzle has no blank cells')
 eligible=(t['native_pred'][ii,(z['cycles']-1)*7]!=y).any(1)
 for key,offset in [('before_L6',-2),('start',-1),('H_only',0),('H_then_L',1)]:
  if not np.array_equal(z[key].argmax(-1),t[meta['selected']+'_pred'][ii,z['cycles']*7+offset]):
   raise ValueError(f'Selected readout branch mismatch: {key}')
 return meta,z,t,ii,y,blank,eligible,files


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/answer_direction'))
 p.add_argument('--records',type=Path,default=root/'intervention')
 p.add_argument('--readout-records',type=Path,default=root/'readout')
 p.add_argument('--output-dir',type=Path,default=root/'statistics')
 p.add_argument('--seed',type=int);p.add_argument('--bootstrap-repeats',type=int,default=5000)
 a=p.parse_args(argv)
 if a.bootstrap_repeats<1:p.error('Positive bootstrap repeat count required')
 a.output=a.output_dir
 if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
 meta,z,t,ii,y,blank,eligible,files=paired_records(a.records,a.readout_records)
 seed=a.seed if a.seed is not None else int(time.time())
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'run_config.json').open('x') as f:json.dump(dict(seed=seed,bootstrap_repeats=a.bootstrap_repeats,readout_manifest_sha256=sha(a.readout_records/'manifest.json'),intervention_manifest_sha256=sha(a.records/'manifest.json')),f,indent=2)
 p0=prob(z['start']);gold=np.eye(10)[y];direction=(gold-p0)*blank[...,None];dn=np.linalg.norm(direction.reshape(len(ii),810),axis=1)
 rows=pd.DataFrame({'board_id':z['ids'],'cycle':z['cycles']})
 n=blank.sum(1);base_pred=z['start'].argmax(-1)
 for arm,key in [('H','H_only'),('L','L_only')]:
  pp=prob(z[key]);delta=(pp-p0)*blank[...,None];norm=np.linalg.norm(delta.reshape(len(ii),810),axis=1);den=norm*dn
  rows[arm+'_cosine']=np.divide((delta*direction).sum((1,2)),den,out=np.full(len(ii),np.nan),where=den>1e-12)
  rows[arm+'_brier_reduction']=(((p0-gold)**2-(pp-gold)**2)*blank[...,None]).sum((1,2))/n
  rows[arm+'_ce_reduction']=((np.log(np.take_along_axis(pp,y[...,None],-1)[...,0].clip(1e-300))-np.log(np.take_along_axis(p0,y[...,None],-1)[...,0].clip(1e-300)))*blank).sum(1)/n
  rows[arm+'_blank_accuracy_gain_pp']=100*((pp.argmax(-1)==y).astype(float)-(base_pred==y).astype(float)).__mul__(blank).sum(1)/n
  rows[arm+'_errors_reduced']=(base_pred!=y).sum(1)-(pp.argmax(-1)!=y).sum(1)
  rows[arm+'_probability_step_norm']=norm
 # Require a defined direction for BOTH arms in directional comparisons.
 joint=rows[['H_cosine','L_cosine']].notna().all(axis=1)
 rows.loc[~joint,['H_cosine','L_cosine']]=np.nan
 rows.to_csv(a.output/'per_state.csv',index=False)
 metrics=['cosine','brier_reduction','ce_reduction','blank_accuracy_gain_pp','errors_reduced','probability_step_norm'];summary=[]
 for window,mask in [(f'all_cycles_2_{meta["horizon"]-1}',np.ones(len(rows),bool)),(f'early_cycles_2_{min(16,meta["horizon"]-1)}',rows.cycle<=16)]:
  g=rows[mask].groupby('board_id').mean(numeric_only=True)
  g.to_csv(a.output/(window+'_per_board.csv'))
  for metric in metrics:
   vals=g[['H_'+metric,'L_'+metric]].dropna().to_numpy()
   summary.append(dict(window=window,metric=metric,**paired_summary(vals,seed,a.bootstrap_repeats)))
 pd.DataFrame(summary).to_csv(a.output/'summary.csv',index=False)
 (a.output/'audit.json').write_text(json.dumps(dict(status='complete',seed=seed,bootstrap_repeats=a.bootstrap_repeats,states=len(rows),puzzles=rows.board_id.nunique(),undefined_cosine_pairs=int((~joint).sum()),all_states_unsolved_at_cycle_entry=bool(eligible.all()),sources=[dict(path=str(f),sha256=sha(f)) for f in files],script_sha256=sha(__file__),analysis='Exploratory reanalysis. Per-puzzle means then paired puzzle bootstrap, saved-seed resampling; not independent training repeats.'),indent=2))
 print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
