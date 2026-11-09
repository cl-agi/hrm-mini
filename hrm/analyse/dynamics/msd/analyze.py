"""Fit MSD exponents from explicitly supplied rollout records; no model fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from hrm.paths import repository_root

SERIES = {'hrm': ('h','l'), 'trm': ('h','l'), 'hrm_h2l1': ('h','l'), 'rt': ('rt',)}
EVAL_L = {'hrm':6,'trm':6,'hrm_h2l1':1,'rt':1}
PROFILES = {
    "common6_47": {
        "segments": (1, 2, 3, 4), "minimum": 6, "maximum": 47,
        "role": "Primary: compare every model, state and window on the same 21 existing lags.",
    },
    "common6_24": {
        "segments": (1, 2, 3, 4), "minimum": 6, "maximum": 24,
        "role": (
            "Sensitivity: retain existing lags up to 24, half the shortest window length; "
            "the actual largest available lag is 22 and the earliest window has at least 26 origins."
        ),
    },
    "late6_3327": {
        "segments": (4,), "minimum": 6, "maximum": 3327,
        "role": "Late-window comparison: all model/state series share the same 39 existing lags.",
    },
    "archived": {
        "segments": (1, 2, 3, 4), "minimum": None, "maximum": None,
        "role": (
            "Reproduce the original plot_s3.slopes rule: positive MSD and lag >= eval_l. "
            "HRM/TRM use lag >= 6; equal-frequency HRM and RT use lag >= 1."
        ),
    },
}
def fit_log_msd(lags: np.ndarray, msd: np.ndarray) -> tuple[float, float, float | None]:
    if len(lags) < 2 or not np.all(np.isfinite(msd)) or not np.all(msd > 0):
        raise ValueError("OLS requires at least two finite, positive MSD values")
    x, y = np.log2(lags), np.log2(msd)
    beta, intercept = np.polyfit(x, y, 1)
    residual_sum = float(np.sum((y - (intercept + beta*x))**2))
    total_sum = float(np.sum((y - y.mean())**2))
    r2 = 1 - residual_sum / total_sum if total_sum > 0 else None
    return float(beta), float(intercept), r2


def check_fitting() -> dict:
    lags = np.asarray([6., 7., 13., 22., 47.])
    msd = 2.5 * lags**0.37
    beta, intercept, r2 = fit_log_msd(lags, msd)
    assert np.isclose(beta, .37, rtol=0, atol=1e-12)
    assert np.isclose(intercept, np.log2(2.5), rtol=0, atol=1e-12)
    assert np.isclose(r2, 1., rtol=0, atol=1e-12)
    assert np.isclose(fit_log_msd(lags, msd/msd[0])[0], beta, rtol=0, atol=1e-12)
    constant_beta, _, constant_r2 = fit_log_msd(lags, np.ones_like(lags))
    assert abs(constant_beta) < 1e-12 and constant_r2 is None
    return {"known_power_law": True, "scale_invariance": True, "constant_curve": True}



def read_csv(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run_key(row):
    return row['run_id'] if row.get('protocol')=='fresh-v1' else int(row['seed'])


def load_curves(root):
    metadata=read_csv(root/'rollout_metadata.csv')
    if not metadata:raise ValueError('Empty rollout metadata')
    runs={}
    protocols={row.get('protocol','legacy') for row in metadata}
    if len(protocols)!=1 or not protocols<= {'legacy','fresh-v1'}:raise ValueError('Use a single supported metadata protocol')
    boundaries=json.loads(metadata[0]['segment_boundaries_l_updates'])
    if len(boundaries)!=5 or boundaries[0]!=0 or any(b-a<8 for a,b in zip(boundaries,boundaries[1:])):raise ValueError('Invalid MSD windows')
    for row in metadata:
        key=row['condition'],run_key(row)
        condition=key[0]
        if condition not in SERIES or key in runs:raise ValueError('Unknown condition or duplicate rollout')
        if json.loads(row['segment_boundaries_l_updates'])!=boundaries or int(row['actual_l_updates'])!=boundaries[-1]:
            raise ValueError('MSD windows differ across runs')
        if int(row['samples'])<1 or int(row['eval_l'])!=EVAL_L[condition]:raise ValueError('Invalid sample count or eval_l')
        if row.get('protocol')!='fresh-v1' and (int(row['best_epoch'])!=19 or boundaries!=[0,48,192,768,4096]):raise ValueError('Legacy records require the epoch-19 checkpoint and standard windows')
        if row.get('protocol')=='fresh-v1' and (not row.get('run_id') or row.get('checkpoint_sha256')!=row['run_id']):raise ValueError('Fresh records require checkpoint identity')
        runs[key]=row
    groups=defaultdict(list);seen=set()
    for row in read_csv(root/'msd_curves.csv'):
        condition,state=row['condition'],row['state'];seed=run_key(row);segment=int(row['segment'])
        if (condition,seed) not in runs or state not in SERIES[condition] or segment not in (1,2,3,4):raise ValueError('Curve without matching rollout/state/window')
        lag=int(row['lag_l_updates']);origins=int(row['origins']);value=float(row['mean'])
        start,end=boundaries[segment-1:segment+1];identity=condition,state,seed,segment,lag
        if identity in seen or lag<1 or origins!=end-start-lag or origins<1:raise ValueError('Duplicate curve or invalid lag/origin count')
        if row['status']!='defined' or not np.isfinite(value) or value<0:raise ValueError('Invalid MSD')
        if int(row['puzzles'])!=int(runs[condition,seed]['samples']) or int(row['eval_l'])!=EVAL_L[condition]:raise ValueError('Curve/rollout count or schedule mismatch')
        seen.add(identity);groups[condition,state,seed,segment].append(dict(condition=condition,state=state,seed=seed,segment=segment,window_start=start,window_end=end,lag=lag,origins=origins,puzzles=int(row['puzzles']),msd=value))
    grids={};records=[]
    for (condition,seed),meta in runs.items():
        declared=json.loads(meta['lag_l_updates'])
        if declared!=sorted(set(declared)) or any(lag<1 for lag in declared):raise ValueError('Invalid declared lag grid')
        for state in SERIES[condition]:
            for segment in (1,2,3,4):
                key=condition,state,seed,segment;current=sorted(groups[key],key=lambda r:r['lag'])
                expected=[lag for lag in declared if lag<boundaries[segment]-boundaries[segment-1]]
                grid=[r['lag'] for r in current]
                if not current or grid!=expected or 6 not in grid:raise ValueError('Missing curve/state/window/lag')
                if segment in grids and grids[segment]!=grid:raise ValueError('Lag grids differ across compared runs')
                grids[segment]=grid;base=next(r['msd'] for r in current if r['lag']==6)
                for row in current:row.update(msd_at_6=base,normalized_msd=row['msd']/base if base>0 else None)
                groups[key]=current;records.extend(current)
    return records,groups,runs


def analyse(groups, profile_mode='standard'):
    profiles=PROFILES if profile_mode=='standard' else {'available':dict(segments=(1,2,3,4),minimum=6,maximum=None)}
    if profile_mode=='standard' and any((r['window_start'],r['window_end'])!=[(0,48),(48,192),(192,768),(768,4096)][r['segment']-1] for rows in groups.values() for r in rows):
        raise ValueError('Standard profiles require the standard windows; use --profile available for custom windows')
    fits=[];common_grids={}
    for profile,settings in profiles.items():
        for (condition,state,seed,segment),rows in sorted(groups.items()):
            if segment not in settings['segments']:continue
            minimum=EVAL_L[condition] if profile=='archived' else settings['minimum']
            selected=[r for r in rows if r['lag']>=minimum and (settings['maximum'] is None or r['lag']<=settings['maximum'])]
            grid=tuple(r['lag'] for r in selected)
            if profile not in ('archived','available'):
                if profile in common_grids and common_grids[profile]!=grid:raise ValueError('Nonshared comparison lag grid')
                common_grids[profile]=grid
            positive=[r for r in selected if r['msd']>0]
            beta=intercept=r2=None
            if len(positive)>=2:
                beta,intercept,r2=fit_log_msd(np.array([r['lag'] for r in positive]),np.array([r['msd'] for r in positive]))
            fits.append(dict(analysis=profile,condition=condition,state=state,seed=seed,segment=segment,
                window_start=rows[0]['window_start'],window_end=rows[0]['window_end'],
                tau_min=min((r['lag'] for r in positive),default=None),tau_max=max((r['lag'] for r in positive),default=None),
                n_lags=len(positive),excluded_zero_lags=len(selected)-len(positive),
                min_origins=min((r['origins'] for r in positive),default=None),beta=beta,intercept=intercept,r2=r2))
    for profile,count in ([('common6_47',21),('common6_24',12),('late6_3327',39)] if profile_mode=='standard' else []):
        if len(common_grids[profile])!=count:raise ValueError(f'{profile} requires {count} shared lags')
    return fits


def summarize(fits):
    grouped=defaultdict(list)
    for row in fits:grouped[row['analysis'],row['condition'],row['state'],row['segment']].append(row)
    summaries=[]
    for (analysis,condition,state,segment),rows in sorted(grouped.items()):
        values=[r['beta'] for r in rows if r['beta'] is not None]
        r2=[r['r2'] for r in rows if r['r2'] is not None]
        summaries.append(dict(analysis=analysis,condition=condition,state=state,segment=segment,
            training_runs=len(rows),defined_runs=len(values),beta_mean=float(np.mean(values)) if values else None,
            beta_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,r2_mean=float(np.mean(r2)) if r2 else None))
    lookup={(r['analysis'],r['condition'],r['seed'],r['segment'],r['state']):r['beta'] for r in fits}
    deltas=[]
    for (analysis,condition,seed,segment,state),h in sorted(lookup.items()):
        if state!='h':continue
        l=lookup[analysis,condition,seed,segment,'l']
        deltas.append(dict(analysis=analysis,condition=condition,seed=seed,segment=segment,beta_h=h,beta_l=l,delta_beta=h-l if h is not None and l is not None else None))
    delta_groups=defaultdict(list)
    for row in deltas:delta_groups[row['analysis'],row['condition'],row['segment']].append(row)
    delta_summaries=[]
    for (analysis,condition,segment),rows in sorted(delta_groups.items()):
        values=[r['delta_beta'] for r in rows if r['delta_beta'] is not None]
        delta_summaries.append(dict(analysis=analysis,condition=condition,segment=segment,training_runs=len(rows),defined_runs=len(values),delta_beta_mean=float(np.mean(values)) if values else None,delta_beta_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    return summaries,deltas,delta_summaries


def summarize_curves(curves):
    grouped=defaultdict(list)
    for row in curves:
        grouped[row['condition'],row['state'],row['segment'],row['lag']].append(row)
    result=[]
    for (condition,state,segment,lag),rows in sorted(grouped.items()):
        # Each run is normalized independently before any cross-run averaging.
        values=[r['normalized_msd'] for r in rows if r['normalized_msd'] is not None]
        result.append(dict(condition=condition,state=state,segment=segment,lag=lag,
            window_start=rows[0]['window_start'],window_end=rows[0]['window_end'],
            origins_per_puzzle=rows[0]['origins'],training_runs=len(rows),defined_runs=len(values),
            normalized_msd_mean=float(np.mean(values)) if values else None,
            normalized_msd_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/msd'))
    parser.add_argument('--records',type=Path,default=root/'rollouts')
    parser.add_argument('--output-dir',type=Path,default=root/'statistics')
    parser.add_argument('--profile',choices=['standard','available'],default='standard')
    args=parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(args.output_dir)
    curves,groups,runs=load_curves(args.records);fits=analyse(groups,args.profile);summaries,deltas,delta_summaries=summarize(fits)
    checks=check_fitting();args.output_dir.mkdir(parents=True,exist_ok=True)
    outputs={'msd_curves.csv':curves,'msd_curve_summary.csv':summarize_curves(curves),'beta_per_seed.csv':fits,'beta_summary.csv':summaries}
    if deltas:
        outputs['delta_per_seed.csv']=deltas
        outputs['delta_summary.csv']=delta_summaries
    fresh=all(v.get('protocol')=='fresh-v1' for v in runs.values())
    if fresh:
        for rows in outputs.values():
            for row in rows:
                if 'seed' in row:row['checkpoint_id']=row.pop('seed')
                if 'training_runs' in row:row['checkpoints']=row.pop('training_runs')
    for name,rows in outputs.items():write_csv(args.output_dir/name,rows)
    report=dict(status='complete',runs=[dict(condition=k[0],run_id=k[1],metadata=v) for k,v in sorted(runs.items())],
        profiles=PROFILES if args.profile=='standard' else {'available':{'minimum_lag':6,'windows':'supplied metadata'}},replicate_unit='checkpoint' if fresh else 'training seed',script_sha256=sha(Path(__file__)),numerical_checks=checks,
        sources={n:sha(args.records/n) for n in ('rollout_metadata.csv','msd_curves.csv')},
        outputs={n:dict(rows=len(rows),sha256=sha(args.output_dir/n)) for n,rows in outputs.items()},
        interpretation=('Full-state MSD fitted independently by checkpoint; sample SD across distinct checkpoints. Sampling seed is separate from checkpoint identity. ' if fresh else '')+'Log2 MSD versus log2 lag OLS with free intercept, per model/state/window/training run; sample SD across available independent training runs, missing for one run. Zero MSD is retained in curves and excluded from logarithmic fits with counts. Origin counts are correlated, not independent samples. This stage does not run models or certify checkpoint/cohort provenance.')
    (args.output_dir/'analysis_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['outputs'],indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
