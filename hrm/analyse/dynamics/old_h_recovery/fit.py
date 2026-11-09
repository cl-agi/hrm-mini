"""Fit E30 position-conditioned old-H decoders from verified feature records."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hrm.paths import repository_root
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from hrm.analyse.sudoku.core.old_h_recovery import fit


def load_features(records):
    plan=json.loads((records/'run_config.json').read_text())
    manifest=json.loads((records/'manifest.json').read_text())
    if manifest.get('status')!='complete' or any(manifest.get(k)!=v for k,v in plan.items()):raise ValueError('Incomplete or inconsistent feature records')
    for name in ('features.pt','split.npz'):
        if sha(records/name)!=manifest['artifacts'][name]:raise ValueError(f'Feature artifact changed: {name}')
    features=torch.load(records/'features.pt',map_location='cpu',weights_only=True)
    n=sum(plan['counts']);tokens=plan['tokens'];positions=features['positions']
    if features['indices'].tolist()!=plan['board_ids'] or len(set(plan['board_ids']))!=n:raise ValueError('Feature puzzle split differs')
    if positions.shape!=(n,tokens) or not np.array_equal(positions.numpy()+1,plan['positions']):raise ValueError('Feature positions differ')
    if ((positions<0)|(positions>80)).any() or any(len(set(row))!=tokens for row in positions.tolist()):raise ValueError('Invalid sampled positions')
    with np.load(records/'split.npz',allow_pickle=False) as split:
        if not np.array_equal(split['indices'],features['indices']) or not np.array_equal(split['positions'],positions+1) or not np.array_equal(split['split'],np.repeat(['train','validation','test'],plan['counts'])):raise ValueError('Saved split differs')
    width=features['x'].shape[-1]
    for key in ['x']+[f'{step}/{name}' for step in plan['steps'] for name in ('h','l','next_l','next_h','bare_next_h','bare_old_h')]:
        value=features[key]
        if value.shape!=(n,tokens,width) or not torch.isfinite(value).all():raise ValueError(f'Invalid feature: {key}')
    if features['is_blank'].shape!=(n,tokens) or features['is_blank'].dtype!=torch.bool:raise ValueError('Invalid blank mask')
    return plan,features


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/memory/records/old_h'))
    p.add_argument('--records',type=Path,default=root/'features');p.add_argument('--output-dir',type=Path,default=root/'decoders')
    p.add_argument('--seed',type=int);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--bootstrap-repeats',type=int,default=1000);p.add_argument('--device',default='cuda')
    a=p.parse_args(argv)
    if a.epochs<0 or a.bootstrap_repeats<1:p.error('Invalid fitting sizes')
    source,features=load_features(a.records)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
    seed=a.seed if a.seed is not None else int(time.time())
    plan=dict(seed=seed,feature_manifest_sha256=sha(a.records/'manifest.json'),feature_records=str(a.records.resolve()),
        counts=source['counts'],tokens=source['tokens'],steps=source['steps'],epochs=a.epochs,
        bootstrap_repeats=a.bootstrap_repeats,device=a.device,
        code_sha256={str(f):sha(Path(f)) for f in (__file__,fit.__code__.co_filename)})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'run_config.json').open('x') as f:json.dump(plan,f,indent=2)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=fit('checkpoint',a.output_dir,features,source['counts'],source['tokens'],source['steps'],a.device,seed,a.epochs,a.bootstrap_repeats)
    if sha(a.records/'features.pt')!=json.loads((a.records/'manifest.json').read_text())['artifacts']['features.pt']:raise RuntimeError('Input features changed')
    report=dict(plan,status='complete',rows=len(rows),artifacts={f.name:sha(f) for f in a.output_dir.iterdir() if f.suffix in ('.pt','.npz') or f.name=='results.json'},
        note='Position means/scales use training puzzles only; ridge and residual MLP use validation selection. Conditional confidence intervals resample held-out puzzles, not backbone training seeds. Decoder initializations are recorded seed substreams.')
    (a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
