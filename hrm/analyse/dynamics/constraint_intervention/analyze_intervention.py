"""Summarize paired fractional-score intervention effects across puzzles."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp
from hrm.paths import repository_root
from hrm.analyse.dynamics.constraint_readout.run import finite


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=repository_root()/'exp/error_readout/records/sudoku_constraints')
    a=p.parse_args(argv);data=json.loads((a.records/'results.json').read_text())
    records=data['observations'];ids=sorted({r['row_id'] for r in records});rows=[]
    rng=np.random.default_rng(data['config']['seed'])
    for dose in sorted({r['arm']['target_fraction'] for r in records if r['arm']['kind']=='critic'}):
        effects=[]
        for i in ids:
            selected=[r for r in records if r['row_id']==i and r['arm']['target_fraction']==dose]
            critic=[r['step_norm'] for r in selected if r['arm']['kind']=='critic']
            controls=[r['step_norm'] for r in selected if r['arm']['kind']=='control']
            if len(critic)!=1 or len(controls)!=8:raise ValueError('Each puzzle/dose requires one critic arm and eight controls')
            effects.append(np.array(critic[0])-np.mean(controls,axis=0))
        effect=np.asarray(effects);draw=rng.integers(0,len(ids),(10000,len(ids)))
        rows.append(dict(dose_percent=100*dose,mean=float(effect[:,0].mean()),
                         ci=np.quantile(effect[draw,0].mean(1),[.025,.975]).tolist(),
                         p=float(ttest_1samp(effect[:,0],0).pvalue) if len(ids)>1 else None,
                         second_L_mean=float(effect[:,1].mean())))
    ordered=sorted([i for i,r in enumerate(rows) if r['p'] is not None and np.isfinite(r['p'])],key=lambda i:rows[i]['p']);last=0
    for k,i in enumerate(ordered):last=max(last,min(1,rows[i]['p']*(len(ordered)-k)));rows[i]['holm_p']=last
    report=dict(n=len(ids),excluded_ids=data['excluded_ids'],rows=rows,config=data['config'])
    (a.records/'intervention_summary.json').write_text(json.dumps(finite(report),indent=2,allow_nan=False))
    print(json.dumps(finite(report)))

if __name__=='__main__':main()
