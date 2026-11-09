"""Fit the original population readout and report its numerical correlation."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from hrm.paths import repository_root
from hrm.analyse.dynamics.constraint_readout.run import finite

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,default=repository_root()/'exp/error_readout/records/sudoku_constraints/population_analysis');root=p.parse_args().records
    parts=[np.load(p) for p in sorted((root/'population').glob('batch_*.npz'))]
    if not parts:raise FileNotFoundError('Run dynamics.constraint_intervention.population first')
    f=np.concatenate([p['features'] for p in parts]);y=np.concatenate([p['target'] for p in parts])
    if f.shape[:2]!=y.shape or f.shape[-1]%2:raise ValueError('Inconsistent feature and target arrays')
    fits={};pred={}
    for name,sl in [('L',slice(0,f.shape[-1]//2)),('HL',slice(None))]:
        x=torch.tensor(f[...,sl].reshape(-1,f[...,sl].shape[-1]),dtype=torch.float64)
        t=torch.tensor(y.ravel(),dtype=torch.float64);mu=x.mean(0);ym=t.mean()
        w=torch.linalg.lstsq(x-mu,(t-ym)[:,None],driver='gelsd').solution[:,0];b=ym-mu@w
        fits[name]=(w,b);pred[name]=(x@w+b).numpy().reshape(y.shape)
    delta=np.concatenate([p['delta'] for p in parts]);norm=np.concatenate([p['norm'] for p in parts]);freeze=np.concatenate([p['freeze_count'] for p in parts])
    torch.save(fits,root/'population_probes.pt')
    np.savez_compressed(root/'population_counts.npz',target=y,predictions=pred['HL'],L_predictions=pred['L'],delta=delta,norm=norm,freeze_count=freeze)
    p=np.maximum(pred['HL'],0);x=delta/norm
    rng=np.random.default_rng(json.loads((root/'config.json').read_text())['seed']);ids=rng.integers(0,len(p),(10000,len(p)))
    a=p-p.mean();b=x-x.mean();s=np.stack([a.sum(1),b.sum(1),(a*a).sum(1),(b*b).sum(1),(a*b).sum(1)],1);z=s[ids].sum(1);n=p.size
    rs=(z[:,4]-z[:,0]*z[:,1]/n)/np.sqrt((z[:,2]-z[:,0]**2/n)*(z[:,3]-z[:,1]**2/n))
    corr=lambda a,b:float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    report=dict(n=len(p),r=corr(p,x),ci=np.quantile(rs,[.025,.975]).tolist(),time_demeaned_r=corr(p-p.mean(0),x-x.mean(0)),norm_mean=float(norm.mean()),norm_sd=float(norm.std()),readout='Joint H/L OLS fitted and scored on the collected population; predictions clipped at zero')
    report=finite(report)
    (root/'population_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
