"""Fit the Mixer geometric lift and score grids with the frozen temporal probe."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm.paths import repository_root
from hrm.analyse.dynamics.constraint_landscape.global_fit import fit_global
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .probe import metric


def checked(directory, artifact, hash_key):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    plan = json.loads((directory/'run_config.json').read_text())
    if manifest.get('status') != 'complete' or any(manifest.get(k) != v for k,v in plan.items()):
        raise ValueError(f'Incomplete or changed plan: {directory}')
    if sha(directory/artifact) != manifest[hash_key]:
        raise ValueError(f'Changed input: {directory/artifact}')
    with np.load(directory/artifact, allow_pickle=False) as f:
        arrays = {k:f[k] for k in f.files}
    return manifest, arrays


def ranges(q, phase, tail_size=1024):
    span = np.ptp(q,axis=0)
    result = {'global':(q.min(0)-.15*span,q.max(0)+.15*span)}
    for h in sorted(set(phase.tolist())):
        tail = q[phase == h][-tail_size:]
        center = tail.mean(0); half = np.maximum(np.ptp(tail,axis=0)*.65,1e-5)
        result[f'H{h}'] = (center-half,center+half)
    return result


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()))


@torch.inference_mode()
def run(geometry, probe, records, output, seed=None, device='cuda', centers=1024,
        global_resolution=401, local_resolution=201, batch_size=256, tail_size=1024):
    geometry,probe,records,output = map(Path,(geometry,probe,records,output))
    if min(centers,batch_size,tail_size) < 1 or min(global_resolution,local_resolution) < 2:
        raise ValueError('Invalid fit/grid sizes')
    gm,d = checked(geometry,'prepared.npz','prepared_sha256')
    pm,p = checked(probe,'probe.npz','probe_sha256')
    replay = json.loads((records/'manifest.json').read_text())
    replay_hash = sha(records/'manifest.json')
    if replay.get('status') != 'complete' or gm['replay_manifest_sha256'] != replay_hash or pm['replay_manifest_sha256'] != replay_hash:
        raise ValueError('Geometry/probe/replay identities differ')
    if gm['h_steps'] != pm['h_steps'] or not np.array_equal(d['phase'],p['phase']) or not np.array_equal(d['step'],p['step']):
        raise ValueError('Geometry and probe phase/step populations differ')
    length = replay['l_steps']+1
    raw = []
    bits = replay['storage'] == 'uint16 BF16 bits'
    for h in range(1,gm['h_steps']+1):
        name = f'H{h:03d}_case{replay["case"]}' + ('_bits.npy' if bits else '.npy')
        if sha(records/name) != replay['trajectory_sha256'][name]:
            raise ValueError('Trajectory changed')
        raw.append(np.load(records/name,mmap_mode='r',allow_pickle=False))
    width = d['B'].shape[0]
    if width % 82 or p['weight'].shape != (width//82,):
        raise ValueError('Probe and latent state widths differ')
    if any(r.shape != (length,82,width//82) or r.dtype != (np.uint16 if bits else np.float32) for r in raw):
        raise ValueError('Invalid trajectory shape/dtype')
    q,coeff,weights,split = [d[k] for k in ('q','coefficients','weights','split')]
    if len(q) != len(raw)*length or set(split.tolist()) != {0,1,2}:
        raise ValueError('Need a complete trajectory and nonempty regression splits')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed = int(time.time()) if seed is None else seed
    plan = dict(seed=seed,geometry_manifest_sha256=sha(geometry/'manifest.json'),
        probe_manifest_sha256=sha(probe/'manifest.json'),replay_manifest_sha256=replay_hash,
        centers=centers,global_resolution=global_resolution,local_resolution=local_resolution,
        batch_size=batch_size,tail_size=tail_size,device=device,case=gm['case'],selection=gm['selection'],
        code_sha256=sha(Path(__file__)),fitter_sha256=sha(Path(__file__).parents[1]/'constraint_landscape/global_fit.py'))
    (output/'run_config.json').write_text(json.dumps(plan,indent=2))
    B,V,base = [torch.as_tensor(d[k],device=device,dtype=torch.float32) for k in ('B','L_V','L_base')]
    weight = torch.as_tensor(p['weight'],device=device,dtype=torch.float64)
    bias = float(p['bias']); sq=width**.5
    def lift(query,c):
        return base+sq*(torch.as_tensor(query,device=device,dtype=torch.float32)@B.T+
                       torch.as_tensor(c[:,:V.shape[1]],device=device,dtype=torch.float32)@V.T)
    def score(z):
        states=z.reshape(-1,82,width//82).to(torch.bfloat16 if bits else torch.float32)
        return (states.float().mean(1).double()@weight+bias).cpu().numpy()
    def native(indices):
        a=np.stack([raw[i//length][i%length] for i in indices])
        z=torch.from_numpy(a)
        if bits:z=z.view(torch.bfloat16)
        return z.to(device).float().flatten(1)
    tr,va,te=[np.flatnonzero(split == i) for i in (0,1,2)]
    model,selection=fit_global(q,coeff,tr,va,weights,device=device,seed=seed,n_centers=centers)
    model.save(output/'validation_model.pt')
    tc=model.predict(q[te]);errors=[];project=[];pred=[]
    for start in range(0,len(te),batch_size):
        ix=te[start:start+batch_size];z=lift(q[ix],tc[start:start+len(ix)])
        errors.extend((z-native(ix)).square().mean(1).sqrt().cpu().numpy())
        pred.extend(score(z))
        project.extend(np.linalg.norm(((z-torch.as_tensor(d['origin'],device=device))@B/sq).cpu().numpy()-q[ix],axis=1))
    native_probe=p['predicted']
    validation=dict(L_rms=stats(errors),probe_comparison=metric(native_probe[te],np.asarray(pred)),projection_error=stats(project))
    del model
    model,refit=fit_global(q,coeff,np.arange(len(q)),[],weights,device=device,seed=seed,selected=selection)
    model.save(output/'kernel_model.pt')
    displayed=[]
    for start in range(0,len(q),batch_size):
        qi=q[start:start+batch_size];displayed.extend(score(lift(qi,model.predict(qi))))
    displayed=np.asarray(displayed)
    np.savez_compressed(output/'trajectory_audit.npz',q=q,phase=d['phase'],step=d['step'],
                        native_probe=native_probe,lift_probe=displayed,target=p['target'],
                        probe_split=p['split'],regression_test_indices=te,regression_test_probe=pred)
    regions={}
    for name,(lo,hi) in ranges(q,d['phase'],tail_size).items():
        n=global_resolution if name=='global' else local_resolution
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],n),np.linspace(lo[1],hi[1],n))
        query=np.c_[xx.ravel(),yy.ravel()];values=[]
        for start in range(0,len(query),batch_size):
            qi=query[start:start+batch_size];values.extend(score(lift(qi,model.predict(qi))))
        values=np.asarray(values).reshape(xx.shape)
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite grid scores')
        np.savez_compressed(output/f'grid_{name}.npz',x=xx,y=yy,probe=values)
        regions[name]=dict(min=float(values.min()),max=float(values.max()),negative_count=int((values<0).sum()),
                           bounds=[lo.tolist(),hi.tolist()],shape=list(values.shape))
    report=dict(plan,status='complete',selection_fit=selection,display_refit=refit,validation=validation,
        display_probe_comparison=metric(native_probe,displayed),
        lift_to_target_test=metric(p['target'][p['split']==2],displayed[p['split']==2]),regions=regions,
        artifact_sha256={f.name:sha(f) for f in sorted(output.iterdir()) if f.suffix in ('.pt','.npz')},
        limitation='Transductive geometric basis; temporal interpolation test, not independent puzzles. '
                   'Geometry refits all states for display; constraint probe remains training-block-only. '
                   'Off-trajectory scores are unconstrained smooth extensions, not measured ground truth.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main(argv=None):
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/mixer'))
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('geometry',root/'geometry'),('probe',root/'probe'),('records',root/'selected'),('output-dir',root/'landscape')]:
        p.add_argument('--'+name,type=Path,default=default)
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda')
    for name,value in [('centers',1024),('global-resolution',401),('local-resolution',201),('batch-size',256),('tail-size',1024)]:
        p.add_argument('--'+name,type=int,default=value)
    a=p.parse_args(argv);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    print(json.dumps(run(a.geometry,a.probe,a.records,a.output_dir,a.seed,a.device,a.centers,
                         a.global_resolution,a.local_resolution,a.batch_size,a.tail_size),indent=2))


if __name__=='__main__':main()
