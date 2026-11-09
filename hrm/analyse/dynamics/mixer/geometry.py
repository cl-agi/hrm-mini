"""Prepare the original terminal-reference plane and transductive Mixer lift basis."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm.paths import repository_root
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from .replay_selected import read_screen, analysis_h_steps


@torch.inference_mode()
def prepare_arrays(states, contexts, seed, normal_rank=256, tail_size=1024):
    """states [H,L+1,D], contexts [H,D]; return score-independent geometry."""
    h_steps, length, width = states.shape
    if h_steps < 2 or length < 2 or width < 3 or normal_rank < 1 or tail_size < 2:
        raise ValueError('Need >=2 phases/states, >=3 features and positive rank')
    if contexts.shape != (h_steps, width) or not torch.isfinite(states).all() or not torch.isfinite(contexts).all():
        raise ValueError('Invalid/nonfinite state geometry')
    torch.manual_seed(seed)
    tail = states[1, -tail_size:]
    mean = tail.mean(0)
    _, _, local = torch.pca_lowrank(tail-mean, q=min(6, *tail.shape), center=False, niter=6)
    vy = states[-1, -1]-mean
    vy -= local[:, :2]@(local[:, :2].T@vy)
    if vy.norm() <= 1e-12:
        raise ValueError('Degenerate terminal-reference direction')
    vy /= vy.norm()
    means = states[:, -tail_size:].mean(1)
    means -= means.mean(0)
    means -= torch.outer(means@vy, vy)
    _, sv, vh = torch.linalg.svd(means, full_matrices=False)
    if sv[0] <= 1e-12:
        raise ValueError('Degenerate phase-mean direction')
    vx = vh[0]
    if torch.dot(means@vx, torch.arange(h_steps, device=states.device, dtype=states.dtype)) < 0:
        vx = -vx
    B = torch.stack([vx, vy], 1)
    source_origin = states[0, 0]
    L = states.flatten(0, 1); sq = width**.5
    q = torch.stack([(L-source_origin)@vx, (L-mean)@vy], 1)/sq
    origin = source_origin-sq*(B@q[0])
    projection_error = float((((L-origin)@B/sq)-q).abs().max())
    if projection_error >= 1e-5:
        raise RuntimeError('Plane coordinate parity check failed')
    mu = L.mean(0)
    _, _, V = torch.pca_lowrank((L-mu)/sq, q=min(normal_rank+16, *L.shape), center=False, niter=3)
    V = V[:, :min(normal_rank, width-2, len(L))]
    uq, rr = torch.linalg.qr(V-B@(B.T@V), mode='reduced')
    ur, sr, _ = torch.linalg.svd(rr, full_matrices=False)
    V = uq@ur[:, sr > 1e-4]
    if V.shape[1] == 0:
        raise ValueError('No residual basis outside the plane')
    V = torch.linalg.qr(V-B@(B.T@V), mode='reduced').Q
    base = mu-B@(B.T@(mu-origin))
    lc = ((L-mu)@V/sq).cpu().numpy()
    hm = contexts.mean(0)
    _, _, hv = torch.linalg.svd((contexts-hm)/sq, full_matrices=False)
    HV = hv[:min(h_steps-1, 5)].T
    hc = ((contexts-hm)@HV/sq).cpu().numpy()
    q = q.cpu().numpy()
    bins = np.clip(((q-q.min(0))/np.maximum(np.ptp(q, axis=0), 1e-9)*255).astype(int), 0, 255)
    _, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    spatial = 1/counts[inv]; weights = .5/len(q)+.5*spatial/spatial.sum()
    steps = np.tile(np.arange(length), h_steps)
    split = np.zeros(len(q), dtype=np.int8)
    split[steps % 19 == 7] = 1; split[steps % 19 == 13] = 2
    split[(steps == 0)|(steps == length-1)] = 0
    values = dict(q=q, phase=np.repeat(np.arange(1,h_steps+1),length), step=steps,
                  coefficients=np.c_[lc, np.repeat(hc,length,axis=0)], weights=weights, split=split)
    values.update({k:v.cpu().numpy() for k,v in dict(L_base=base,L_V=V,B=B,origin=origin,H_mean=hm,H_V=HV).items()})
    oracle = base+sq*(torch.as_tensor(q,device=L.device)@B.T+torch.as_tensor(lc,device=L.device)@V.T)
    rms = (oracle-L).square().mean(1).sqrt()
    return values, dict(coordinate_parity_max_abs=projection_error, rank_L=V.shape[1], rank_H=HV.shape[1],
        oracle_L_rms=dict(mean=float(rms.mean()),max=float(rms.max())),
        B_orthogonality_error=float((B.T@B-torch.eye(2,device=B.device)).abs().max()),
        B_V_cross_error=float((B.T@V).abs().max()))


def run(records, output, screen_records=None, seed=None, device='cuda', normal_rank=256, tail_size=1024):
    records, output = Path(records), Path(output)
    manifest = json.loads((records/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Replay is incomplete')
    frozen = json.loads((records/'run_config.json').read_text())
    if any(manifest.get(k) != v for k,v in frozen.items()):
        raise ValueError('Replay differs from its plan')
    source = Path(screen_records or manifest['screen_records'])
    screen, cohort, _ = read_screen(source)
    if sha(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Screen changed')
    h_steps = analysis_h_steps(manifest)
    if h_steps < 2:
        raise ValueError('At least two phases required')
    index = cohort['ids'].tolist().index(manifest['case'])
    paths = []
    bits = manifest['storage'] == 'uint16 BF16 bits'
    for h in range(1,h_steps+1):
        name = f'H{h:03d}_case{manifest["case"]}' + ('_bits.npy' if bits else '.npy')
        if sha(records/name) != manifest['trajectory_sha256'][name]:
            raise ValueError('Trajectory changed')
        paths.append(records/name)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed = int(time.time()) if seed is None else seed
    plan = dict(seed=seed,replay_manifest_sha256=sha(records/'manifest.json'),h_steps=h_steps,
                normal_rank=normal_rank,tail_size=tail_size,device=device,code_sha256=sha(Path(__file__)),
                case=manifest['case'],selection=manifest['selection'])
    (output/'run_config.json').write_text(json.dumps(plan,indent=2))
    states, contexts = [], []
    width = manifest['config'].get('arch',{}).get('hidden_size',manifest['config'].get('transformer',{}).get('hidden_size'))
    for h,path in enumerate(paths,1):
        raw = np.load(path,allow_pickle=False)
        if raw.shape != (manifest['l_steps']+1,82,width) or raw.dtype != (np.uint16 if bits else np.float32):
            raise ValueError('Invalid trajectory shape/dtype')
        z = torch.from_numpy(raw)
        if bits:z=z.view(torch.bfloat16)
        states.append(z.to(device).float().flatten(1))
        boundary = torch.load(source/f'H{h:03d}.pt',map_location='cpu',weights_only=True)
        contexts.append(boundary['H_before'][index].to(device).float().flatten())
    values, audit = prepare_arrays(torch.stack(states),torch.stack(contexts),seed,normal_rank,tail_size)
    np.savez_compressed(output/'prepared.npz',**values)
    report = dict(plan,**audit,status='complete',prepared_sha256=sha(output/'prepared.npz'),
        pca='All observed states, transductive basis; no constraint targets used',
        split='Step modulo19: validation7/test13; phase endpoints in training',
        weights='Half uniform states, half uniform occupied 256x256 coordinate bins',
        limitation='Near-trajectory interpolation; off-trajectory lift is not uniquely identified')
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main(argv=None):
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/mixer'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=root/'selected');p.add_argument('--output-dir',type=Path,default=root/'geometry')
    p.add_argument('--screen-records',type=Path);p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda')
    p.add_argument('--normal-rank',type=int,default=256);p.add_argument('--tail-size',type=int,default=1024)
    a=p.parse_args(argv);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    print(json.dumps(run(a.records,a.output_dir,a.screen_records,a.seed,a.device,a.normal_rank,a.tail_size),indent=2))


if __name__=='__main__':main()
