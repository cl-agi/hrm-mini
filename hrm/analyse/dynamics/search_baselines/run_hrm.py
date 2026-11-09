"""E31 frozen HRM micro-update trajectories on the same public Sudoku puzzle."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root,checkpoint_config
from hrm.training.data import local_directory,puzzle_directory
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.jacobian.models import load_config,analysis_architecture,unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records,resolve_test_csv
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
from hrm.analyse.dynamics.answer_direction.train_secondary_hl_readout import model_hash
from .run_sa import check_puzzle


@torch.no_grad()
def trace(model,inputs,cycles):
    """Record initial state and each L/H update; check every native outer cycle."""
    if cycles<1:raise ValueError('Positive inference-cycle horizon required')
    x=model.embed(inputs);shape=(*inputs.shape,model.zH_init.shape[-1])
    h=model.zH_init.expand(shape);l=model.zL_init.expand(shape)
    hs=[];ls=[];predictions=[];operations=[]
    def record(operation):
        hs.append(h[0].float().cpu().numpy().copy());ls.append(l[0].float().cpu().numpy().copy())
        predictions.append((model.lm_head(h)[0,1:,1:].argmax(-1)+1).cpu().numpy());operations.append(operation)
    record('initial')
    for _ in range(cycles):
        start=dict(z_H=h.clone(),z_L=l.clone())
        for _ in range(model.H_cycles):
            for _ in range(model.L_cycles):l=model.L_level(l+h+x);record('L')
            h=model.H_level(h+l);record('H')
        native,_=model(start,inputs)
        if not torch.equal(h,native['z_H']) or not torch.equal(l,native['z_L']):raise RuntimeError('Micro-update trajectory differs from native forward')
    return dict(zH=np.stack(hs),zL=np.stack(ls),predictions=np.stack(predictions),operations=np.asarray(operations))


def distances(h,l):
    # Sum in float32 as in the original extractor, reduce distances in float64.
    states=(h.astype(np.float32)+l.astype(np.float32)).reshape(len(h),-1).astype(np.float64)
    sq=np.square(states).sum(1);d=np.sqrt(np.maximum(sq[:,None]+sq[None,:]-2*states@states.T,0))
    np.fill_diagonal(d,0);scale=float(d.max())
    return d/scale if scale>0 else np.zeros_like(d),scale


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/search'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--board-id',type=int,default=2);p.add_argument('--cycles',type=int,default=16)
    p.add_argument('--device',default='cuda');p.add_argument('--output-dir',type=Path,default=root/'hrm');p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.cycles<1:p.error('Positive inference-cycle horizon required')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    if (a.output_dir/'started.json').exists():raise FileExistsError(a.output_dir)
    cp=hub.checkpoint('sudoku',a.checkpoint)
    cfg=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(cfg);name=arch.pop('name','')
    if name not in ('hrm@HRM','hrm.models.hrm@HRM','hrm_ablations@HRM','hrm.models.ablations@HRM') or not arch.get('hh_link',True) or not arch.get('dual_module',True) or arch.get('sym_io',False) or arch.get('is_mlp_mixer',False) or arch.get('rope_steps') is not None:raise ValueError('E31 requires a standard full-link dual-module attention HRM')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6):raise ValueError('Sudoku 2H6L checkpoint required')
    data=local_directory(a.dataset) if a.dataset else puzzle_directory('sudoku');records=read_test_records(data)
    if not 0<=a.board_id<len(records):raise ValueError('Board ID outside the test split')
    row=records[a.board_id];puzzle=np.array([int(c) if c!='.' else 0 for c in row.question]);check_puzzle(puzzle)
    plan=dict(board_id=a.board_id,puzzle=row.question,target=row.answer,dataset_sha256=sha(resolve_test_csv(data)),
        checkpoint_sha256=sha(cp),checkpoint=str(cp),config=cfg,cycles=a.cycles,device=a.device,
        schedule='Each inference cycle has two H phases, each six L updates then one H update',
        readout='Native H readout; exclude BOS and class 0; no lookahead or feedback',code_sha256=sha(Path(__file__)))
    saved=a.output_dir/'run_config.json'
    if saved.exists():
        if json.loads(saved.read_text())!=plan:raise ValueError('Prepared settings changed; use a new directory')
    else:
        if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
        a.output_dir.mkdir(parents=True,exist_ok=True);saved.write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (a.output_dir/'started.json').open('x') as f:json.dump(dict(plan_sha256=sha(saved)),f)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    before=model_hash(model);inputs=torch.tensor(np.pad(puzzle,(1,0))[None],dtype=torch.long,device=a.device)
    result=trace(model,inputs,a.cycles);target=np.array(list(map(int,row.answer)))
    successes=np.flatnonzero((result['predictions']==target).all(-1));first=int(successes[0]) if len(successes) else None
    end=first if first is not None else len(result['operations'])-1
    distance,scale=distances(result['zH'][:end+1],result['zL'][:end+1])
    times=np.arange(end+1);ix=np.array([np.argmin(np.abs(times-t)) for t in np.linspace(0,end,99)])
    if before!=model_hash(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen model changed')
    np.savez_compressed(a.output_dir/'trajectory.npz',puzzle=puzzle,target=target,**result,window_end=end,
        sampled_indices=ix,distance=distance[np.ix_(ix,ix)])
    report=dict(status='complete',native_forward_parity=True,first_success=first,window_ends_at_solution=first is not None,
        window_end=end,total_states=len(result['operations']),unique_sampled_frames=len(set(ix.tolist())),
        distance_scale=scale,zero_distance_trajectory=scale==0,plan_sha256=sha(saved),
        unchanged_backbone_sha256=before,artifacts={'trajectory.npz':sha(a.output_dir/'trajectory.npz')},
        note='Distance uses all tokens of zH+zL, normalized by the maximum over the entire selected prefix. Horizon failures are not successful trajectories; this fresh case need not reproduce the archived 98-update solution.')
    (a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
