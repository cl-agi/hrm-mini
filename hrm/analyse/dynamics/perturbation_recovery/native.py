"""native-schedule Sudoku trajectory and read-only L lookahead."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hrm import hub
from hrm.paths import checkpoint_config, repository_root
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


@torch.inference_mode()
def trajectory(model,x,y,h_steps=8):
    """Read-only H lookahead after each L update; commit H only after six L."""
    if h_steps<2 or h_steps%2:raise ValueError('Use a positive even number of H updates for native-cycle parity')
    emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
    hs=[h.float().cpu().numpy()];ls=[l.float().cpu().numpy()]
    look=[];committed=[];native_checks=0
    for phase in range(1,h_steps+1):
        if phase%2:
            native,_=model(dict(z_H=h.clone(),z_L=l.clone()),x)
        for step in range(1,7):
            l=model.L_level(l+h+emb)
            look.append(model.lm_head(model.H_level(h+l)).argmax(-1)[:,1:].cpu().numpy())
            hs.append(h.float().cpu().numpy());ls.append(l.float().cpu().numpy())
        h=model.H_level(h+l)
        committed.append(model.lm_head(h).argmax(-1)[:,1:].cpu().numpy())
        hs.append(h.float().cpu().numpy());ls.append(l.float().cpu().numpy())
        if phase%2==0:
            if not torch.equal(h,native['z_H']) or not torch.equal(l,native['z_L']):
                raise RuntimeError('Read-only trajectory differs from native model.forward')
            native_checks+=1
    look=np.asarray(look);committed=np.asarray(committed);target=y[:,1:].cpu().numpy()
    errors=(look!=target).sum(-1);herrors=(committed!=target).sum(-1)
    rows=[]
    for i in range(len(x)):
        hit=np.flatnonzero(errors[:,i]==0);hhit=np.flatnonzero(herrors[:,i]==0)
        first=int(hit[0])+1 if len(hit) else None
        rows.append(dict(first_correct_lookahead_L=first,
            first_correct_lookahead_phase=(first-1)//6+1 if first else None,
            first_correct_lookahead_local_L=(first-1)%6+1 if first else None,
            first_correct_committed_H=int(hhit[0])+1 if len(hhit) else None,
            committed_wrong_cells=herrors[:,i].tolist(),lookahead_wrong_cells=errors[:,i].tolist()))
    arrays=dict(H=np.asarray(hs),L=np.asarray(ls),lookahead=look,committed=committed,
        lookahead_errors=errors,committed_errors=herrors,inputs=x.cpu().numpy(),targets=y.cpu().numpy())
    return arrays,dict(rows=rows,native_cycle_bitwise_checks=native_checks,
        state_order='initial, then (L1..L6,H) per H phase; H held during L',
        readout='argmax over all ten classes, excluding BOS; targets used only after recurrence')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/recurrence'))/'native')
    p.add_argument('--seed',type=int);p.add_argument('--boards',type=int,default=1);p.add_argument('--board-ids',type=int,nargs='+')
    p.add_argument('--h-steps',type=int,default=8)
    p.add_argument('--device',default='cuda');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if a.boards<1 or a.h_steps<2 or a.h_steps%2:p.error('Positive boards and even H steps >=2 required')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    output=a.output_dir;old=json.loads((output/'run_config.json').read_text()) if (output/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else (old['seed'] if old else int(time.time()))
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config)
    if arch.pop('name','') not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False):p.error('Standard attention HRM required')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6) or arch.get('rope_steps') is not None:p.error('Stationary Sudoku HRM with native 2H6L required')
    records=read_test_records(a.dataset if a.dataset else puzzle_directory('sudoku'))
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=a.board_ids)
    plan=dict(seed=seed,board_ids=list(ids),selection='explicit row IDs' if a.board_ids else 'random sample, no outcome selection',
              checkpoint=str(cp),checkpoint_sha256=sha(cp),config=config,dataset_sha256=dataset_sha256(records),
              h_steps=a.h_steps,device=a.device,deterministic_algorithms=True,torch_seed=seed,code_sha256=sha(Path(__file__)))
    if old is not None and old!=plan:raise ValueError('Changed settings/input; use a new output directory')
    if old is None:
        if output.exists() and any(output.iterdir()):raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True);(output/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (output/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    x,y,_=next(iter(build_loader(records,ids,len(ids))));x=x.to(a.device);y=y.to(a.device)
    arrays,summary=trajectory(model,x,y,a.h_steps)
    arrays['board_ids']=np.asarray(ids)
    np.savez_compressed(output/'trajectory.npz',**arrays)
    report=dict(plan,status='complete',**summary,
        artifact_sha256={'trajectory.npz':sha(output/'trajectory.npz')},
        limitation='Fresh or explicitly selected case(s); historical 31/9/1/0 errors and first-correct timing are not guaranteed')
    (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
