"""Compute full-state MSD curves from Sudoku checkpoints and public puzzles."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.models.hrm import HRM
from hrm.models.trm import TRM
from hrm.models.rt import RecurrentTransformer
from hrm.models.ablations import HRM as AblationHRM
from hrm.analyse.dynamics.jacobian.models import analysis_architecture, load_config, unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.training.data import puzzle_directory
from .analyze import sha, write_csv, SERIES, EVAL_L, main as analyze

BOUNDARIES = [0,48,192,768,4096]
LAGS = [1,2,3,4,5,6,7,8,9,10,13,14,15,17,19,20,22,26,27,28,30,36,38,44,45,47,53,69,74,76,103,106,130,143,161,223,246,376,383,575,657,1128,1938,3327]


def build_model(config, state, device):
    arch=analysis_architecture(config);name=arch.pop('name','hrm@HRM')
    module=name.split('@')[0].removeprefix('arch.').removeprefix('hrm.models.')
    if arch.get('rope_steps') is not None or arch.get('is_mlp_mixer',False) or (arch['seq_len'],arch['vocab_size'])!=(82,10):
        raise ValueError('Use stationary attention Sudoku checkpoints')
    if module=='rt':kind='rt';cls=RecurrentTransformer
    elif module in ('hrm','trm','hrm_ablations','ablations'):
        if not arch.get('hh_link',True) or arch.get('sym_io',False):raise ValueError('Use the additive H/L recurrence')
        if arch.get('L_cycles') not in (1,6):raise ValueError('Supported L update intervals are 1 and 6')
        tied=module=='trm' or not arch.get('dual_module',True)
        if tied and arch['L_cycles']!=6:raise ValueError('TRM requires L_cycles=6 for this comparison')
        kind='trm' if tied else 'hrm_h2l1' if arch['L_cycles']==1 else 'hrm'
        if arch.get('H_cycles')!=2:raise ValueError('Use H_cycles=2 for this comparison')
        cls=TRM if module=='trm' else AblationHRM if module in ('hrm_ablations','ablations') or any(k in arch for k in ('dual_module','hh_link','sym_io')) else HRM
    else:raise ValueError(f'Unsupported MSD architecture: {name}')
    model=cls(arch).to(device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(state),strict=True)
    return model,kind


@torch.inference_mode()
def states(model, inputs, updates, kind):
    """State index zero is initialization; each subsequent index advances L/core once."""
    embedding=model.embed(inputs)
    if kind=='rt':
        z=model.z_init.expand_as(embedding).clone();yield {'rt':z.float().cpu().numpy()}
        for _ in range(updates):
            z=model.core(z+embedding);yield {'rt':z.float().cpu().numpy()}
    else:
        h=model.zH_init.expand_as(embedding).clone();l=model.zL_init.expand_as(embedding).clone()
        yield {'h':h.float().cpu().numpy(),'l':l.float().cpu().numpy()}
        for step in range(1,updates+1):
            l=model.L_level(l+h+embedding)
            if step%model.L_cycles==0:h=model.H_level(h+l)
            yield {'h':h.float().cpu().numpy(),'l':l.float().cpu().numpy()}


def window_msd(history, lag, chunk=32):
    """Mean squared displacement over origins, puzzles, tokens and coordinates."""
    origins=len(history)-lag
    if lag<1 or origins<1:raise ValueError('Lag must lie inside the window')
    total=0.;count=0
    for start in range(0,origins,chunk):
        end=min(start+chunk,origins)
        delta=history[start+lag:end+lag]-history[start:end]
        total+=np.square(delta).sum(dtype=np.float64);count+=delta.size
    return total/count


def collect(model, loader, boundaries, lags, kind, device):
    totals={};puzzles=0
    for inputs,_,_ in loader:
        stream=states(model,inputs.to(device),boundaries[-1],kind)
        for segment,(start,end) in enumerate(zip(boundaries,boundaries[1:]),1):
            first=next(stream)
            buffers={state:np.empty((end-start,*value.shape),dtype=np.float32) for state,value in first.items()}
            for state,value in first.items():buffers[state][0]=value
            for offset in range(1,end-start):
                for state,value in next(stream).items():buffers[state][offset]=value
            for state,history in buffers.items():
                for lag in lags:
                    if lag<end-start:
                        key=state,segment,lag
                        totals[key]=totals.get(key,0.)+len(inputs)*window_msd(history,lag)
            del buffers,history,first
        next(stream)  # Complete the final update at the exclusive horizon.
        puzzles+=len(inputs)
    if not puzzles:raise ValueError('Empty selected cohort')
    return {key:value/puzzles for key,value in totals.items()},puzzles


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=repository_root()/'exp/dynamics/records/msd')
    p.add_argument('--seed',type=int,help='Puzzle sampling seed, frozen during preparation')
    p.add_argument('--boards',type=int,default=256);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--boundaries',type=int,nargs='+',default=BOUNDARIES)
    p.add_argument('--device',default='cuda');p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if min(a.boards,a.batch_size)<1:p.error('Positive cohort and batch sizes required')
    bounds=a.boundaries
    if len(bounds)!=5 or bounds[0]!=0 or any(b-a<8 for a,b in zip(bounds,bounds[1:])):p.error('Use five increasing boundaries starting at zero; windows need at least eight states')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    out=a.output_dir;old=json.loads((out/'run_config.json').read_text()) if (out/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else old['sampling_seed'] if old else int(time.time())
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    records=read_test_records(a.dataset or puzzle_directory('sudoku'))
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=None)
    lags=[lag for lag in LAGS if lag<max(np.diff(bounds))]
    plan=dict(sampling_seed=seed,board_ids=list(ids),checkpoint=str(cp),checkpoint_sha256=sha(cp),config=config,
        dataset_sha256=dataset_sha256(records),boundaries=bounds,lags=lags,batch_size=a.batch_size,device=a.device,
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('collect.py','analyze.py')})
    if old is not None and old!=plan:raise ValueError('Settings changed; use a new output directory')
    if old is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True);(out/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    model,kind=build_model(config,torch.load(cp,map_location='cpu',weights_only=True),a.device)
    with (out/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    values,count=collect(model,build_loader(records,ids,a.batch_size),bounds,lags,kind,a.device)
    runs=out/'rollouts';runs.mkdir()
    # Checkpoint identity defines the replication unit; sampling is recorded separately.
    identity=plan['checkpoint_sha256'];training_seeds=config.get('seeds',[])
    meta=dict(protocol='fresh-v1',condition=kind,run_id=identity,seed=training_seeds[0] if len(training_seeds)==1 else '',
        samples=count,eval_l=EVAL_L[kind],best_epoch='',actual_l_updates=bounds[-1],
        segment_boundaries_l_updates=json.dumps(bounds),lag_l_updates=json.dumps(lags),sampling_seed=seed,
        checkpoint_sha256=identity,dataset_sha256=plan['dataset_sha256'])
    write_csv(runs/'rollout_metadata.csv',[meta])
    rows=[dict(protocol='fresh-v1',condition=kind,run_id=identity,seed=meta['seed'],state=state,segment=segment,
        lag_l_updates=lag,origins=bounds[segment]-bounds[segment-1]-lag,mean=value,
        eval_l=EVAL_L[kind],puzzles=count,status='defined') for (state,segment,lag),value in sorted(values.items())]
    write_csv(runs/'msd_curves.csv',rows)
    profile='standard' if bounds==BOUNDARIES else 'available'
    analyze(['--records',str(runs),'--output-dir',str(out/'statistics'),'--profile',profile])
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',condition=kind,plan=plan,
        outputs={str(f.relative_to(out)):sha(f) for f in [runs/'rollout_metadata.csv',runs/'msd_curves.csv',out/'statistics/analysis_manifest.json']}),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
