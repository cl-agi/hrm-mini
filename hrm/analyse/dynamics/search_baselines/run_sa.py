"""Generate fresh E31 simulated-annealing records from public Sudoku data."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from hrm.training.data import local_directory,puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records,resolve_test_csv
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def check_puzzle(puzzle):
    if puzzle.shape!=(81,) or not np.issubdtype(puzzle.dtype,np.integer) or np.any((puzzle<0)|(puzzle>9)):raise ValueError('Expected 81 digits in 0..9')
    grid=puzzle.reshape(9,9)
    units=[*grid,*grid.T,*[grid[r:r+3,c:c+3].flatten() for r in (0,3,6) for c in (0,3,6)]]
    if any(len(u[u!=0])!=len(set(u[u!=0])) for u in units):raise ValueError('Conflicting given clues')


def final_window(trace,boards,frames=99):
    """Nearest recorded states at evenly spaced proposal counts after last restart."""
    starts=np.flatnonzero(trace[:,4]==1);start=int(starts[-1]) if len(starts) else 0
    times=trace[start:,0];wanted=np.linspace(times[0],times[-1],frames)
    right=np.searchsorted(times,wanted).clip(0,len(times)-1);left=np.maximum(right-1,0)
    ix=np.where(np.abs(times[left]-wanted)<=np.abs(times[right]-wanted),left,right)+start
    sampled=boards[ix];distance=(sampled[:,None,:]!=sampled[None,:,:]).mean(-1)
    return dict(record_indices=ix,proposal_counts=trace[ix,0].astype(np.int64),boards=sampled,distance=distance,
        window_start=int(times[0]),window_end=int(times[-1]),unique_frames=int(len(set(ix.tolist()))))


def main(argv=None):
    from .annealing import solve,energy,valid
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/search'))
    p.add_argument('--dataset',type=Path);p.add_argument('--board-id',type=int,default=2)
    p.add_argument('--seeds',type=int,nargs=3);p.add_argument('--budget',type=int,default=10000000)
    p.add_argument('--stride',type=int,default=100);p.add_argument('--output-dir',type=Path,default=root/'sa')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.budget<1 or a.stride<1:p.error('Positive proposal budget and trace stride required')
    if (a.output_dir/'started.json').exists():raise FileExistsError(a.output_dir)
    data=local_directory(a.dataset) if a.dataset else puzzle_directory('sudoku')
    records=read_test_records(data)
    if not 0<=a.board_id<len(records):raise ValueError('Board ID outside the test split')
    r=records[a.board_id];puzzle=np.array([int(c) if c!='.' else 0 for c in r.question],dtype=np.int64)
    check_puzzle(puzzle)
    old=json.loads((a.output_dir/'run_config.json').read_text()) if (a.output_dir/'run_config.json').exists() else None
    seeds=a.seeds or (old['seeds'] if old else [])
    if not seeds:
        for _ in range(3):
            seed=int(time.time())
            while seed in seeds:time.sleep(.05);seed=int(time.time())
            seeds.append(seed)
    if len(set(seeds))!=3 or any(not 0<=s<2**32 for s in seeds):raise ValueError('Three distinct uint32 seeds required')
    plan=dict(seeds=seeds,board_id=a.board_id,puzzle=r.question,target=r.answer,dataset_sha256=sha(resolve_test_csv(data)),
        budget=a.budget,stride=a.stride,calibration_proposals=1000,chain_length=int((puzzle==0).sum())**2,
        cooling=.99,stagnant_chains=20,frames=99,code_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('annealing.py'))})
    if old is not None and old!=plan:raise ValueError('Prepared settings changed; use a new output directory')
    if old is None:
        if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError(a.output_dir)
        a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (a.output_dir/'started.json').open('x') as f:json.dump(dict(unix=time.time()),f)
    target=np.array(list(map(int,r.answer)),dtype=np.int64);summaries=[]
    for seed in seeds:
        start=time.perf_counter();b,trace,boards,steps,reheats,accepted,uphill,t0=solve(puzzle,seed,a.budget,a.stride)
        solved=valid(b,puzzle)
        if solved!=(energy(b)==0):raise ValueError('Independent solution validation disagrees with energy')
        if not np.all(boards[:,puzzle!=0]==puzzle[puzzle!=0]):raise ValueError('Given clues changed')
        window=final_window(trace,boards)
        np.savez_compressed(a.output_dir/f'seed_{seed}.npz',puzzle=puzzle,target=target,final=b,trace=trace,boards=boards,
            sampled_indices=window['record_indices'],sampled_proposals=window['proposal_counts'],distance=window['distance'])
        summaries.append(dict(seed=seed,solved=solved,matches_target=bool(np.array_equal(b,target)),steps=int(steps),
            final_energy=int(energy(b)),reheats=int(reheats),accepted=int(accepted),uphill_accepted=int(uphill),
            initial_temperature=float(t0),seconds_including_jit=time.perf_counter()-start,
            window_start=window['window_start'],window_end=window['window_end'],unique_sampled_frames=window['unique_frames'],
            window_ends_at_solution=solved))
    report=dict(status='complete',plan_sha256=sha(a.output_dir/'run_config.json'),runs=summaries,
        artifacts={p.name:sha(p) for p in a.output_dir.glob('seed_*.npz')},
        note='Fresh public-test row and seeds; not a claim to recover the archived 631535-proposal successful window. Unsolved runs retain budget-censored windows. Distance is normalized 81-cell Hamming.')
    (a.output_dir/'manifest.json').write_text(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
